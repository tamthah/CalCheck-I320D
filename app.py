"""CalCheck · single-image nutrition estimates using the team's frozen MobileNetV2."""
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st
import tensorflow as tf
from tensorflow import keras

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "calcheck_MobileNetV2_frozen.keras"
CSV_PATH = HERE / "calcheck_dishes_with_validation.csv"
TARGET_COLUMNS = ["calories_kcal", "protein_g", "carbs_g", "fat_g"]

st.set_page_config(page_title="CalCheck · Food Nutrition Estimates", page_icon="🍽️", layout="centered")
st.markdown("""
<style>
.block-container {max-width: 850px; padding-top: 2.4rem;}
.hero {background:linear-gradient(115deg,#153b39,#28665d); border-radius:20px; padding:30px 32px; color:white; margin-bottom:22px;}
.hero h1 {font-size:2.5rem; font-weight:750; margin:0 0 8px; color:#fff;}
.hero p {font-size:1.04rem; opacity:.92; margin:0; color:#fff;}
.eyebrow {letter-spacing:2px; font-size:.73rem; font-weight:700; color:#c8e9da; margin-bottom:8px;}
.model-note {font-size:.82rem; color:#687c78; padding:6px 0 14px;}
</style>
""", unsafe_allow_html=True)
st.markdown('''<div class="hero"><div class="eyebrow">I320D · UNSUPERVISED LEARNERS</div>
<h1>🍽️ CalCheck</h1><p>Upload a food photo. Get an estimate of calories and macronutrients in seconds.</p></div>''', unsafe_allow_html=True)
st.markdown("**How it works:** upload one food photo → CalCheck analyzes it using our frozen MobileNetV2 → see four estimates.")
st.caption("Model: frozen MobileNetV2 · Dataset: Nutrition5k · Overhead RGB images · 224 × 224 input")

@st.cache_resource(show_spinner="Loading our food recognition model…")
def get_model():
    # No custom training loss is required for inference.
    model = keras.models.load_model(MODEL_PATH, compile=False)
    if tuple(model.input_shape[1:]) != (224, 224, 3) or model.output_shape[-1] != 4:
        raise ValueError("Unexpected model input/output shape. Check the selected checkpoint.")
    return model

@st.cache_data(show_spinner=False)
def get_train_stats():
    df = pd.read_csv(CSV_PATH)
    required = TARGET_COLUMNS + ["split"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError("CSV lacks columns: " + ", ".join(missing))
    train = df.loc[df["split"].eq("train"), TARGET_COLUMNS]
    if train.empty or train.isna().any().any():
        raise ValueError("Invalid training labels in dataset CSV.")
    # Pandas .std() uses sample std (ddof=1), exactly as in team training notebook.
    mean = train.mean().to_numpy(dtype=np.float32)
    std = train.std().to_numpy(dtype=np.float32)
    if np.any(std <= 0):
        raise ValueError("Invalid standard deviation in training labels.")
    return mean, std

def prepare_photo(photo):
    # Match training: decode RGB -> tf.image.resize_with_pad to 224x224 -> raw [0,255] float32.
    # The model already contains mobilenet_v2.preprocess_input: do NOT divide by 255 here.
    photo = ImageOps.exif_transpose(photo).convert("RGB")
    pixels = np.asarray(photo, dtype=np.uint8)
    tensor = tf.image.resize_with_pad(tf.convert_to_tensor(pixels), 224, 224)
    return tf.expand_dims(tf.cast(tensor, tf.float32), axis=0)

st.subheader("1 · Upload your meal")
photo_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], help="Try an overhead photo similar to the Nutrition5k images used during training.")
if photo_file:
    try:
        # Guard against unusually large files.
        if photo_file.size > 15 * 1024 * 1024:
            st.error("Please upload an image smaller than 15 MB.")
            st.stop()
        photo = Image.open(BytesIO(photo_file.getvalue()))
        photo.verify()
        photo = Image.open(BytesIO(photo_file.getvalue()))
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("The selected file could not be opened as an image. Try another JPG or PNG.")
        st.stop()
    st.image(photo, caption="Your uploaded food photo", use_container_width=True)
    if st.button("Estimate nutrition", type="primary", use_container_width=True):
        if not MODEL_PATH.exists() or not CSV_PATH.exists():
            st.error("Required model/CSV file is missing. Put the frozen .keras file and validation CSV beside app.py.")
            st.stop()
        try:
            with st.spinner("Estimating nutritional content…"):
                model = get_model()
                means, stds = get_train_stats()
                scaled = model.predict(prepare_photo(photo), verbose=0)
                if scaled.shape != (1, 4):
                    raise ValueError(f"Unexpected model prediction shape: {scaled.shape}")
                raw = scaled[0] * stds + means
                if not np.all(np.isfinite(raw)):
                    raise ValueError("The model returned an invalid prediction.")
                results = np.maximum(raw, 0)  # Display only; evaluation uses unclipped values.
            st.subheader("2 · Estimated nutrition")
            a, b = st.columns(2)
            with a:
                st.metric("🔥 Calories", f"{results[0]:,.0f} kcal")
                st.metric("🍞 Carbohydrates", f"{results[2]:.1f} g")
            with b:
                st.metric("💪 Protein", f"{results[1]:.1f} g")
                st.metric("🥑 Fat", f"{results[3]:.1f} g")
            st.success("Prediction complete")
            st.caption("Values are estimates, not measurements. Negative raw model outputs, if any, are shown as zero for display only.")
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")
else:
    st.info("Upload a JPG or PNG to start. You can use a photo from the Nutrition5k overhead_rgb folder.")

st.divider()
st.subheader("Important limitations")
st.warning("**Nutrition estimates only.** CalCheck uses one overhead RGB image and cannot accurately measure food weight or depth, see hidden ingredients, or determine cooking oils and preparation methods. Different lighting, camera angles, and portion sizes can affect results. Do not treat these estimates as verified nutritional information.")
st.caption("Built for a UT Austin I320D class project · CalCheck / Unsupervised Learners")
