# app/streamlit_app.py
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageFilter

# Ensure the repository root is on sys.path so sibling package `src` can be imported
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.predict import preprocess_pil, load_trained_model, predict_tensor
from src.utils import ensure_dir

# Optional dependency for drawing. If it's not installed, provide a graceful fallback.
# pip install streamlit-drawable-canvas
try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    st_canvas = None


DEFAULT_WEIGHTS = REPO_ROOT / "models" / "mnist_cnn.pt"


st.set_page_config(page_title="MNIST Classifier (Techy)", layout="wide")

st.title("🧠 MNIST Classifier — From Scratch (CNN) + Demo")
st.caption("Dibuja un dígito o sube una imagen. El modelo predice 0–9 y muestra probabilidades.")


@st.cache_resource
def _load_model_cached(weights_path: str):
    model, device = load_trained_model(weights_path=weights_path, out_1=16, out_2=32)
    return model, device


def add_noise(img: Image.Image, sigma: float) -> Image.Image:
    if sigma <= 0:
        return img
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, size=arr.shape).astype(np.float32)
    arr2 = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr2)


def ensure_weights_exist():
    if DEFAULT_WEIGHTS.exists():
        return True
    st.warning("No encuentro pesos entrenados en `models/mnist_cnn.pt`. Entrena primero con `python -m src.train`.")
    return False


left, right = st.columns([1, 1])

with left:
    st.subheader("1) Entrada")
    mode = st.radio("Elige modo:", ["🖊️ Dibujar", "🖼️ Subir imagen"], horizontal=True)

    noise_sigma = st.slider("Ruido (para probar robustez)", 0.0, 60.0, 0.0, 1.0)

    pil_img = None

    if mode == "🖊️ Dibujar":
        st.write("Dibuja un dígito (0–9). Ideal: trazo grueso y centrado.")
        if st_canvas is None:
            st.warning("Para usar el modo dibujo instala: pip install streamlit-drawable-canvas")
            uploaded_fallback = st.file_uploader("No tienes el componente de dibujo: sube una imagen en su lugar", type=["png", "jpg", "jpeg"])
            if uploaded_fallback is not None:
                pil_img = Image.open(uploaded_fallback)
        else:
            canvas = st_canvas(
                fill_color="black",
                stroke_width=18,
                stroke_color="white",
                background_color="black",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )

            if canvas.image_data is not None:
                # canvas image_data is RGBA float array 0..255
                img_arr = canvas.image_data.astype(np.uint8)
                pil_img = Image.fromarray(img_arr).convert("L")

    else:
        uploaded = st.file_uploader("Sube una imagen con un dígito (ideal fondo negro, dígito blanco)", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            pil_img = Image.open(uploaded)

    if pil_img is not None:
        # Make MNIST-like (invert if needed): MNIST is white digit on black background
        pil_img = pil_img.convert("L")
        # Heuristic: if background seems white, invert
        if np.mean(np.array(pil_img)) > 127:
            pil_img = ImageOps.invert(pil_img)

        pil_img = add_noise(pil_img, noise_sigma)

        st.write("Vista previa (antes de 28x28):")
        st.image(pil_img, use_container_width=True)

with right:
    st.subheader("2) Predicción")
    if pil_img is None:
        st.info("Primero dibuja o sube una imagen.")
    else:
        if not ensure_weights_exist():
            st.stop()

        model, device = _load_model_cached(str(DEFAULT_WEIGHTS))

        x = preprocess_pil(pil_img)  # [1,1,28,28]
        pred, probs = predict_tensor(model, x, device)

        st.markdown(f"## ✅ Predicción: **{pred}**")
        top3 = np.argsort(probs)[::-1][:3]
        st.write("Top-3:", ", ".join([f"{i} ({probs[i]:.2%})" for i in top3]))

        st.write("Input final 28x28 (lo que ve el modelo):")
        x_img = (x.squeeze(0).squeeze(0).numpy() * 255).astype(np.uint8)
        st.image(x_img, clamp=True, width=220)

        st.write("Probabilidades por clase:")
        st.bar_chart(probs)

st.divider()
st.subheader("📌 Tips (para el workshop)")
st.markdown(
    """
- Si el modelo falla, prueba con un dígito **centrado** y trazos **gruesos**.
- El slider de ruido te muestra **robustez**: el modelo no es magia.
- Para portafolio: sube un screenshot de tu demo a `assets/demo.png` y completa el README.
"""
)
