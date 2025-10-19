import os
import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from utils import maybe_resize
from denoise import (
    gaussian_filter,
    median_filter,
    nlm_filter_colored,
    unsharp_mask,
)


st.set_page_config(page_title="Image Denoising Studio", layout="wide")
st.title("Image Denoising Studio")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    max_size = st.number_input("Max size (px)", min_value=256, max_value=4096, value=1024, step=64)
    apply_unsharp = st.checkbox("Unsharp mask", value=False)
    unsharp_amount = st.slider("Unsharp amount", 0.0, 2.0, 0.2, 0.05)
    unsharp_sigma = st.slider("Unsharp sigma", 0.1, 3.0, 1.0, 0.1)

    st.header("Noise Simulation")
    add_noise = st.checkbox("Simulate noise", value=False)
    noise_type = st.selectbox("Type", ["gaussian", "saltpepper"]) if add_noise else None
    sigma = st.slider("Gaussian sigma", 1, 50, 20) if add_noise and noise_type == "gaussian" else None
    sp_level = st.slider("Salt&pepper p", 0.0, 0.2, 0.03, 0.01) if add_noise and noise_type == "saltpepper" else None

    st.header("Filters")
    chosen = st.multiselect(
        "Select filters",
        ["gaussian", "median", "nlm"],
        default=["median", "nlm"],
    )
    ksize = st.slider("Kernel size (odd)", 3, 11, 5, 2)
    g_sigma = st.slider("Gaussian sigma", 0.1, 5.0, 1.5, 0.1)
    nlm_h = st.slider("NLM h", 1, 30, 10)
    nlm_hColor = st.slider("NLM hColor", 1, 30, 10)
    nlm_tws = st.slider("NLM templateWindowSize", 3, 15, 7, 2)
    nlm_sws = st.slider("NLM searchWindowSize", 7, 31, 21, 2)
    


def to_bgr(np_img: np.ndarray) -> np.ndarray:
    if np_img.ndim == 2:
        return cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
    return np_img


def load_image(file) -> np.ndarray:
    if file is None:
        return None
    bytes_data = file.read()
    pil = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return to_bgr(img)


def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    gauss = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_sp_noise(img: np.ndarray, amount: float) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    num_salt = int(amount * h * w / 2)
    coords = (np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt))
    out[coords] = 255
    coords = (np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt))
    out[coords] = 0
    return out


img = load_image(uploaded)
col1, col2 = st.columns(2)

if img is not None:
    img = maybe_resize(img, max_size)
    with col1:
        st.subheader("Input")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

    work = img.copy()
    if add_noise and noise_type:
        if noise_type == "gaussian":
            work = add_gaussian_noise(work, sigma)
        else:
            work = add_sp_noise(work, sp_level)

    results = []
    if "gaussian" in chosen:
        g = gaussian_filter(work, ksize=ksize, sigma=g_sigma)
        results.append(("gaussian", g))
    if "median" in chosen:
        m = median_filter(work, ksize=ksize)
        results.append(("median", m))
    if "nlm" in chosen:
        n = nlm_filter_colored(work, h=nlm_h, hColor=nlm_hColor, templateWindowSize=nlm_tws, searchWindowSize=nlm_sws)
        results.append(("nlm", n))

    if apply_unsharp:
        results = [(name, unsharp_mask(img_, amount=unsharp_amount, sigma=unsharp_sigma)) for name, img_ in results]

    with col2:
        st.subheader("Results")
        for name, out in results:
            st.markdown(f"**{name}**")
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB")

    if results:
        st.download_button(
            label="Download first result",
            data=cv2.imencode('.png', results[0][1])[1].tobytes(),
            file_name=f"denoised_{results[0][0]}.png",
            mime="image/png",
        )
else:
    st.info("Upload an image in the sidebar to begin.")


