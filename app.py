'''
Untuk menjalankan aplikasi streamlit secara local, lakukan instalasi modul streamlit melalui command prompt dengan perintah
`pip install streamlit`, kemudian setelah berhasil terinstall aplikasi dapat berjalan dengan mengetikkan perintah
`streamlit run app.py` pada tempat dimana kamu menyimpan file app.py milikmu. Jangan lupa tambahkan file requirements juga
yang berisi library python yang dipakai agar aplikasi bisa berjalan.
'''

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from image_utils import *

st.set_page_config("Aplikasi Pengolahan Citra", layout="wide")
st.title("🖼️ Aplikasi Pengolahan Citra Interaktif")
st.markdown("Unggah gambar dan pilih metode pengolahan citra dari menu di sebelah kiri.")

st.sidebar.title("🔧 Pilih Metode")
method = st.sidebar.selectbox("Metode Pengolahan", [
    "Grayscale",
    "Gaussian Blur",
    "Otsu Thresholding",
    "Prewitt Edge Detection",
    "Sobel Edge Detection",
    "Histogram Equalization",
    "Quantizing Compression"
])

uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.subheader("Gambar Asli")
    st.image(image, use_column_width=True)

    result = None
    if method == "Grayscale":
        result = to_grayscale(image_bgr)
    elif method == "Gaussian Blur":
        result = gaussian_blur(image_bgr)
    elif method == "Otsu Thresholding":
        gray = to_grayscale(image_bgr)
        result = apply_otsu_threshold(gray)
    elif method == "Prewitt Edge Detection":
        result = prewitt_edge(image_bgr)
    elif method == "Sobel Edge Detection":
        gray = to_grayscale(image_bgr)
        result = sobel_edge(gray)
    elif method == "Histogram Equalization":
        gray = to_grayscale(image_bgr)
        result = histogram_equalization(gray)
    elif method == "Quantizing Compression":
        levels = st.sidebar.slider("Level Kuantisasi", 2, 64, 16)
        result = quantize_compression(image_bgr, levels)

    if result is not None:
        st.subheader(f"Hasil: {method}")
        st.image(result, use_column_width=True, clamp=True, channels="GRAY" if len(result.shape) == 2 else "BGR")

else:
    st.warning("⚠️ Harap unggah gambar terlebih dahulu.")

