import streamlit as st
from ultralytics import YOLO
import os
import uuid
from PIL import Image
import shutil

import cv2
print("OpenCV Version:", cv2.__version__)

# Setup folders
UPLOAD_FOLDER = "static/uploads"
PREDICT_FOLDER = "runs/detect/predict"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv8 model
model = YOLO("best_weights_only.pt")

st.title("Helmet Detection with YOLOv8")

# Upload file
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Simpan file yang diupload
    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Hapus folder predict sebelumnya (jika ada)
    if os.path.exists(PREDICT_FOLDER):
        shutil.rmtree(PREDICT_FOLDER)

    # Jalankan prediksi
    results = model(filepath, save=True)

    # Ambil path hasil deteksi
    result_path = os.path.join(results[0].save_dir, filename)

    # Tampilkan hasil deteksi
    st.subheader("Detected Image")
    st.image(result_path, use_column_width=True)

    # Tombol untuk download hasil deteksi
    with open(result_path, "rb") as file:
        btn = st.download_button(
            label="Download result",
            data=file,
            file_name="detected_" + filename,
            mime="image/jpeg"
        )
