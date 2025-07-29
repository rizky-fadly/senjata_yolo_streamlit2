import streamlit as st
from ultralytics import YOLO
import os
import uuid
from PIL import Image
import shutil

# Inisialisasi folder
UPLOAD_DIR = "static/uploads"
OUTPUT_DIR = "runs/detect/predict"
os.makedirs(UPLOAD_DIR, exist_ok=True)


detector = YOLO("best.pt")

st.title("Gun Detection App")

# Upload gambar
image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if image_file:
    # Simpan gambar dengan nama acak
    unique_name = f"{uuid.uuid4()}.jpg"
    saved_path = os.path.join(UPLOAD_DIR, unique_name)
    with open(saved_path, "wb") as f:
        f.write(image_file.getbuffer())

    # Bersihkan folder output sebelumnya
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    # Deteksi objek dalam gambar
    detection_result = detector(saved_path, save=True)

    # Ambil path dari hasil prediksi
    output_image_path = os.path.join(detection_result[0].save_dir, unique_name)

    # Tampilkan gambar hasil deteksi
    st.subheader("Detection Result")
    st.image(output_image_path, use_column_width=True)

    # Opsi download gambar hasil
    with open(output_image_path, "rb") as img_file:
        st.download_button(
            label="Download Detected Image",
            data=img_file,
            file_name=f"result_{unique_name}",
            mime="image/jpeg"
        )
