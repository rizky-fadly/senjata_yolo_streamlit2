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

# Load model dan perbaiki urutan label
detector = YOLO("best.pt")
detector.names = ['pisau', 'pistol']  # Sesuai dataset.yaml

st.title("Gun & Knife Detection App")

# Upload gambar
gambar = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if gambar:
    unique_name = f"{uuid.uuid4()}.jpg"
    saved_path = os.path.join(UPLOAD_DIR, unique_name)
    with open(saved_path, "wb") as f:
        f.write(gambar.getbuffer())

    # Bersihkan output lama
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    # Jalankan deteksi
    results = detector(saved_path, save=True)

    # Ambil hasil gambar
    output_image_path = os.path.join(results[0].save_dir, unique_name)

    # Tampilkan hasil
    st.subheader("Hasil Deteksi")
    st.image(output_image_path, use_column_width=True)

    st.subheader("Detail Deteksi")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = detector.names[cls_id]
        st.write(f"- Kelas: {label}, Confidence: {conf:.2f}")

    # Download tombol
    with open(output_image_path, "rb") as img_file:
        st.download_button(
            label="Download Gambar Hasil",
            data=img_file,
            file_name=f"result_{unique_name}",
            mime="image/jpeg"
        )
