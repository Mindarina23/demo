import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

st.title("🌾 Demo YOLOv11 - Deteksi Penyakit Daun Padi")

# --- Load Model ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # pastikan file best.pt ada di folder project

model = load_model()

# --- Upload File ---
uploaded_file = st.file_uploader(
    "Upload gambar daun padi",
    type=["jpg", "jpeg", "png", "webp", "bmp"]
)

# --- Jika ada file diupload ---
if uploaded_file is not None:
    # Buka gambar dengan PIL
    img = Image.open(uploaded_file).convert("RGB")
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)

    # Font (fallback kalau Arial tidak ada di server)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Tampilkan gambar asli
    st.image(img, caption="📷 Gambar asli")

    # Jalankan prediksi
    st.write("🔍 Sedang mendeteksi...")
    results = model.predict(np.array(img), conf=0.5, verbose=False)

    # Tampilkan hasil deteksi dengan bounding box manual
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                # Ambil koordinat bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])   # ID class
                cls_name = r.names[cls_id] # Nama class
                conf = float(box.conf[0])  # Confidence

                # Gambar kotak (hijau)
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)

                # Gambar label di atas box
                text = f"{cls_name} {conf:.2f}"
                text_size = draw.textbbox((x1, y1), text, font=font)
                draw.rectangle(text_size, fill="lime")
                draw.text((x1, y1), text, fill="black", font=font)

            st.image(img_draw, caption="✅ Hasil deteksi (bounding box)")
        else:
            st.info("🚫 Tidak ada penyakit yang terdeteksi.")
