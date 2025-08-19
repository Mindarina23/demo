import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Deteksi Penyakit Daun Padi", page_icon="🌾", layout="centered")
st.title("🌾 Demo YOLOv11 - Deteksi Penyakit Daun Padi")

# --- Load Model ---
@st.cache_resource
def load_model():
    # pastikan best.pt ada di folder yang sama dengan app
    return YOLO("best.pt")

model = load_model()

# --- Upload File ---
uploaded_file = st.file_uploader(
    "📤 Upload gambar daun padi",
    type=["jpg", "jpeg", "png", "webp", "bmp"]
)

# --- Jika ada file diupload ---
if uploaded_file is not None:
    # Buka gambar asli
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    st.image(img_np, caption="📷 Gambar asli", use_container_width=True)

    # Jalankan prediksi
    st.write("🔍 Sedang mendeteksi penyakit...")
    results = model.predict(
        img_np,
        conf=0.40,          # confidence threshold
        imgsz=640,          # ukuran gambar input
        line_thickness=2,   # tebal bounding box
        verbose=False
    )

    # Tampilkan hasil prediksi
    for r in results:
        # gambar hasil dengan bounding box, label, dan confidence
        vis_bgr = r.plot()
        vis_rgb = vis_bgr[:, :, ::-1]

        st.image(vis_rgb, caption="✅ Hasil deteksi", use_container_width=True)

        # Tampilkan detail deteksi (kelas + confidence)
        if r.boxes is not None and len(r.boxes) > 0:
            st.subheader("📊 Detail Deteksi")
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                st.write(f"- **{cls_name}** dengan confidence: {conf:.2f}")
        else:
            st.info("Tidak ada penyakit terdeteksi pada gambar ini.")
