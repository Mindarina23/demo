import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

st.title("🌾 Demo YOLOv11 - Deteksi Penyakit Daun Padi")

# --- Load Model ---
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error("❌ Model 'best.pt' tidak ditemukan. Upload dulu ke project folder.")
        st.stop()
    return YOLO(model_path)

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
    img_np = np.array(img)  # convert ke numpy array

    # Tampilkan gambar asli
    st.image(img_np, caption="📷 Gambar asli", use_column_width=True)

    # Jalankan prediksi
    st.write("🔍 Sedang mendeteksi penyakit...")
    results = model.predict(img_np, conf=0.4, imgsz=640, verbose=False)

    # Tampilkan hasil deteksi dengan bounding box
    for r in results:
        vis_rgb = r.plot()  # hasil deteksi sudah dalam RGB
        st.image(vis_rgb, caption="✅ Hasil deteksi", use_column_width=True)

        # --- Tambahkan informasi prediksi ---
        if r.boxes is not None and len(r.boxes) > 0:
            st.subheader("📊 Hasil Deteksi Penyakit:")
            data = []
            for box in r.boxes:
                cls_id = int(box.cls[0])          # class id
                cls_name = model.names[cls_id]    # nama class
                conf = float(box.conf[0])         # confidence
                data.append({"Penyakit": cls_name, "Confidence": f"{conf:.2f}"})

            st.table(data)
        else:
            st.warning("❌ Tidak ada penyakit yang terdeteksi pada gambar ini.")
