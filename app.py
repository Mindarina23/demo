import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

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
    img_np = np.array(img)  # convert ke numpy array supaya aman

    # Tampilkan gambar asli
    st.image(img_np, caption="📷 Gambar asli")

    # Jalankan prediksi
    st.write("🔍 Sedang mendeteksi...")
    results = model.predict(img_np, conf=0.5, verbose=False)

    # Tampilkan hasil deteksi dengan bounding box
    for r in results:
        vis_bgr = r.plot()             # hasil deteksi BGR (sudah ada box + label)
        vis_rgb = vis_bgr[:, :, ::-1]  # ubah ke RGB
        st.image(vis_rgb, caption="✅ Hasil deteksi dengan bounding box")

        # Ambil detail deteksi (label + confidence)
        if r.boxes is not None and len(r.boxes) > 0:
            st.subheader("📋 Detail Deteksi")
            for box in r.boxes:
                cls_id = int(box.cls[0])   # ID class
                cls_name = r.names[cls_id] # Nama class
                conf = float(box.conf[0])  # Confidence
                st.write(f"- {cls_name} ({conf:.2f})")
        else:
            st.info("🚫 Tidak ada penyakit yang terdeteksi.")
