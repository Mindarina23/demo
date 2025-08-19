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
    img_np = np.array(img)

    # Tampilkan gambar asli
    st.image(img_np, caption="📷 Gambar asli")

    # Jalankan prediksi (samakan param Colab)
    st.write("🔍 Sedang mendeteksi...")
    results = model.predict(
        source=img_np,
        conf=0.40,           # sama seperti Colab
        imgsz=640,           # ukuran input konsisten
        line_thickness=2,    # biar bbox jelas
        show_labels=True,
        show_conf=True,
        verbose=False
    )

    # Ambil hasil pertama
    r = results[0]

    # Tampilkan gambar hasil prediksi
    vis_bgr = r.plot()             # hasil prediksi dengan bbox
    vis_rgb = vis_bgr[:, :, ::-1]  # convert BGR → RGB
    st.image(vis_rgb, caption="✅ Hasil deteksi")

    # Tampilkan detail deteksi
    st.subheader("📋 Detail Deteksi")
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id] if model.names else str(cls_id)
            st.write(f"- {label} ({conf:.2%})")
    else:
        st.write("❌ Tidak ada objek terdeteksi pada gambar ini.")
