import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import pandas as pd

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
    # Simpan file upload ke disk (biar sama dengan Colab)
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Tampilkan gambar asli
    img = Image.open(temp_path).convert("RGB")
    st.image(img, caption="📷 Gambar asli", use_column_width=True)

    # Jalankan prediksi persis seperti di Colab
    st.write("🔍 Sedang mendeteksi penyakit...")
    results = model.predict(
        source=temp_path,
        conf=0.4,
        imgsz=640,
        line_thickness=2,
        verbose=False
    )

    # Ambil hasil prediksi
    for r in results:
        # Tampilkan hasil bounding box di gambar
        vis_bgr = r.plot()
        vis_rgb = vis_bgr[:, :, ::-1]
        st.image(vis_rgb, caption="✅ Hasil deteksi", use_column_width=True)

        # --- Buat tabel detail deteksi ---
        if r.boxes is not None and len(r.boxes) > 0:
            data = []
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                data.append({
                    "Penyakit": cls_name,
                    "Confidence": f"{conf:.2f}",
                    "x1": int(xyxy[0]),
                    "y1": int(xyxy[1]),
                    "x2": int(xyxy[2]),
                    "y2": int(xyxy[3]),
                })
            
            df = pd.DataFrame(data)
            st.subheader("📊 Detail Deteksi:")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("❌ Tidak ada penyakit yang terdeteksi pada gambar ini.")

    # Hapus file temp kalau mau
    if os.path.exists(temp_path):
        os.remove(temp_path)
