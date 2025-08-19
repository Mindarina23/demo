import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import pandas as pd

st.title("🌾 Demo YOLOv11 - Deteksi Penyakit Daun Padi (Multi-class Friendly)")

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
    st.image(img, caption="📷 Gambar asli", use_container_width=True)

    # Jalankan prediksi
    st.write("🔍 Sedang mendeteksi penyakit...")
    results = model.predict(
        source=temp_path,
        conf=0.1,      # lebih rendah = tangkap lebih banyak
        iou=0.7,       # biar nggak terlalu agresif hapus box lain
        imgsz=640,
        line_thickness=2,
        verbose=False
    )

    # --- Tampilkan hasil deteksi dengan bounding box ---
    for r in results:
        vis_bgr = r.plot()             # hasil deteksi BGR
        vis_rgb = vis_bgr[:, :, ::-1]  # ubah ke RGB
        st.image(vis_rgb, caption="✅ Hasil deteksi", use_container_width=True)

        # --- Buat tabel hasil deteksi ---
        if r.boxes is not None and len(r.boxes) > 0:
            st.subheader("📊 Deteksi Penyakit (Detail):")

            deteksi_list = []
            for box in r.boxes:
                cls_id = int(box.cls[0])       # class id
                cls_name = model.names[cls_id] # nama class
                conf = float(box.conf[0])      # confidence
                xyxy = box.xyxy[0].tolist()    # [x1, y1, x2, y2]

                deteksi_list.append({
                    "Penyakit": cls_name,
                    "Confidence": f"{conf:.2f}",
                    "Koordinat [x1, y1, x2, y2]": [round(v, 1) for v in xyxy]
                })

            df = pd.DataFrame(deteksi_list)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("❌ Tidak ada penyakit yang terdeteksi pada gambar ini.")

    # Hapus file temp kalau mau
    if os.path.exists(temp_path):
        os.remove(temp_path)
