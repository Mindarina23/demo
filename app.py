import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Demo YOLOv11 - Deteksi Penyakit Daun Padi")

@st.cache_resource
def load_model():
    return YOLO("best.pt")   # pastikan file best.pt ada

model = load_model()

uploaded_file = st.file_uploader("Upload gambar daun padi", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar asli", use_container_width=True)

    st.write("🔍 Sedang mendeteksi...")
    results = model.predict(np.array(img), conf=0.5, verbose=False)

    for r in results:
        vis_bgr = r.plot()
        vis_rgb = vis_bgr[:, :, ::-1]
        st.image(vis_rgb, caption="Hasil deteksi", use_container_width=True)
