from ultralytics import YOLO
import streamlit as st
from PIL import Image

model = YOLO("best.pt")  # pastikan file ini ada di repo

st.title("YOLOv11 Rice Disease Detection Demo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")

    results = model.predict(img)
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="Prediction Result")
