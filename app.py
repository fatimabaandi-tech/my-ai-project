
import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "caltech101_mobilenetv2.keras"
META_PATH = "model_metadata.json"

st.set_page_config(
    page_title="Caltech-101 AI Classifier",
    page_icon="📷",
    layout="centered"
)

st.title("📷 Caltech-101 Object Classifier")
st.write("Take a photo or upload an image to identify an object.")

if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
    st.error("Model files are missing. Please upload the model and metadata to GitHub.")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

with open(META_PATH, "r") as f:
    metadata = json.load(f)

CLASS_NAMES = metadata["class_names"]
IMG_SIZE = tuple(metadata["img_size"])

st.subheader("1. Take a Photo")
camera_image = st.camera_input("Open Camera")

st.subheader("2. Or Upload an Image")
uploaded_image = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

image_file = camera_image if camera_image is not None else uploaded_image

if image_file is not None:
    image = Image.open(image_file).convert("RGB")

    st.image(image, caption="Selected Image", use_container_width=True)

    if st.button("Predict Object", type="primary"):
        with st.spinner("Predicting..."):
            resized_image = image.resize(IMG_SIZE)
            image_array = np.array(resized_image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = preprocess_input(image_array)

            probabilities = model.predict(image_array, verbose=0)[0]
            top_indices = np.argsort(probabilities)[::-1][:3]

            best_index = top_indices[0]
            best_class = CLASS_NAMES[best_index]
            confidence = probabilities[best_index] * 100

        st.success("Prediction Complete!")
        st.subheader("Predicted Object")
        st.write(f"**{best_class}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

        st.subheader("Top 3 Predictions")
        for index in top_indices:
            class_name = CLASS_NAMES[index]
            score = probabilities[index] * 100
            st.write(f"**{class_name}** — {score:.2f}%")
            st.progress(float(probabilities[index]))

else:
    st.info("Please take a photo or upload an image to start.")
