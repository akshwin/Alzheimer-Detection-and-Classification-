import streamlit as st

# Set page config FIRST
st.set_page_config(page_title="NeuroAlz AI", layout="centered", page_icon="🧠")

import numpy as np
import tensorflow as tf
from PIL import Image
import os
import uuid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- Constants ---- #
CLASS_NAMES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
LAST_CONV_LAYER_NAME = 'conv3'  # Update this to match your model's last conv layer

# Folders
UPLOAD_FOLDER = 'static/upload'
HEATMAP_FOLDER = 'static/grad'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

# ---- Load model ---- #
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model('model.h5')

model = load_trained_model()

# ---- Utility functions ---- #
def preprocess_image(image_path, target_size=(128, 128)):
    img = Image.open(image_path).convert('L')
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # Shape: (1, H, W, 1)
    return img

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap + 1e-10)
    return heatmap.numpy()

def overlay_heatmap(image_path, heatmap):
    original_img = Image.open(image_path).convert('RGB')
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(original_img.size)

    fig, ax = plt.subplots()
    ax.imshow(original_img)
    ax.imshow(heatmap_resized, cmap='jet', alpha=0.4)
    ax.axis('off')

    output_path = os.path.join(HEATMAP_FOLDER, f"heatmap_{uuid.uuid4()}.png")
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return output_path

# ---- Sidebar ---- #
with st.sidebar:
    st.markdown("## 🧠 NeuroAlz AI ")
    st.markdown("""
    This application uses a **deep learning model** to classify brain MRI scans into four stages of **Alzheimer's Disease**  
    using **Convolutional Neural Networks (CNNs)** and **Grad-CAM** heatmaps for interpretability.
    """)

    with st.expander("🔍 Model Details"):
        st.markdown("""
        - **Architecture Used**: Custom CNN  
        - **Input Shape**: Grayscale 128x128  
        - **Training**: On Alzheimer's MRI dataset  
        - **Explainability**: Grad-CAM heatmaps show activated regions
        """)

    with st.expander("📂 Classes Detected"):
        st.markdown("""
        - 🔴 **Moderate Demented**  
        - 🟠 **Mild Demented**  
        - 🟡 **Very Mild Demented**  
        - 🟢 **Non Demented**
        """)

    with st.expander("📁 Dataset Info"):
        st.markdown("""
        - Real MRI brain scans  
        - Open-source dataset (Alzheimer's Dataset on Kaggle)  
        - Resized to 128x128 pixels and converted to grayscale
        """)

    st.markdown("---")
    st.markdown("👨‍💻 Developed by Akshwin T")
    st.markdown("📬 Contact: [akshwint.2003@gmail.com](mailto:akshwint.2003@gmail.com)")

# ---- Main UI ---- #
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🧠 NeuroAlz-AI : An Alzheimer's Stage Classifier </h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an MRI brain scan to classify the Alzheimer's stage and see the Grad-CAM heatmap.</p>", unsafe_allow_html=True)
st.markdown("---")

# Upload and Sample Image Button
uploaded_file = st.file_uploader("📤 Upload an MRI Image (JPG/PNG)", type=["png", "jpg", "jpeg"])
use_sample = st.button("🖼️ Use Sample Image")

if uploaded_file or use_sample:
    try:
        if use_sample:
            img_path = os.path.join("upload_image", "sample.png")  # sample image from upload_image/
        else:
            temp_filename = f"{uuid.uuid4()}.png"
            img_path = os.path.join(UPLOAD_FOLDER, temp_filename)
            with open(img_path, "wb") as f:
                f.write(uploaded_file.read())

        with st.spinner("🔎 Analyzing..."):
            preprocessed = preprocess_image(img_path)
            prediction = model.predict(preprocessed)
            pred_class = CLASS_NAMES[np.argmax(prediction)]
            heatmap = make_gradcam_heatmap(preprocessed, model, LAST_CONV_LAYER_NAME)
            heatmap_path = overlay_heatmap(img_path, heatmap)

        # Display side-by-side images: Uploaded Left, Heatmap Right
        col_left, col_right = st.columns(2)
        with col_left:
            st.image(img_path, caption="🧾 Uploaded MRI", use_container_width=True)
        with col_right:
            st.image(heatmap_path, caption="🔥 Grad-CAM Heatmap", use_container_width=True)

        # Predicted stage
        st.markdown("---")
        st.markdown(
            f"<div style='text-align:center; font-size:24px; font-weight:bold; color:#1ABC9C;'>🧬 Predicted Stage: <span style='color:white;'>{pred_class}</span></div>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")