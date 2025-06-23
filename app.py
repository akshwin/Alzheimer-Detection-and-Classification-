import streamlit as st

# Set page config FIRST
st.set_page_config(page_title="Alzheimer's MRI Classifier")

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
LAST_CONV_LAYER_NAME = 'conv3'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

    output_path = os.path.join(UPLOAD_FOLDER, f"heatmap_{uuid.uuid4()}.png")
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return output_path

# ---- Sidebar ---- #
with st.sidebar:
    st.title("🧠 About the App")
    st.markdown("""
    This application uses a **Deep Learning model** to classify brain MRI scans into four stages of **Alzheimer's Disease**:
    
    - 🟢 Non Demented  
    - 🟡 Very Mild Demented  
    - 🟠 Mild Demented  
    - 🔴 Moderate Demented

    ---
    **How to use:**
    1. Upload a brain MRI image (JPG/PNG).
    2. The model will predict the stage.
    3. A Grad-CAM heatmap will be generated to visualize important regions.

    ---
    **What is Grad-CAM?**  
    Grad-CAM highlights the parts of the brain image that were most important for the model's prediction. This increases **transparency** and helps in **interpretability**.

    ---
    📩 *For academic or medical collaborations, contact: akshwint.2003@gmail.com*
    """)

# ---- Main UI ---- #
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🧠 Alzheimer's Stage Detection from MRI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an MRI brain scan to classify the Alzheimer's stage and see the Grad-CAM heatmap.</p>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("📤 Upload an MRI Image (JPG/PNG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
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

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_path, caption="🧾 Uploaded MRI", use_column_width=True)
        with col2:
            st.image(heatmap_path, caption="🔥 Grad-CAM Heatmap", use_column_width=True)

        st.markdown("---")
        st.markdown(
            f"<div style='text-align:center; font-size:24px; font-weight:bold; color:#1ABC9C;'>🧬 Predicted Stage: <span style='color:#2C3E50;'>{pred_class}</span></div>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
