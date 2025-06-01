import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os
import uuid

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

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(image_path, heatmap):
    original_img = Image.open(image_path).convert('RGB')
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(original_img.size)

    # Save superimposed image using matplotlib
    fig, ax = plt.subplots()
    ax.imshow(original_img)
    ax.imshow(heatmap_resized, cmap='jet', alpha=0.4)
    ax.axis('off')

    output_path = f"static/uploads/heatmap_{uuid.uuid4()}.png"
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return output_path