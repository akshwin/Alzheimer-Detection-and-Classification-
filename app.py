from flask import Flask, render_template, request
from keras.models import load_model
import os
import numpy as np
from utils import preprocess_image, make_gradcam_heatmap, overlay_heatmap
import uuid

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # This creates the folder if it doesn't exist

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('model.h5')
CLASS_NAMES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
LAST_CONV_LAYER_NAME = 'conv3'  # Make sure this layer exists in your model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error="No file uploaded")
    
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error="No selected file")

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.png")
    file.save(img_path)

    # Preprocess the image
    preprocessed = preprocess_image(img_path)
    prediction = model.predict(preprocessed)
    pred_class = CLASS_NAMES[np.argmax(prediction)]

    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(preprocessed, model, LAST_CONV_LAYER_NAME)
    heatmap_path = overlay_heatmap(img_path, heatmap)

    # Render the result on index.html
    return render_template('index.html',
                           prediction=pred_class,
                           uploaded_image=img_path,
                           heatmap_image=heatmap_path)

# For local development only
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))