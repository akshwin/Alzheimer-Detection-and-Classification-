import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import uuid
import tensorflow as tf
import numpy as np
from PIL import Image

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your model
model = tf.keras.models.load_model('model.h5')

class_names = ['glioma', 'meningioma', 'no tumor', 'pituitary']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(str(uuid.uuid4()) + ".png")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Image preprocessing
        img = Image.open(filepath).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        return render_template('result.html', prediction=predicted_class, image_path=filepath)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)