
# 🧠 Alzheimer Detection and Classification

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen)

A deep learning-based web application that classifies the stage of Alzheimer’s Disease from MRI scans. Built using **TensorFlow**, **Keras**, and **Streamlit**, this tool helps in identifying the severity level—**Mild Demented**, **Moderate Demented**, **Very Mild Demented**, or **Non-Demented**.

🔗 **Try the Live Demo**: [Alzheimer Stage Classifier](https://alzheimer-stage-classifier.streamlit.app/)

---

## 📂 Project Structure

```bash
.
├── static/                 # Static files (CSS, images, etc.)
├── upload_image/          # Uploaded image storage
├── .gitignore             # Git ignored files
├── LICENSE                # License (MIT)
├── README.md              # Project README
├── app.py                 # Streamlit app main file
├── model.h5               # Trained CNN model
├── requirements.txt       # Dependencies
````

---

## 📊 Model Details

* **Architecture**: CNN
* **Input**: MRI Brain Scan Image
* **Output Classes**:

  * Non-Demented 🧠
  * Very Mild Demented 🧠
  * Mild Demented 🧠
  * Moderate Demented 🧠

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Alzheimer-Detection-and-Classification.git
cd Alzheimer-Detection-and-Classification
```

### 2. Create Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

* Python 🐍
* TensorFlow/Keras
* Streamlit
* NumPy & OpenCV
* Matplotlib
* PIL

---

## 📸 Screenshots

<img src="static/sample_ui.png" width="700"/>

---

## 📬 Contact

* Author: [Akshwin T](https://github.com/akshwin01)
* LinkedIn: [linkedin.com/in/akshwin](https://linkedin.com/in/akshwin)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---