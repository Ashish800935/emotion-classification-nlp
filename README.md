# Emotion Classification NLP

An NLP-based deep learning application that predicts the emotion expressed in a given text. The project explores RNN-based architectures and deploys the final model through a FastAPI web application.

## 🎯 Features

* Text preprocessing and tokenization
* Emotion classification using deep learning
* Six emotion classes:

  * 😢 Sadness
  * 😄 Joy
  * ❤️ Love
  * 😠 Anger
  * 😨 Fear
  * 😲 Surprise
* Confidence score for predictions
* Probability distribution for all emotions
* FastAPI REST API
* Interactive web interface
* Swagger API documentation

## 🧠 Model Pipeline

```text
Input Text
    ↓
Text Preprocessing
    ↓
Tokenization
    ↓
Sequence Padding
    ↓
Embedding
    ↓
BiGRU
    ↓
Dense Layers
    ↓
Softmax
    ↓
Emotion Prediction
```

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* NLP
* NumPy
* FastAPI
* Pydantic
* Uvicorn
* HTML, CSS & JavaScript

## 📁 Project Structure

```text
emotion-classification-nlp/
│
├── artifacts/
│   └── models/
├── notebooks/
│   └── emotion_prediction.ipynb
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── main.py
├── requirements.txt
├── runtime.txt
└── README.md
```

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Ashish800935/emotion-classification-nlp.git
cd emotion-classification-nlp
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI server

```bash
uvicorn main:app --reload
```

Open the application:

```text
http://127.0.0.1:8000/
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## 🔌 API

### `GET /health`

Checks whether the server and model are loaded.

### `POST /predict`

Example request:

```json
{
  "text": "I am extremely happy today!"
}
```

Returns the predicted emotion, confidence score, and probabilities for all six emotions.

## 📊 Model Performance

The final BiGRU model achieved **92.25% test accuracy**, **88.61% macro F1-score**, and **92.45% weighted F1-score** on the test set.

