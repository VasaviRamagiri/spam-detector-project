from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import tensorflow as tf
from transformers import BertTokenizer
import os

# Load model and scaler
model = tf.keras.models.load_model("model/spam_model.keras")
scaler = joblib.load("model/scaler.pkl")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

app = Flask(__name__, template_folder="templates")
max_len = 128

# Feature extraction
def extract_features(subject, message):
    text = f"{subject} {message}"
    features = [
        len(subject),
        len(message),
        sum(c in "!@#$%^&*()_+" for c in text),
        sum(c.isdigit() for c in text),
        sum(c.isupper() for c in text),
        np.mean([len(w) for w in text.split()]) if text.split() else 0,
        len(text.split()),
        int("http" in text or "www" in text),
        int("Re:" in subject or "Fwd:" in subject)
    ]
    return features

# Serve the HTML page
@app.route("/")
def index():
    return render_template("index.html")

# API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    subject = data.get("subject", "")
    message = data.get("message", "")

    text = f"{subject} {message}"
    tokenized = tokenizer(text, padding='max_length', truncation=True,
                          max_length=max_len, return_tensors='np')['input_ids']

    raw_features = np.array(extract_features(subject, message)).reshape(1, -1)
    norm_features = scaler.transform(raw_features)

    model_input = np.hstack((tokenized, norm_features))
    pred = model.predict(model_input)[0][0]
    label = int(pred > 0.5)

    return jsonify({
        "prediction": "Spam" if label == 1 else "Ham",
        "confidence": f"{pred:.2f}"
    })

if __name__ == "__main__":
    app.run(debug=True)
