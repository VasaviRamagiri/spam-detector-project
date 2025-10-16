 Improving Spam Detection Using BERT and Deep Learning

An advanced, hybrid spam detection system leveraging **BERT** for contextual features, refined by nature-inspired optimization algorithms **Grey Wolf Optimization (GWO)** and **Particle Swarm Optimization (PSO)**, and deployed via a **Flask** web application.

Project Overview

This project addresses the challenge of evolving and sophisticated spam emails by proposing a robust deep learning framework. The core innovation lies in the fusion of contextual features extracted by BERT with traditional handcrafted linguistic features, followed by aggressive feature selection and hyperparameter optimization.

The final deployed model, a **GRU (Gated Recurrent Unit) model optimized with PSO**, achieved a superior performance metric.

Core Methodology & Novelty

### **Feature Engineering**

  * **Component:** **BERT-Based Tokenization** + **Handcrafted Features**.
  * **Purpose:** Captures both deep semantic context and superficial spam indicators (e.g., links, punctuation frequency, uppercase letters).
  * **Impact:** Increased representational depth for better classification.

### **Feature Selection**

  * **Component:** **Grey Wolf Optimization (GWO)**.
  * **Purpose:** Reduces data dimensionality by selecting the most pertinent feature subset, eliminating redundancy.
  * **Impact:** Improved generalization and reduced training time by **15-20%**.

### **Model Tuning**

  * **Component:** **Particle Swarm Optimization (PSO)**.
  * **Purpose:** Optimizes key hyperparameters (e.g., learning rate and GRU units) of the final classifier.
  * **Impact:** Accelerated convergence and maximized model performance.

### **Final Classifier**

  * **Component:** **GRU (Gated Recurrent Unit)**.
  * **Purpose:** Chosen for its efficiency in handling sequential data.
  * **Impact:** Achieved the highest accuracy when combined with PSO.

Performance Highlights

The hybrid GRU model with PSO achieved the highest accuracy:

  * **GRU + PSO:** **99.31% Accuracy**
  * **BiLSTM + GWO:** 98.79% Accuracy
  * **LSTM + GWO:** 96.72% Accuracy
  * **Random Forest (RF) + TF-IDF:** 88.13% Accuracy

Repository Structure

```
spam-detector-project/
├── spam_detector_app/          # Flask application directory
│   ├── app.py                  # Main prediction API and feature extraction logic
│   ├── templates/              # HTML templates for the web interface
│   │   └── index.html          
│   └── model/                  # Trained models and auxiliary files
│       ├── spam_model.keras    # Final trained Keras model
│       └── scaler.pkl          # MinMaxScaler object for feature scaling
├── TRAINING GRU.ipynb          # Jupyter Notebook for GRU training and PSO
├── gwo-bert-bilstm.ipynb       # Jupyter Notebook for BiLSTM training and GWO
├── requirements.txt            # Python dependencies for deployment
└── Procfile                    # Command for the web server (Gunicorn)
```

Local Setup and Installation

### Prerequisites

  * Python 3.9+
  * Git

### Installation Steps

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/VasaviRamagiri/spam-detector-project.git
    cd spam-detector-project
    ```

2.  **Create a virtual environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    *(Note: The `requirements.txt` file is inside the `spam_detector_app` folder.)*

    ```bash
    pip install -r spam_detector_app/requirements.txt
    ```

### Running the Flask Application

1.  Ensure your virtual environment is active.
2.  Set the Flask environment variable:
    ```bash
    export FLASK_APP=spam_detector_app/app.py
    ```
3.  Run the application in development mode:
    ```bash
    flask run
    ```

The application will now be running on `http://127.0.0.1:5000/`.
