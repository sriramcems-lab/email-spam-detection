# 📧 Email Spam Detection System

A lightweight, local machine learning application for classifying email messages as **Spam** or **Ham (Safe Email)**. Built with Python, TF-IDF vectorization, Multinomial Naive Bayes, Flask, and a modern glassmorphic web UI.

---

## 🌟 Key Features

- **Machine Learning Engine**: TF-IDF Feature Extraction paired with a Naive Bayes classifier (**95.83% Accuracy**).
- **Fast REST API**: Lightweight Flask server providing real-time predictions at `POST /predict`.
- **Modern Web Interface**: Dark-themed, glassmorphic UI featuring live character count, preset samples, confidence meter, and key trigger words tag cloud.
- **Automated Validation**: Built-in test suite (`test_app.py`) for verifying API responses.

---

## 📁 Repository Structure

```text
email-spam-detector/
├── model.py              # Dataset generator & ML model training pipeline
├── app.py                # Flask web API server & static file host
├── test_app.py           # Automated test suite for API verification
├── dataset.csv           # Training dataset (120 spam & ham samples)
├── spam_model.joblib     # Serialized Naive Bayes model
├── tfidf_vectorizer.joblib # Serialized TF-IDF vectorizer
├── .gitignore            # Git ignore rules
└── static/
    ├── index.html        # Glassmorphic frontend UI
    └── style.css         # Custom styling system
```

---

## 🚀 Quick Start Guide

### 1. Requirements
- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`

### 2. Setup Virtual Environment & Dependencies

Using `uv`:
```bash
uv venv
uv pip install scikit-learn pandas numpy flask joblib requests
```

Or standard `pip`:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install scikit-learn pandas numpy flask joblib requests
```

### 3. Train the Model (Optional)
The pre-trained model files (`spam_model.joblib` and `tfidf_vectorizer.joblib`) are included. To re-train from scratch:

```bash
python model.py
```

### 4. Run the Web Application

```bash
python app.py
```

Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

### 5. Run Automated Tests

```bash
python test_app.py
```

---

## 📊 Model Evaluation Report

```text
Accuracy Score: 95.83%

              precision    recall  f1-score   support
         Ham       1.00      0.92      0.96        12
        Spam       0.92      1.00      0.96        12
```
