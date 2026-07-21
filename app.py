import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from model import preprocess_text

app = Flask(__name__, static_folder='static', static_url_path='')

MODEL_PATH = "spam_model.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"

# Load model and vectorizer
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Model or vectorizer file missing. Please run model.py first.")

classifier = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
classes = list(classifier.classes_) # e.g. ['Ham', 'Spam']

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True, silent=True)
        if not data or 'text' not in data:
            return jsonify({"error": "Invalid request. 'text' field is required."}), 400
        
        raw_text = data.get('text', '').strip()
        if not raw_text:
            return jsonify({"error": "Email content cannot be empty."}), 400
        
        # Preprocess text
        cleaned = preprocess_text(raw_text)
        if not cleaned:
            return jsonify({
                "prediction": "Ham",
                "confidence": 0.50,
                "spam_probability": 0.50,
                "ham_probability": 0.50,
                "detected_keywords": [],
                "clean_text": ""
            })
        
        # Vectorize
        features = vectorizer.transform([cleaned])
        
        # Predict
        prediction = classifier.predict(features)[0]
        probabilities = classifier.predict_proba(features)[0]
        
        # Map probabilities
        class_prob_map = dict(zip(classes, probabilities))
        spam_prob = float(class_prob_map.get("Spam", 0.0))
        ham_prob = float(class_prob_map.get("Ham", 0.0))
        
        confidence = float(class_prob_map.get(prediction, 0.5))
        
        # Extract matching feature keywords present in the text for detailed insights
        feature_names = vectorizer.get_feature_names_out()
        nonzero_indices = features.nonzero()[1]
        detected_keywords = [feature_names[idx] for idx in nonzero_indices]
        
        return jsonify({
            "prediction": str(prediction),
            "confidence": round(confidence, 4),
            "spam_probability": round(spam_prob, 4),
            "ham_probability": round(ham_prob, 4),
            "detected_keywords": detected_keywords[:10], # Top 10 matching TF-IDF tokens
            "cleaned_text": cleaned
        })
        
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    print("[Server] Starting Email Spam Detector Flask Server on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
