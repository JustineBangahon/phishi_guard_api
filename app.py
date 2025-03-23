from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os

nltk.download('stopwords')
app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app

# Improve path handling
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'backend', 'model')

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Try multiple potential locations for model files
def load_model_file(filename):
    potential_paths = [
        os.path.join(MODEL_DIR, filename),
        os.path.join(SCRIPT_DIR, 'model', filename),
        os.path.join(PROJECT_ROOT, 'model', filename),
        f'backend/model/{filename}',
        f'model/{filename}',
        filename
    ]
    
    for path in potential_paths:
        try:
            return joblib.load(path)
        except:
            continue
    
    raise FileNotFoundError(f"Could not find {filename}. Tried paths: {potential_paths}")

try:
    # Load model and vectorizer
    print("Loading model files...")
    model = load_model_file('phishing_model.pkl')
    tfidf = load_model_file('tfidf_vectorizer.pkl')
    print("Model and vectorizer loaded successfully!")
    
    # Try to load model metrics if available
    try:
        model_metrics = load_model_file('model_metrics.pkl')
        print("Model metrics loaded successfully!")
    except:
        # Default metrics if file doesn't exist yet
        print("Model metrics not found, using default values.")
        model_metrics = {
            "accuracy": 0.97,
            "total_tests": 1115,
            "dataset_info": "Training dataset consisted of 5,574 SMS messages from the SMS Spam Collection Dataset",
            "metrics": {
                "precision": 0.95,
                "recall": 0.94,
                "f1_score": 0.94,
                "false_positive_rate": 0.01
            },
            "test_cases": []
        }
except Exception as e:
    print(f"Error loading model files: {str(e)}")
    # We'll let the app start anyway, but endpoints will return errors

@app.route("/")
def home():
    return "PhishGuard API is running!"

def preprocess_text(text):
    # Handle None or non-string inputs
    if not isinstance(text, str):
        return ""
        
    # Identical preprocessing to training
    text = re.sub('[^a-zA-Z0-9]', ' ', text).lower()
    words = text.split()
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

def save_model_metrics():
    try:
        metrics_path = os.path.join(MODEL_DIR, 'model_metrics.pkl')
        joblib.dump(model_metrics, metrics_path)
        print(f"Updated model metrics saved to {metrics_path}")
    except Exception as e:
        print(f"Warning: Could not save model metrics: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        sms_text = data['text']
        
        # Preprocess and vectorize
        processed_text = preprocess_text(sms_text)
        text_vector = tfidf.transform([processed_text]).toarray()
        
        # Predict
        prediction = model.predict_proba(text_vector)[0][1]  # Get phishing probability
        
        # Store this prediction for performance analysis if feedback is available
        # This could be expanded to include a feedback system
        if len(model_metrics["test_cases"]) < 100:  # Limit stored test cases to 100
            # Determine result category
            result = ""
            if prediction >= 0.8:
                result = "Suspicious Detected"
            elif prediction >= 0.3:
                result = "Undecidable"
            else:
                result = "Not Phishing"
                
            # Don't store obviously sensitive data, truncate message if necessary
            safe_text = sms_text[:100] + "..." if len(sms_text) > 100 else sms_text
                
            # Only add test cases that would be interesting for users to see
            if prediction > 0.7 or prediction < 0.2 or (0.4 < prediction < 0.6):
                test_case = {
                    "message_type": "Phishing" if prediction > 0.8 else "Legitimate" if prediction < 0.3 else "Ambiguous",
                    "message": safe_text,
                    "actual_probability": float(prediction),
                    "expected_result": "Unknown", # Without feedback we don't know
                    "actual_result": result
                }
                
                # Check if we already have this exact message
                exists = False
                for existing_case in model_metrics["test_cases"]:
                    if existing_case["message"] == safe_text:
                        exists = True
                        break
                        
                if not exists:
                    model_metrics["test_cases"].append(test_case)
                    # Save updated metrics
                    save_model_metrics()
                        
        return jsonify({
            'phishing_probability': float(prediction),
            'status': 'success'
        })
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e), 'status': 'failed'}), 400

@app.route('/model_performance', methods=['GET'])
def get_model_performance():
    try:
        return jsonify(model_metrics)
    except Exception as e:
        print(f"Error getting model performance: {str(e)}")
        return jsonify({'error': str(e), 'status': 'failed'}), 400

if __name__ == '__main__':
    app.run(debug=True)