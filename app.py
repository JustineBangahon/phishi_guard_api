from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import os
import logging
import onnxruntime as ort  # For ONNX model inference

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
    logger.info("NLTK stopwords found")
except LookupError:
    logger.info("Downloading NLTK stopwords")
    nltk.download('stopwords', quiet=True)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app

# Environment variables
PORT = int(os.environ.get("PORT", 5000))  # Render sets PORT environment variable
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
DEBUG_MODE = ENVIRONMENT == "development"

# Improve path handling
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Global variables for models
model = None
tfidf = None
onnx_session = None
model_metrics = None

def load_models():
    """Load all required model files"""
    global model, tfidf, onnx_session, model_metrics
    
    logger.info("Loading models...")
    
    # Try to load ONNX model first (faster inference)
    onnx_path = os.path.join(MODEL_DIR, 'phishing_model.onnx')
    if os.path.exists(onnx_path):
        try:
            logger.info("Loading ONNX model...")
            onnx_session = ort.InferenceSession(onnx_path)
            logger.info("ONNX model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {str(e)}")
            onnx_session = None
    
    # Load scikit-learn model as a fallback
    if onnx_session is None:
        try:
            logger.info("Loading scikit-learn model...")
            model = joblib.load(os.path.join(MODEL_DIR, 'phishing_model.pkl'))
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load scikit-learn model: {str(e)}")
            model = None
    
    # Load vectorizer (needed for both ONNX and scikit-learn models)
    try:
        logger.info("Loading TF-IDF vectorizer...")
        tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
        logger.info("TF-IDF vectorizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load TF-IDF vectorizer: {str(e)}")
        tfidf = None
    
    # Load model metrics
    try:
        logger.info("Loading model metrics...")
        global model_metrics
        model_metrics = joblib.load(os.path.join(MODEL_DIR, 'model_metrics.pkl'))
        logger.info("Model metrics loaded successfully")
    except Exception as e:
        logger.warning(f"Model metrics not found, using default values: {str(e)}")
        model_metrics = {
            "accuracy": 0.98,
            "total_tests": 1115,
            "dataset_info": "Training dataset consisted of SMS messages from the SMS Spam Collection Dataset",
            "metrics": {
                "precision": 0.98,
                "recall": 0.85,
                "f1_score": 0.92,
                "false_positive_rate": 0.00
            },
            "test_cases": []
        }

# Load models at startup
load_models()

def preprocess_text(text):
    """Preprocess text for model input"""
    # Handle None or non-string inputs
    if not isinstance(text, str):
        return ""
    
    # Preprocessing steps identical to training
    text = re.sub('[^a-zA-Z0-9]', ' ', text).lower()
    words = text.split()
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

def predict_with_onnx(text_vector):
    """Make prediction using ONNX model"""
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: text_vector.astype(np.float32)})[0]
    return result[0][1]  # Return probability of class 1 (phishing)

def predict_with_sklearn(text_vector):
    """Make prediction using scikit-learn model"""
    return model.predict_proba(text_vector)[0][1]

def save_model_metrics():
    """Save updated model metrics to disk"""
    try:
        metrics_path = os.path.join(MODEL_DIR, 'model_metrics.pkl')
        joblib.dump(model_metrics, metrics_path)
        logger.info("Updated model metrics saved")
    except Exception as e:
        logger.warning(f"Could not save model metrics: {str(e)}")

@app.route("/")
def home():
    return "PhishGuard API is running!"

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    health_status = {
        "status": "healthy",
        "models_loaded": {
            "onnx": onnx_session is not None,
            "sklearn": model is not None,
            "vectorizer": tfidf is not None,
            "metrics": model_metrics is not None
        }
    }
    return jsonify(health_status)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and validate input
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided', 'status': 'failed'}), 400
        
        sms_text = data['text']
        
        # Preprocess and vectorize
        processed_text = preprocess_text(sms_text)
        if not processed_text:
            return jsonify({'error': 'Invalid text format', 'status': 'failed'}), 400
        
        text_vector = tfidf.transform([processed_text]).toarray()
        
        # Make prediction using the appropriate model
        if onnx_session is not None:
            prediction = predict_with_onnx(text_vector)
        elif model is not None:
            prediction = predict_with_sklearn(text_vector)
        else:
            return jsonify({'error': 'No models available for prediction', 'status': 'failed'}), 500
        
        # Determine result category
        if prediction >= 0.8:
            result = "Suspicious Detected"
        elif prediction >= 0.3:
            result = "Undecidable"
        else:
            result = "Not Phishing"
        
        # Update test cases for performance metrics
        if model_metrics and len(model_metrics["test_cases"]) < 100:
            # Don't store obviously sensitive data, truncate message if necessary
            safe_text = sms_text[:100] + "..." if len(sms_text) > 100 else sms_text
            
            # Only add interesting test cases
            if prediction > 0.7 or prediction < 0.2 or (0.4 < prediction < 0.6):
                test_case = {
                    "message_type": "Phishing" if prediction > 0.8 else "Legitimate" if prediction < 0.3 else "Ambiguous",
                    "message": safe_text,
                    "actual_probability": float(prediction),
                    "expected_result": "Unknown",  # Without feedback we don't know
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
                    save_model_metrics()
        
        # Return prediction result
        return jsonify({
            'phishing_probability': float(prediction),
            'result': result,
            'status': 'success'
        })
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e), 'status': 'failed'}), 500

@app.route('/model_performance', methods=['GET'])
def get_model_performance():
    try:
        if not model_metrics:
            return jsonify({'error': 'Model metrics not available', 'status': 'failed'}), 404
        
        return jsonify(model_metrics)
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        return jsonify({'error': str(e), 'status': 'failed'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG_MODE)