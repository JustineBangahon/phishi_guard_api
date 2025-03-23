import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_model_to_onnx():
    """Convert the trained scikit-learn model to ONNX format for faster inference"""
    # Determine file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try multiple potential locations for model files
    potential_model_dirs = [
        os.path.join(script_dir, 'model'),
        os.path.join(script_dir, 'backend', 'model'),
        'model',
        'backend/model',
    ]
    
    # Find the model directory that contains the files
    model_dir = None
    for directory in potential_model_dirs:
        if os.path.exists(os.path.join(directory, 'phishing_model.pkl')):
            model_dir = directory
            break
    
    if model_dir is None:
        raise FileNotFoundError("Could not find model directory with required files")
    
    model_path = os.path.join(model_dir, 'phishing_model.pkl')
    tfidf_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    onnx_path = os.path.join(model_dir, 'phishing_model.onnx')
    
    # Load the trained model and vectorizer
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    logger.info(f"Loading vectorizer from {tfidf_path}")
    tfidf = joblib.load(tfidf_path)
    
    # Get the number of features from TF-IDF
    n_features = tfidf.get_feature_names_out().shape[0]
    logger.info(f"Vectorizer has {n_features} features")
    
    # Define input type
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    # Convert to ONNX
    logger.info("Converting model to ONNX format")
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    
    # Save the ONNX model
    logger.info(f"Saving ONNX model to {onnx_path}")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    logger.info("ONNX conversion completed successfully")
    return onnx_path

if __name__ == "__main__":
    try:
        onnx_file = convert_model_to_onnx()
        print(f"Model successfully converted to ONNX format: {onnx_file}")
    except Exception as e:
        logger.error(f"Error converting model to ONNX: {str(e)}")
        print(f"Error: {str(e)}")