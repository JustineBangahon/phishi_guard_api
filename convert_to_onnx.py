import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load the trained model and vectorizer
model = joblib.load('backend/model/phishing_model.pkl')
tfidf = joblib.load('backend/model/tfidf_vectorizer.pkl')

# Define input type (adjust 5000 to match TF-IDF max_features)
initial_type = [('float_input', FloatTensorType([None, 5000]))]

# Convert to ONNX
onnx_model = convert_sklearn(model, initial_types=initial_type)
with open("backend/model/phishing_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())





    