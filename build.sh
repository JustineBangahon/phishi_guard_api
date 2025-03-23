#!/usr/bin/env bash
# Build script for Render

# Exit on error
set -o errexit

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Ensure NLTK data is downloaded
python -c "import nltk; nltk.download('stopwords')"

# Convert scikit-learn model to ONNX (only if it doesn't exist)
if [ ! -f "backend/model/phishing_model.onnx" ]; then
    python convert_to_onnx.py
fi

echo "Build completed successfully!"