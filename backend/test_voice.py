#!/usr/bin/env python
"""
Quick test to verify voice feature extraction and model loading
"""
import os
import sys
import numpy as np
import tensorflow as tf
from feature_utils import extract_voice_features, get_default_voice_features

# Test 1: Check if model file exists
print("\n=== TEST 1: Model File Check ===")
model_path = "models/voice_dnn_1.keras"
if not os.path.exists(model_path):
    model_path = "models/voice_dnn.keras"

if os.path.exists(model_path):
    print(f"✓ Model found at: {model_path}")
    print(f"  File size: {os.path.getsize(model_path) / 1e6:.1f} MB")
else:
    print("✗ Model file not found!")
    sys.exit(1)

# Test 2: Load model
print("\n=== TEST 2: Model Loading ===")
try:
    model = tf.keras.models.load_model(model_path)
    print(f"✓ Model loaded successfully")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

# Test 3: Feature extraction with default values
print("\n=== TEST 3: Feature Extraction (Default) ===")
try:
    features_df = get_default_voice_features()
    print(f"✓ Generated default features")
    print(f"  Shape: {features_df.shape}")
    print(f"  Columns: {list(features_df.columns[:10])}...")  # First 10
    features_array = features_df.values.astype(np.float32)
    print(f"  Feature vector shape: {features_array.shape}")
except Exception as e:
    print(f"✗ Error extracting features: {e}")
    sys.exit(1)

# Test 4: Prediction with sample features
print("\n=== TEST 4: Model Prediction ===")
try:
    prediction = model.predict(features_array, verbose=0)
    probability = float(prediction[0][0] * 100)
    print(f"✓ Model prediction successful")
    print(f"  Raw output: {prediction[0][0]:.4f}")
    print(f"  Probability: {probability:.1f}%")
    
    if probability < 45:
        risk = "Low"
    elif probability < 65:
        risk = "Medium"
    else:
        risk = "High"
    print(f"  Risk level: {risk}")
except Exception as e:
    print(f"✗ Error during prediction: {e}")
    sys.exit(1)

print("\n=== ALL TESTS PASSED ===")
print("Backend is ready to use!")
