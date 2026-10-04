import sys
import os
import numpy as np
import joblib

sys.path.append(r"H:\Parkinson ML-Model\backend")
from custom_app import get_agnostic_features

MODEL_DIR = r"H:\Parkinson ML-Model\backend\custom_model"

test_file = r"C:\Users\HP\Downloads\088e1011-ecb2-443d-acb9-e51402b05925.wav"

if not os.path.exists(test_file):
    print("File not found.")
    exit(1)

features, jitter, shimmer, hnr = get_agnostic_features(test_file)
if features is None:
    print("Failed to extract features.")
    exit(1)

model = joblib.load(os.path.join(MODEL_DIR, "robust_ensemble.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "robust_scaler.joblib"))

scaled = scaler.transform(features)
prob = model.predict_proba(scaled)[0]

print(f"Prediction Probabilities: HC={prob[0]:.2f}, PD={prob[1]:.2f}")
print(f"Jitter: {jitter:.4f}")
print(f"Shimmer: {shimmer:.4f}")
print(f"HNR: {hnr:.4f}")

print("\nScaled Features (Outliers > 2 std dev):")
for i, s in enumerate(scaled[0]):
    if abs(s) > 2:
        print(f"Feature {i}: {s:.2f}")

