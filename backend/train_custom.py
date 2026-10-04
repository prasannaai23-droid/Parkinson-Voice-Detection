import os
import glob
import librosa
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

HC_DIR = r"H:\Parkinson ML-Model\HC_AH"
PD_DIR = r"H:\Parkinson ML-Model\PD_AH"
MODEL_DIR = r"H:\Parkinson ML-Model\backend\custom_model"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        
        # Extract features
        # 1. MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # 2. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)[0]
        
        # 3. F0 (Fundamental Frequency) for Jitter/Shimmer proxy
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        
        # Handle nans in f0
        f0_valid = f0[~np.isnan(f0)]
        if len(f0_valid) > 1:
            f0_mean = np.mean(f0_valid)
            f0_std = np.std(f0_valid)
            
            # Proxy for jitter (period-to-period variability)
            periods = 1.0 / f0_valid
            period_diffs = np.abs(np.diff(periods))
            jitter = np.mean(period_diffs) / np.mean(periods) if np.mean(periods) > 0 else 0
        else:
            f0_mean = 0
            f0_std = 0
            jitter = 0
            
        # Shimmer proxy (amplitude variability)
        rms = librosa.feature.rms(y=y)[0]
        if len(rms) > 1:
            rms_diffs = np.abs(np.diff(rms))
            shimmer = np.mean(rms_diffs) / np.mean(rms) if np.mean(rms) > 0 else 0
        else:
            shimmer = 0

        # Harmonic-to-Noise Ratio (HNR) proxy
        hnr = np.mean(librosa.effects.harmonic(y)) if len(y) > 0 else 0

        features = {
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'zcr_mean': np.mean(zero_crossing_rate),
            'f0_mean': f0_mean,
            'f0_std': f0_std,
            'jitter_proxy': jitter,
            'shimmer_proxy': shimmer,
            'hnr_proxy': hnr
        }
        
        for i in range(20):
            features[f'mfcc_{i+1}_mean'] = mfccs_mean[i]
            features[f'mfcc_{i+1}_std'] = mfccs_std[i]
            
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_data():
    X = []
    y = []
    
    # Load Healthy
    print("Loading Healthy samples...")
    for f in glob.glob(os.path.join(HC_DIR, "*.wav")) + glob.glob(os.path.join(HC_DIR, "*.mp3")):
        features = extract_features(f)
        if features:
            X.append(features)
            y.append(0)  # Healthy = 0
            
    # Load PD
    print("Loading PD samples...")
    for f in glob.glob(os.path.join(PD_DIR, "*.wav")) + glob.glob(os.path.join(PD_DIR, "*.mp3")):
        features = extract_features(f)
        if features:
            X.append(features)
            y.append(1)  # PD = 1
            
    return pd.DataFrame(X), np.array(y)

if __name__ == "__main__":
    print("Starting feature extraction and training...")
    X, y = load_data()
    
    if len(X) == 0:
        print("No data found! Please check the directories.")
        exit(1)
        
    print(f"Loaded {len(X)} samples in total (Class 0: {sum(y==0)}, Class 1: {sum(y==1)}).")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    print("Training XGBoost Model...")
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred))
    
    # Save Model & Scaler
    model_path = os.path.join(MODEL_DIR, "xgb_model.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    features_path = os.path.join(MODEL_DIR, "feature_names.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(X.columns.tolist(), features_path)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print("Training complete!")
