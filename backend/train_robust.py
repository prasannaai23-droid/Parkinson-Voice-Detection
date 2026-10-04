import os
import glob
import librosa
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

HC_DIR = r"H:\Parkinson ML-Model\HC_AH"
PD_DIR = r"H:\Parkinson ML-Model\PD_AH"
MODEL_DIR = r"H:\Parkinson ML-Model\backend\custom_model"

def extract_robust_features(file_path):
    try:
        # Load audio with 22050 Hz
        y, sr = librosa.load(file_path, sr=22050)
        
        # Pre-emphasis filter to boost high frequencies
        y_pre = librosa.effects.preemphasis(y)
        
        # Trim silence
        y_trim, _ = librosa.effects.trim(y_pre, top_db=20)
        if len(y_trim) < sr * 0.5:
            y_trim = y_pre # if too short, use original
            
        y = y_trim
        
        # Extract features
        # 1. MFCCs (40 coeffs)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # Delta and Delta-Delta MFCCs
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        delta_mean = np.mean(delta_mfccs, axis=1)
        delta2_mean = np.mean(delta2_mfccs, axis=1)
        
        # 2. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)[0]
        
        # 3. F0 (Fundamental Frequency) for Jitter/Shimmer proxy
        f0, _, _ = librosa.pyin(y, fmin=75, fmax=600)
        
        f0_valid = f0[~np.isnan(f0)]
        if len(f0_valid) > 1:
            f0_mean = np.mean(f0_valid)
            f0_std = np.std(f0_valid)
            periods = 1.0 / f0_valid
            period_diffs = np.abs(np.diff(periods))
            jitter = (np.mean(period_diffs) / np.mean(periods)) * 100 if np.mean(periods) > 0 else 0
        else:
            f0_mean = 0
            f0_std = 0
            jitter = 0
            
        # Shimmer proxy
        rms = librosa.feature.rms(y=y)[0]
        if len(rms) > 1:
            rms_diffs = np.abs(np.diff(rms))
            shimmer = (np.mean(rms_diffs) / np.mean(rms)) * 100 if np.mean(rms) > 0 else 0
        else:
            shimmer = 0

        # HNR proxy
        harmonic = librosa.effects.harmonic(y)
        hnr = np.mean(harmonic) if len(harmonic) > 0 else 0

        features = [
            np.mean(spectral_centroid),
            np.std(spectral_centroid),
            np.mean(spectral_bandwidth),
            np.mean(spectral_rolloff),
            np.mean(zero_crossing_rate),
            np.mean(rms),
            np.std(rms),
            f0_mean,
            f0_std,
            jitter,
            shimmer,
            hnr
        ]
        
        features.extend(np.mean(spectral_contrast, axis=1))
        features.extend(mfccs_mean)
        features.extend(mfccs_std)
        features.extend(delta_mean)
        features.extend(delta2_mean)
            
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
        features = extract_robust_features(f)
        if features:
            X.append(features)
            y.append(0)
            
    # Load PD
    print("Loading PD samples...")
    for f in glob.glob(os.path.join(PD_DIR, "*.wav")) + glob.glob(os.path.join(PD_DIR, "*.mp3")):
        features = extract_robust_features(f)
        if features:
            X.append(features)
            y.append(1)
            
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("Starting robust feature extraction...")
    X, y = load_data()
    
    if len(X) == 0:
        print("No data found!")
        exit(1)
        
    print(f"Loaded {len(X)} samples. Feature length: {X.shape[1]}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Ensemble
    print("Training Robust Ensemble Model...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    svc = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb_clf), ('svc', svc), ('gb', gb)],
        voting='soft'
    )
    
    # Evaluate with CV
    cv = StratifiedKFold(n_splits=5)
    cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    print(f"CV Accuracy: {np.mean(cv_scores)*100:.2f}% (+/- {np.std(cv_scores)*100:.2f}%)")
    
    ensemble.fit(X_train_scaled, y_train)
    
    y_pred = ensemble.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred))
    
    model_path = os.path.join(MODEL_DIR, "robust_ensemble.joblib")
    scaler_path = os.path.join(MODEL_DIR, "robust_scaler.joblib")
    
    joblib.dump(ensemble, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Model saved to {model_path}")
    print("Robust training complete!")
