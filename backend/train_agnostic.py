import os
import glob
import librosa
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

HC_DIR = r"H:\Parkinson ML-Model\HC_AH"
PD_DIR = r"H:\Parkinson ML-Model\PD_AH"
MODEL_DIR = r"H:\Parkinson ML-Model\backend\custom_model"

def extract_pure_vocal_tract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        
        y_trim, _ = librosa.effects.trim(y, top_db=20)
        if len(y_trim) < sr * 0.5:
            y_trim = y
        y = y_trim
        
        # 1. Dynamic MFCCs (Deltas and Delta-Deltas) - completely cancels out static mic frequency response!
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        delta_mean = np.mean(delta_mfccs, axis=1)
        delta_std = np.std(delta_mfccs, axis=1)
        delta2_mean = np.mean(delta2_mfccs, axis=1)
        delta2_std = np.std(delta2_mfccs, axis=1)

        # We completely drop Jitter, Shimmer, and HNR because they are highly dependent on the microphone's inherent noise floor.
        
        features = []
        features.extend(delta_mean)
        features.extend(delta_std)
        features.extend(delta2_mean)
        features.extend(delta2_std)
            
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_data():
    X = []
    y = []
    
    print("Loading Healthy samples...")
    for f in glob.glob(os.path.join(HC_DIR, "*.wav")) + glob.glob(os.path.join(HC_DIR, "*.mp3")):
        features = extract_pure_vocal_tract_features(f)
        if features:
            X.append(features)
            y.append(0)
            
    print("Loading PD samples...")
    for f in glob.glob(os.path.join(PD_DIR, "*.wav")) + glob.glob(os.path.join(PD_DIR, "*.mp3")):
        features = extract_pure_vocal_tract_features(f)
        if features:
            X.append(features)
            y.append(1)
            
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = load_data()
    
    if len(X) == 0:
        exit(1)
        
    print(f"Loaded {len(X)} samples. Feature length: {X.shape[1]}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
    xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, scale_pos_weight=0.9, random_state=42)
    svc = SVC(probability=True, kernel='rbf', C=1.0, class_weight='balanced', random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb_clf), ('svc', svc)],
        voting='soft'
    )
    
    ensemble.fit(X_train_scaled, y_train)
    
    y_pred = ensemble.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred))
    
    model_path = os.path.join(MODEL_DIR, "robust_ensemble.joblib")
    scaler_path = os.path.join(MODEL_DIR, "robust_scaler.joblib")
    
    joblib.dump(ensemble, model_path)
    joblib.dump(scaler, scaler_path)
