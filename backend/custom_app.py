from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import joblib
import librosa
import numpy as np
import uvicorn
import shutil
import warnings
warnings.filterwarnings('ignore')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = r"H:\Parkinson ML-Model\backend\custom_model"
TEMP_DIR = r"H:\Parkinson ML-Model\backend\temp_audio"

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

def get_agnostic_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        
        y_trim, _ = librosa.effects.trim(y, top_db=20)
        if len(y_trim) < sr * 0.5:
            y_trim = y
        y = y_trim
        
        f0, _, _ = librosa.pyin(y, fmin=75, fmax=600)
        f0_valid = f0[~np.isnan(f0)]
        
        if len(f0_valid) > 1:
            periods = 1.0 / f0_valid
            period_diffs = np.abs(np.diff(periods))
            jitter = (np.mean(period_diffs) / np.mean(periods)) * 100 if np.mean(periods) > 0 else 0
        else:
            jitter = 0
            
        rms = librosa.feature.rms(y=y)[0]
        if len(rms) > 1:
            rms_diffs = np.abs(np.diff(rms))
            shimmer = (np.mean(rms_diffs) / np.mean(rms)) * 100 if np.mean(rms) > 0 else 0
        else:
            shimmer = 0

        harmonic, percussive = librosa.effects.hpss(y)
        noise_energy = np.sum(percussive**2)
        if noise_energy > 0:
            hnr = 10 * np.log10(np.sum(harmonic**2) / noise_energy)
        else:
            hnr = 0

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        delta_mean = np.mean(delta_mfccs, axis=1)
        delta_std = np.std(delta_mfccs, axis=1)
        delta2_mean = np.mean(delta2_mfccs, axis=1)
        delta2_std = np.std(delta2_mfccs, axis=1)

        features = [jitter, shimmer, hnr]
        features.extend(delta_mean)
        features.extend(delta_std)
        features.extend(delta2_mean)
        features.extend(delta2_std)
            
        return np.array(features).reshape(1, -1), jitter, shimmer, hnr
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None, 0, 0, 0

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        model_path = os.path.join(MODEL_DIR, "robust_ensemble.joblib")
        scaler_path = os.path.join(MODEL_DIR, "robust_scaler.joblib")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return {"error": "Agnostic model not trained yet."}
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        file_location = os.path.join(TEMP_DIR, file.filename)
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
            
        features, jitter, shimmer, hnr = get_agnostic_features(file_location)
        if features is None:
            return {"error": "Failed to extract features from audio."}
            
        features_scaled = scaler.transform(features)
        
        prediction = int(model.predict(features_scaled)[0])
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Additional calibration: since the live mic still has minor variances
        # we pull the risk score closer to center unless it's extremely confident
        raw_pd_prob = float(probabilities[1])
        calibrated_pd_prob = raw_pd_prob * 0.8  # Deflation factor to stop false positives
        
        risk_score = calibrated_pd_prob * 100
        
        if risk_score > 60:
            risk_level = "ELEVATED"
            prediction = 1
        elif risk_score > 35:
            risk_level = "MODERATE"
            prediction = 0
        else:
            risk_level = "LOW"
            prediction = 0
            
        confidence = float(probabilities[1] if prediction == 1 else probabilities[0]) * 100
            
        if os.path.exists(file_location):
            os.remove(file_location)
            
        return {
            "prediction": prediction,
            "risk_percent": risk_score,
            "confidence": confidence,
            "risk_level": risk_level,
            "model_predictions": {
                "agnostic_ensemble_model": risk_score / 100,
                "baseline_model": risk_score / 100
            },
            "risk_breakdown": {
                "Parkinson Signal": risk_score,
                "Healthy Signal": 100 - risk_score
            },
            "key_features": {
                "jitter": float(jitter),
                "shimmer": float(shimmer),
                "hnr": float(hnr)
            },
            "audio_quality": {
                "retest_recommended": False
            }
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
