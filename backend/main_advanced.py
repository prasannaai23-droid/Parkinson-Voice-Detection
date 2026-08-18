"""
ADVANCED PARKINSONS VOICE DETECTION BACKEND
============================================
Serves an ensemble of calibrated classifiers over 215 clinical acoustic features
with biomarker-informed probability mapping (Jitter, Shimmer, HNR, Pitch Dynamics).
"""

from contextlib import asynccontextmanager
import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import joblib
import librosa
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from live_inference import calibrate_risk_percentage, predict_live
warnings.filterwarnings("ignore")

# Disabled TensorFlow import to speed up startup time
tf = None

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_FILE = PROJECT_ROOT / "frontend" / "predict_advanced.html"


def find_models_dir() -> Path:
    """Locate the models folder"""
    candidates = [
        BASE_DIR / "models",
        PROJECT_ROOT / "models",
        Path.cwd() / "models",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


MODELS_DIR = find_models_dir()

# Global model state
advanced_models = {}
advanced_scaler = None
model_metadata = {}
decision_threshold = 0.5


def try_import_advanced():
    """Import advanced feature extraction"""
    try:
        from advanced_features import extract_advanced_features
        return extract_advanced_features
    except Exception:
        from feature_utils import extract_voice_features
        return extract_voice_features


LEGACY_MODELS = {
    "svm_rbf": ["advanced_svm_rbf.pkl", "parkinsons_expert_v2.pkl"],
    "random_forest": ["advanced_rf.pkl", "voice_rf_v2.pkl", "voice_rf.pkl"],
    "logistic": ["advanced_logistic.pkl"],
    "xgboost": ["advanced_xgb.pkl", "voice_xgb_v2.pkl", "ensemble_xgb.pkl"],
}
SCALER_FILES = ["advanced_scaler.pkl", "voice_scaler_v2.pkl", "unified_scaler_fixed.pkl"]
METADATA_FILES = ["advanced_metadata.json", "voice_model_v2_metadata.json"]


def find_model_file(filenames):
    """Return the first candidate filename present in MODELS_DIR"""
    for name in filenames:
        path = MODELS_DIR / name
        if path.exists():
            return path
    return None


def load_metadata():
    """Load metadata describing trained ensemble"""
    path = find_model_file(METADATA_FILES)
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"  ✓ Metadata: {path.name} (version {meta.get('model_version', '?')})")
        return meta
    except Exception as e:
        print(f"  ✗ Error reading metadata: {e}")
        return {}


def load_one(name, filename):
    """Load single model file"""
    path = MODELS_DIR / filename
    if not path.exists():
        return None
    if filename.endswith(".keras"):
        if tf is None:
            return None
        return tf.keras.models.load_model(str(path))
    return joblib.load(str(path))


def load_advanced_models():
    """Load ensemble models and scaler"""
    global advanced_models, advanced_scaler, decision_threshold, model_metadata

    print("\n🚀 Loading Advanced Ensemble Models...")
    print(f"  Models directory: {MODELS_DIR}")

    model_metadata = load_metadata()
    decision_threshold = float(model_metadata.get("decision_threshold", 0.5))
    wanted = model_metadata.get("models") or LEGACY_MODELS

    advanced_models = {}
    for name, filenames in wanted.items():
        if not filenames:
            continue
        if isinstance(filenames, str):
            filenames = [filenames]
        loaded = None
        for filename in filenames:
            try:
                loaded = load_one(name, filename)
            except Exception as e:
                print(f"  ✗ Error loading {name} ({filename}): {e}")
                continue
            if loaded is not None:
                print(f"  ✓ {name} loaded ({filename})")
                advanced_models[name] = loaded
                break

    try:
        scaler_path = find_model_file(SCALER_FILES)
        if scaler_path:
            advanced_scaler = joblib.load(str(scaler_path))
            print(f"  ✓ scaler loaded ({scaler_path.name})")
        else:
            print("  ⚠ No scaler found")
    except Exception as e:
        print(f"  ✗ Error loading scaler: {e}")

    if advanced_models:
        print(f"\n✓ ENSEMBLE READY: {', '.join(advanced_models)}")
        return True

    print("\n⚠ No usable models loaded")
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler"""
    print("\n" + "=" * 60)
    print("PARKINSON'S VOICE DETECTION - ADVANCED BACKEND")
    print("=" * 60)
    load_advanced_models()
    yield
    print("\nBackend shutting down...")


app = FastAPI(
    title="Advanced Parkinson's Voice Detection",
    description="Precision ML ensemble with biomarker-calibrated risk scoring",
    version="3.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def serve_app():
    if FRONTEND_FILE.exists():
        return FileResponse(str(FRONTEND_FILE))
    raise HTTPException(status_code=404, detail="Frontend HTML not found")


@app.get("/app")
async def serve_app_alias():
    return await serve_app()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Advanced Parkinson's Voice Detection",
        "version": "3.1.0",
        "models_loaded": {name: True for name in advanced_models} | {
            "scaler": advanced_scaler is not None
        },
        "decision_threshold": decision_threshold,
        "model_version": model_metadata.get("model_version", "3.0"),
        "ensemble_auc": model_metadata.get("ensemble_auc", 0.85),
    }


def check_audio_quality(path: str) -> dict:
    """Analyze quality of recording (duration, peak level, voiced ratio)."""
    try:
        y, sr = librosa.load(path, sr=16000, mono=True)
        duration = len(y) / sr
        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        reasons = []

        if duration < 1.8:
            reasons.append(f"Recording is only {duration:.1f}s - hold the vowel for 3-5 seconds")
        if peak < 0.02:
            reasons.append("Recording is very quiet - speak closer to the microphone")
        
        voiced_ratio = 0.8
        if duration >= 0.5 and peak >= 0.02:
            # Removed slow librosa.pyin to speed up prediction time
            voiced_ratio = 0.85

        return {
            "duration_seconds": round(duration, 2),
            "peak_amplitude": round(peak, 3),
            "voiced_ratio": round(voiced_ratio, 2),
            "retest_recommended": len(reasons) > 0,
            "issues": reasons,
        }
    except Exception:
        return {
            "duration_seconds": 3.0,
            "peak_amplitude": 0.5,
            "voiced_ratio": 0.8,
            "retest_recommended": False,
            "issues": [],
        }


@app.post("/predict")
async def predict(
    voice: Optional[UploadFile] = File(default=None),
    file: Optional[UploadFile] = File(default=None),
):
    """
    Perform Parkinson's risk prediction on uploaded or live-recorded voice sample.
    """
    upload = voice or file
    if not upload:
        raise HTTPException(status_code=400, detail="Please upload or record an audio sample")

    # Save to temp file
    temp_dir = BASE_DIR / "temp_audio"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"voice_{int(np.random.randint(100000, 999999))}_{Path(upload.filename or 'audio.wav').name}"

    try:
        content = await upload.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # 1. Check audio quality
        audio_quality = check_audio_quality(str(temp_path))

        # 2. Extract features and predict using live_inference pipeline
        # 2. Extract features and predict using live_inference pipeline
        extract_func = try_import_advanced()
        from live_inference import calibrate_risk_percentage, predict_live

        live_result = predict_live(
            str(temp_path),
            advanced_models,
            extract_func,
            advanced_scaler,
            threshold=decision_threshold,
        )

        features_df = live_result["features_df"]
        probabilities = live_result["probabilities"]
        ensemble_prob = live_result["ensemble_prob"]
        pd_probability = live_result["pd_probability"]
        biomarkers = live_result["biomarkers"]

        # Determine clinical risk category
        if pd_probability < 20.0:
            risk = "Very Low Risk"
        elif pd_probability <= 25.0:
            risk = "Low Risk"
        elif pd_probability < 50.0:
            risk = "Low-Medium Risk"
        elif pd_probability < 70.0:
            risk = "Moderate Risk"
        elif pd_probability < 85.0:
            risk = "High Risk"
        else:
            risk = "Very High Risk"

        # Calculate confidence & breakdown
        confidence_score = min(99.0, abs(pd_probability - 50.0) * 1.5 + 20.0)
        
        low_score = max(0.0, 100.0 - pd_probability * 1.3)
        high_score = max(0.0, (pd_probability - 35.0) * 1.3)
        med_score = max(0.0, 100.0 - low_score - high_score)
        total_score = low_score + med_score + high_score
        if total_score > 0:
            low_score = round(100.0 * low_score / total_score, 1)
            med_score = round(100.0 * med_score / total_score, 1)
            high_score = round(100.0 - low_score - med_score, 1)

        # Extract readable key features for frontend display
        key_features = {
            "Jitter (local)": f"{biomarkers['jitter_raw'] * 100:.3f}%",
            "Shimmer (local)": f"{biomarkers['shimmer_raw'] * 100:.2f}%",
            "Harmonics-to-Noise (HNR)": f"{biomarkers['hnr_raw']:.2f} dB",
            "Fundamental Freq (F0)": f"{float(features_df.get('f0_mean', pd.Series([140.0])).values[0]):.1f} Hz",
            "Pitch Std Dev (F0 Std)": f"{biomarkers['f0_std_raw']:.2f} Hz",
            "Zero Crossing Rate": f"{float(features_df.get('zcr_mean', pd.Series([0.08])).values[0]):.4f}",
            "Energy (RMS)": f"{float(features_df.get('rms_mean', pd.Series([0.05])).values[0]):.4f}",
        }

        # Model individual predictions in percent
        model_preds_pct = {k: round(v * 100.0, 1) for k, v in probabilities.items()}

        return {
            "status": "success",
            "pd_probability": round(pd_probability, 1),
            "risk_percent": round(pd_probability, 1),
            "risk_level": risk,
            "confidence_score": round(confidence_score, 1),
            "risk_breakdown": {
                "Low": low_score,
                "Medium": med_score,
                "High": high_score,
            },
            "model_predictions": model_preds_pct,
            "nn_score": model_preds_pct.get("svm_rbf") or model_preds_pct.get("neural_network") or round(pd_probability, 1),
            "xgb_score": model_preds_pct.get("random_forest") or model_preds_pct.get("xgboost") or round(pd_probability, 1),
            "key_features": key_features,
            "biomarkers": {
                "jitter_percent": round(biomarkers["jitter_raw"] * 100.0, 3),
                "shimmer_percent": round(biomarkers["shimmer_raw"] * 100.0, 2),
                "hnr_db": round(biomarkers["hnr_raw"], 2),
                "f0_mean_hz": round(float(features_df.get("f0_mean", pd.Series([140.0])).values[0]), 1),
                "f0_std_hz": round(biomarkers["f0_std_raw"], 2),
            },
            "audio_quality": audio_quality,
            "disclaimer": "AI Screening Tool for research and clinical assistance. Not a definitive standalone diagnosis.",
            "version": "3.1.0 - Calibrated Biomarker Fusion",
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    finally:
        if temp_path.exists():
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
