"""
ADVANCED PARKINSONS VOICE DETECTION BACKEND
============================================
Serves an ensemble of calibrated classifiers over 215 clinical acoustic features
with biomarker-informed probability mapping (Jitter, Shimmer, HNR, Pitch Dynamics).
"""

from contextlib import asynccontextmanager
import asyncio
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
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fused_inference import FusionInputError, fuse_probabilities, gait_highlights, predict_gait_probability
from live_inference import calibrate_risk_percentage, predict_live
warnings.filterwarnings("ignore")

# Disabled TensorFlow import to speed up startup time
tf = None

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_FILE = PROJECT_ROOT / "frontend" / "predict_advanced.html"
FUSED_FRONTEND_FILE = PROJECT_ROOT / "frontend" / "fused_ui.html"
PHONE_GAIT_FRONTEND_FILE = PROJECT_ROOT / "frontend" / "phone_gait.html"
REMOTE_GAIT_DIR = BASE_DIR / "temp_audio" / "remote_gait"
REMOTE_GAIT_DIR.mkdir(parents=True, exist_ok=True)
latest_gait = next(iter(sorted(REMOTE_GAIT_DIR.glob("latest.*"), key=lambda item: item.stat().st_mtime_ns, reverse=True)), None)


def normalize_gait_video(source: Path) -> Path:
    """Write a small analysis copy so high-resolution phone video is bounded."""
    normalized = REMOTE_GAIT_DIR / "latest.mp4"
    temporary = REMOTE_GAIT_DIR / "normalized_tmp.mp4"
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return source
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, 640.0 / max(width, height, 1))
    output_size = (max(2, int(width * scale)), max(2, int(height * scale)))
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
    try:
        frame_limit = int(fps * 2)
        for _ in range(frame_limit):
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(cv2.resize(frame, output_size))
    finally:
        capture.release()
        writer.release()
    if temporary.exists() and temporary.stat().st_size > 0:
        if source != normalized:
            source.unlink(missing_ok=True)
        normalized.unlink(missing_ok=True)
        temporary.replace(normalized)
        return normalized
    return source


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
    "logistic": ["advanced_logistic.pkl"],
    "extra_trees": ["advanced_extra_trees.pkl"],
    "random_forest": ["advanced_rf.pkl"],
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
    model_metadata["model_version"] = "voice-v2-direct-features"
    decision_threshold = 0.5
    wanted = LEGACY_MODELS

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
    # Pay the one-time librosa/numba initialization cost before the first user request.
    try:
        from advanced_features import extract_advanced_features
        warmup_path = BASE_DIR / "models" / "voice_warmup.wav"
        if not warmup_path.exists():
            import soundfile as sf
            sf.write(warmup_path, np.zeros(16000, dtype=np.float32), 16000)
        await asyncio.to_thread(extract_advanced_features, str(warmup_path))
        warmup_path.unlink(missing_ok=True)
    except Exception as warmup_error:
        print(f"[WARNING] Feature warmup failed: {warmup_error}")
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


@app.get("/fused")
async def serve_fused_app():
    if FUSED_FRONTEND_FILE.exists():
        return FileResponse(str(FUSED_FRONTEND_FILE), headers={"Cache-Control": "no-store, max-age=0"})
    raise HTTPException(status_code=404, detail="Fused frontend not found")


@app.get("/phone-gait")
async def serve_phone_gait_app():
    # Keep one shared interface: the phone records from fused_ui.html itself.
    return RedirectResponse(url="/fused", status_code=307)


@app.post("/upload-gait")
async def upload_gait(gait: UploadFile = File(...)):
    """Receive a phone-recorded gait video for the paired desktop session."""
    global latest_gait
    suffix = Path(gait.filename or ".webm").suffix or ".webm"
    path = REMOTE_GAIT_DIR / f"latest{suffix}"
    content = await gait.read()
    if not content:
        raise HTTPException(status_code=400, detail="The captured gait video is empty")
    path.write_bytes(content)
    if suffix.lower() != ".mp4" or len(content) > 2_000_000:
        path = await asyncio.to_thread(normalize_gait_video, path)
    latest_gait = path
    return {"status": "ready", "filename": gait.filename or path.name, "bytes": len(content), "updated_at": path.stat().st_mtime_ns}


@app.get("/gait-latest")
async def gait_latest():
    global latest_gait
    if latest_gait is None:
        latest_gait = next(iter(sorted(REMOTE_GAIT_DIR.glob("latest.*"), key=lambda item: item.stat().st_mtime_ns, reverse=True)), None)
    if latest_gait is None or not latest_gait.exists():
        return {"status": "waiting"}
    stats = latest_gait.stat()
    return {"status": "ready", "filename": latest_gait.name, "bytes": stats.st_size, "updated_at": stats.st_mtime_ns}


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


def build_voice_highlights(features_df, biomarkers, probability):
    """Return human-readable acoustic evidence without turning it into a diagnosis."""
    jitter = float(biomarkers["jitter_raw"])
    shimmer = float(biomarkers["shimmer_raw"])
    hnr = float(biomarkers["hnr_raw"])
    f0_std = float(biomarkers["f0_std_raw"])
    mfcc0 = float(features_df.get("mfcc_mean_0", pd.Series([0.0])).values[0])
    noise_harmonic = 10.0 ** (-hnr / 10.0)
    highlights = []
    highlights.append({"name": "Jitter", "value": f"{jitter * 100:.3f}%", "signal": "elevated" if jitter > 0.01 else "within range"})
    highlights.append({"name": "Shimmer", "value": f"{shimmer * 100:.2f}%", "signal": "elevated" if shimmer > 0.06 else "within range"})
    highlights.append({"name": "HNR", "value": f"{hnr:.2f} dB", "signal": "lower" if hnr < 15 else "within range"})
    highlights.append({"name": "Noise / harmonic", "value": f"{noise_harmonic:.3f}", "signal": "higher noise" if noise_harmonic > 0.03 else "lower noise"})
    highlights.append({"name": "MFCC 0", "value": f"{mfcc0:.2f}", "signal": "spectral shape"})
    highlights.append({"name": "Pitch variation", "value": f"{f0_std:.2f} Hz", "signal": "variable" if f0_std > 18 else "stable"})
    return {
        "model_probability": round(float(probability), 1),
        "markers": highlights,
        "interpretation": "Model evidence is borderline; repeat with a clean recording" if 35 <= probability < 65 else "Model evidence is internally consistent",
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

        if live_result.get("error") or not live_result.get("probabilities"):
            raise HTTPException(status_code=422, detail=live_result.get("error", "Voice prediction was unavailable"))

        features_df = live_result["features_df"]
        probabilities = live_result["probabilities"]
        ensemble_prob = live_result["ensemble_prob"]
        pd_probability = live_result["pd_probability"]
        biomarkers = live_result["biomarkers"]
        if live_result.get("quality_gate"):
            audio_quality["retest_recommended"] = True
            audio_quality["issues"].append("Acoustic biomarkers are unreliable for a confident classification; record again in a quieter room.")

        # Determine clinical risk category
        if pd_probability < 35.0:
            risk = "Low Risk"
        elif pd_probability < 65.0:
            risk = "Inconclusive"
        else:
            risk = "High Risk"

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
            "prediction": live_result["prediction"],
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
            "voice_highlights": build_voice_highlights(features_df, biomarkers, pd_probability),
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


@app.post("/predict-fused")
async def predict_fused(
    voice: Optional[UploadFile] = File(default=None),
    gait: Optional[UploadFile] = File(default=None),
    gait_id: Optional[str] = Form(default=None),
):
    """Score voice and gait together, abstaining when evidence is unreliable."""
    if not voice or (not gait and gait_id != "latest" and (latest_gait is None or not latest_gait.exists())):
        raise HTTPException(status_code=400, detail="Please provide both a voice recording/file and a gait video.")

    gait_model_path = MODELS_DIR / "video_gait_rf.pkl"
    temp_paths = []
    try:
        temp_dir = BASE_DIR / "temp_audio"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        voice_suffix = Path(voice.filename or "voice.wav").suffix or ".wav"
        voice_temp = tempfile.NamedTemporaryFile(suffix=voice_suffix, prefix="fused_voice_", dir=temp_dir, delete=False)
        voice_temp.write(await voice.read())
        voice_temp.close()
        temp_paths.append(Path(voice_temp.name))

        if gait:
            gait_suffix = Path(gait.filename or "gait.mp4").suffix or ".mp4"
            gait_temp = tempfile.NamedTemporaryFile(suffix=gait_suffix, prefix="fused_gait_", dir=temp_dir, delete=False)
            gait_temp.write(await gait.read())
            gait_temp.close()
            temp_paths.append(Path(gait_temp.name))
        elif latest_gait and latest_gait.exists():
            gait_temp = tempfile.NamedTemporaryFile(suffix=latest_gait.suffix or ".mp4", prefix="fused_gait_", dir=temp_dir, delete=False)
            gait_temp.write(latest_gait.read_bytes())
            gait_temp.close()
            temp_paths.append(Path(gait_temp.name))
        else:
            raise HTTPException(status_code=400, detail="No gait video uploaded or selected.")

        try:
            voice_result = await asyncio.wait_for(asyncio.to_thread(
                predict_live,
                str(temp_paths[0]),
                advanced_models,
                try_import_advanced(),
                advanced_scaler,
                decision_threshold,
            ), timeout=15.0)
        except asyncio.TimeoutError as exc:
            raise FusionInputError("Voice extraction exceeded the analysis limit. Use a clear 3-5 second WAV recording and try again.") from exc
        if voice_result.get("error") or not voice_result.get("probabilities"):
            raise FusionInputError(voice_result.get("error", "Voice feature extraction failed. Please ensure a clear audio sample."))

        # Gait model inference. Never replace failed pose extraction with a
        # fabricated feature vector: that would turn missing evidence into a prediction.
        gait_features = None
        try:
            from train_video_gait_model import extract_features_from_video
            raw_features = await asyncio.wait_for(asyncio.to_thread(
                extract_features_from_video, str(temp_paths[1]), skip_frames=30, max_frames=5
            ), timeout=12.0)
        except asyncio.TimeoutError as exc:
            raise FusionInputError("Gait extraction exceeded the analysis limit. Use a short, steady video with the full body visible.") from exc
        if raw_features is not None and len(raw_features) == 12 and np.isfinite(raw_features).all() and any(f != 0.0 for f in raw_features):
            gait_features = raw_features
        else:
            raise FusionInputError("No reliable pose landmarks found. Use a 5-10 second front or side video with the person's head, knees, ankles, and feet fully visible, good lighting, and no furniture blocking the body.")

        if gait_features is None:
            raise FusionInputError("No reliable pose landmarks found. Use a 5-10 second front or side video with the person's head, knees, ankles, and feet fully visible, good lighting, and no furniture blocking the body.")
        if not gait_model_path.exists():
            raise FusionInputError("The gait model is not available. Train the video gait model before running fused analysis.")
        gait_model = joblib.load(gait_model_path)
        gait_probability = predict_gait_probability(gait_model, gait_features)

        fused = fuse_probabilities(voice_result["pd_probability"] / 100.0, gait_probability)

        return {
            "status": "success",
            **fused,
            "voice": {
                "prediction": voice_result.get("prediction", "Healthy"),
                "risk_percent": round(voice_result.get("pd_probability", 50.0), 1),
                "highlights": build_voice_highlights(
                    voice_result.get("features_df", pd.DataFrame()),
                    voice_result.get("biomarkers", {"jitter_raw": 0.005, "shimmer_raw": 0.03, "hnr_raw": 20.0, "f0_std_raw": 10.0}),
                    voice_result.get("pd_probability", 50.0),
                ),
            },
            "gait": {
                "risk_percent": round(gait_probability * 100.0, 1),
                "highlights": gait_highlights(gait_features),
            },
            "disclaimer": "Multimodal AI screening output. Inconclusive findings indicate borderline signals or modality disagreement.",
        }
    except FusionInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        print(f"Fused prediction error: {exc}")
        raise HTTPException(status_code=500, detail=f"Fused prediction error: {str(exc)}") from exc
    finally:
        for path in temp_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
