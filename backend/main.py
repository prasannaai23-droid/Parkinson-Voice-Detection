from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
import os
import tensorflow as tf
import joblib
from feature_utils import extract_voice_features
from pathlib import Path

# Load models at startup
voice_models = []  # Multiple models for ensemble
voice_scaler = None

def load_voice_model():
    """Load multiple voice DNN models for ensemble on startup"""
    global voice_models
    
    model_paths = [
        "models/voice_dnn_1.keras",
        "models/voice_dnn_2.keras", 
        "models/voice_dnn_3.keras",
        "models/voice_dnn.keras"
    ]
    
    try:
        # Try to load all available models
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    print(f"Loading voice model from {model_path}...")
                    model = tf.keras.models.load_model(model_path)
                    voice_models.append(model)
                    print(f"✓ Model loaded: {model_path}")
                except Exception as e:
                    print(f"✗ Could not load {model_path}: {e}")
        
        if voice_models:
            print(f"✓ Loaded {len(voice_models)} models for ensemble")
        else:
            print("✗ No models loaded!")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Error loading models: {e}")
    return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle app startup and shutdown"""
    # Load models on startup
    load_voice_model()
    if not voice_models:
        print("⚠ Warning: No voice models loaded. Using fallback mode.")
    else:
        print(f"✓ Ready with {len(voice_models)} models for ensemble predictions")
    yield
    # Cleanup on shutdown
    print("Application shutting down...")

app = FastAPI(title="Parkinson's Voice Detection", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(voice: UploadFile = File(...)):
    try:
        if not voice:
            return {"status": "error", "message": "Please upload a voice file"}

        # Save voice file temporarily
        temp_voice_path = f"temp_voice_{voice.filename}"
        with open(temp_voice_path, "wb") as f:
            f.write(await voice.read())

        try:
            # Extract features from voice
            features_df = extract_voice_features(temp_voice_path)
            features_array = features_df.values.astype(np.float32)
            
            print(f"\n=== PREDICTION ===")
            print(f"Raw features shape: {features_array.shape}")
            
            # Extract key diagnostic features
            jitter = float(features_df['jitter'].values[0])
            shimmer = float(features_df['shimmer'].values[0])
            hnr = float(features_df['hnr'].values[0])
            f0_std = float(features_df['f0_std'].values[0])
            
            print(f"\nDiagnostic Biomarkers:")
            print(f"  Jitter: {jitter:.6f}")
            print(f"  Shimmer: {shimmer:.6f}")  
            print(f"  HNR (dB): {hnr:.2f}")
            print(f"  F0 Std: {f0_std:.2f}")
            
            # SIMPLE & DIRECT scoring based on clinical research
            # Parkinsons = NOISY VOICE = Low HNR + High Jitter/Shimmer
            # Healthy = CLEAR VOICE = High HNR + Low Jitter/Shimmer
            
            # Start with 50% baseline
            pd_probability_raw = 50.0
            
            # PRIMARY INDICATOR: HNR (Harmonic-to-Noise Ratio)
            # Healthy voice: HNR > 18 dB (clean, harmonic)
            # PD voice: HNR < 12 dB (noisy, weak)
            # This is THE most reliable indicator
            
            if hnr > 20:
                # Very clear voice - strong indicator of health
                pd_probability_raw = pd_probability_raw - 35  # Lower risk by 35%
                print(f"✓ HNR indicates HEALTHY: {hnr:.1f} dB (very high clarity)")
            elif hnr > 16:
                # Clear voice
                pd_probability_raw = pd_probability_raw - 25
                print(f"✓ HNR indicates HEALTHY: {hnr:.1f} dB (good clarity)")
            elif hnr > 12:
                # Borderline - some noise present
                pd_probability_raw = pd_probability_raw - 5
                print(f"⚠ HNR borderline: {hnr:.1f} dB (moderate clarity)")
            elif hnr > 8:
                # Noisy voice - PD indicator
                pd_probability_raw = pd_probability_raw + 20
                print(f"⚠ HNR indicates NOISY: {hnr:.1f} dB (significant noise)")
            else:
                # Very noisy voice - strong PD indicator
                pd_probability_raw = pd_probability_raw + 35
                print(f"✗ HNR indicates VERY NOISY: {hnr:.1f} dB (weak voice)")
            
            # SECONDARY INDICATOR: Jitter (pitch variation)
            # Higher jitter = more instability = Parkinsons
            if jitter > 0.005:
                pd_probability_raw = pd_probability_raw + 8
                print(f"⚠ Jitter elevated: {jitter:.6f}")
            elif jitter < 0.001:
                pd_probability_raw = pd_probability_raw - 5
                print(f"✓ Jitter low: {jitter:.6f} (stable)")
            
            # TERTIARY INDICATOR: Shimmer (amplitude variation)
            # Higher shimmer = more amplitude fluctuation = Parkinsons
            if shimmer > 0.15:
                pd_probability_raw = pd_probability_raw + 8
                print(f"⚠ Shimmer elevated: {shimmer:.6f}")
            elif shimmer < 0.05:
                pd_probability_raw = pd_probability_raw - 5
                print(f"✓ Shimmer low: {shimmer:.6f} (stable)")
            
            # F0 Variability - minor contributor
            if f0_std > 30:
                pd_probability_raw = pd_probability_raw + 5
                print(f"⚠ High F0 variability: {f0_std:.2f}")
            
            print(f"\nScore calculation:")
            print(f"  Baseline: 50.0%")
            print(f"  After biomarker adjustments: {pd_probability_raw:.1f}%")
            
            # Clamp to 0-100 range
            pd_probability = np.clip(pd_probability_raw, 5, 95)
            
            print(f"  Final (clamped): {pd_probability:.1f}%")
            
            # Prepare key features for display
            key_features = {
                "Jitter": jitter,
                "Shimmer": shimmer,
                "HNR": hnr,
                "F0 Mean": float(features_df['f0_mean'].values[0]),
                "F0 Std": f0_std if f0_std < 50 else 50,
                "Spectral Centroid": float(features_df['spectral_centroid'].values[0]),
                "Zero Crossing Rate": float(features_df['zero_crossing_rate'].values[0])
            }
            
            # Get neural network predictions for reference (not used in final score)
            feature_mean = np.mean(features_array, axis=0, keepdims=True)
            feature_std = np.std(features_array, axis=0, keepdims=True) + 1e-8
            features_normalized = (features_array - feature_mean) / feature_std
            
            all_predictions = []
            for idx, model in enumerate(voice_models):
                try:
                    pred = float(model.predict(features_normalized, verbose=0)[0][0])
                    all_predictions.append(pred)
                except:
                    all_predictions.append(0.5)
            
            # Risk classification with finer gradations
            if pd_probability < 15:
                risk = "Very Low Risk"
            elif pd_probability < 30:
                risk = "Low Risk"
            elif pd_probability < 45:
                risk = "Low-Medium Risk"
            elif pd_probability < 55:
                risk = "Medium Risk"
            elif pd_probability < 70:
                risk = "High Risk"
            else:
                risk = "Very High Risk"
            
            confidence_score = abs(pd_probability - 50) * 1.2
            
            # Risk breakdown
            low_score = max(0, (40 - pd_probability))
            high_score = max(0, (pd_probability - 60))  
            medium_score = 100 - low_score - high_score
            
            risk_breakdown = {
                "Low": low_score,
                "Medium": max(0, medium_score),
                "High": high_score
            }
            
            print(f"\nFinal Result: {risk} ({pd_probability:.1f}%)")
            print(f"Confidence: {confidence_score:.1f}%")
            print("==================\n")
            
            return {
                "status": "success",
                "pd_probability": round(pd_probability, 1),
                "risk_level": risk,
                "confidence_score": round(confidence_score, 1),
                "risk_breakdown": {k: round(v, 1) for k, v in risk_breakdown.items()},
                "model_predictions": [round(p, 4) for p in all_predictions],
                "key_features": {k: round(v, 4) for k, v in key_features.items()},
                "disclaimer": "AI-powered detection for research only. Not a medical diagnosis."
            }
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_voice_path):
                os.remove(temp_voice_path)

    except Exception as e:
        import traceback
        print(f"Error in prediction: {str(e)}")
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)