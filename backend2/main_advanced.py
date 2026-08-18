"""
ADVANCED PARKINSONS VOICE DETECTION BACKEND
============================================
Uses ensemble ML models:
- Advanced Neural Network
- Random Forest
- XGBoost
With 190+ advanced acoustic features
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
import os
import tensorflow as tf
import joblib
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Global model variables
advanced_nn_model = None
advanced_rf_model = None
advanced_xgb_model = None
advanced_scaler = None

def try_import_advanced():
    """Try to import advanced feature extraction"""
    try:
        from advanced_features import extract_advanced_features
        return extract_advanced_features
    except:
        print("⚠ Advanced features module not available, using fallback")
        from feature_utils import extract_voice_features
        return extract_voice_features


def load_advanced_models():
    """Load all three trained models for ensemble prediction"""
    global advanced_nn_model, advanced_rf_model, advanced_xgb_model, advanced_scaler
    
    print("\n🚀 Loading Advanced Ensemble Models...")
    
    models_loaded = 0
    
    # Try to load Neural Network
    try:
        if os.path.exists("models/advanced_nn.keras"):
            print("  Loading Neural Network...")
            advanced_nn_model = tf.keras.models.load_model("models/advanced_nn.keras")
            print("  ✓ Neural Network loaded")
            models_loaded += 1
        else:
            print("  ⚠ advanced_nn.keras not found")
    except Exception as e:
        print(f"  ✗ Error loading NN: {e}")
    
    # Try to load Random Forest
    try:
        if os.path.exists("models/advanced_rf.pkl"):
            print("  Loading Random Forest...")
            advanced_rf_model = joblib.load("models/advanced_rf.pkl")
            print("  ✓ Random Forest loaded")
            models_loaded += 1
        else:
            print("  ⚠ advanced_rf.pkl not found")
    except Exception as e:
        print(f"  ✗ Error loading RF: {e}")
    
    # Try to load XGBoost
    try:
        if os.path.exists("models/advanced_xgb.pkl"):
            print("  Loading XGBoost...")
            advanced_xgb_model = joblib.load("models/advanced_xgb.pkl")
            print("  ✓ XGBoost loaded")
            models_loaded += 1
        else:
            print("  ⚠ advanced_xgb.pkl not found")
    except Exception as e:
        print(f"  ✗ Error loading XGBoost: {e}")
    
    # Load scaler
    try:
        if os.path.exists("models/advanced_scaler.pkl"):
            print("  Loading Feature Scaler...")
            advanced_scaler = joblib.load("models/advanced_scaler.pkl")
            print("  ✓ Scaler loaded")
        else:
            print("  ⚠ advanced_scaler.pkl not found")
    except Exception as e:
        print(f"  ✗ Error loading scaler: {e}")
    
    if models_loaded >= 3:
        print(f"\n✓ ADVANCED ENSEMBLE READY ({models_loaded}/3 models loaded)")
        return True
    elif models_loaded > 0:
        print(f"\n⚠ PARTIAL MODELS LOADED ({models_loaded}/3) - Train models first!")
        return True
    else:
        print("\n⚠ NO ADVANCED MODELS FOUND - Run: python train_advanced_model.py")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown handler"""
    print("\n" + "█"*60)
    print("█ PARKINSON'S VOICE DETECTION - ADVANCED BACKEND")
    print("█"*60)
    
    # Load advanced models on startup
    models_available = load_advanced_models()
    
    if models_available:
        print("\n✓ Backend ready for advanced predictions")
    else:
        print("\n⚠ Backend running in limited mode")
        print("   To enable advanced models:")
        print("   1. pip install -r requirements.txt")
        print("   2. python train_advanced_model.py")
    
    yield
    
    print("\n✓ Backend shutting down...")


app = FastAPI(
    title="Advanced Parkinson's Voice Detection",
    description="Professional ML ensemble for voice analysis",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Advanced Parkinson's Voice Detection",
        "version": "2.0.0",
        "models_loaded": {
            "neural_network": advanced_nn_model is not None,
            "random_forest": advanced_rf_model is not None,
            "xgboost": advanced_xgb_model is not None,
            "scaler": advanced_scaler is not None
        }
    }


@app.post("/predict")
async def predict(voice: UploadFile = File(...)):
    """
    Advanced ensemble prediction using:
    - Neural Network (40% weight)
    - Random Forest (30% weight)
    - XGBoost (30% weight)
    """
    try:
        if not voice:
            return {"status": "error", "message": "Please upload a voice file"}

        # Save voice file temporarily
        temp_voice_path = f"temp_voice_{voice.filename}"
        with open(temp_voice_path, "wb") as f:
            f.write(await voice.read())

        try:
            print(f"\n🎤 Processing voice: {voice.filename}")
            
            # Extract advanced features
            extract_func = try_import_advanced()
            features_df = extract_func(temp_voice_path)
            
            if features_df is None or features_df.shape[0] == 0:
                return {"status": "error", "message": "Failed to extract voice features"}
            
            features_array = features_df.values.astype(np.float32)
            print(f"  ✓ Features extracted: shape {features_array.shape}")
            
            # Make predictions using ensemble
            predictions = {}
            probabilities = {}
            
            # Check if at least some models are loaded
            if advanced_scaler is None:
                return {
                    "status": "error",
                    "message": "Models not trained yet. Run: python train_advanced_model.py"
                }
            
            # Scale features
            features_scaled = advanced_scaler.transform(features_array)
            
            # Collect individual predictions
            all_probs = []
            
            # Neural Network prediction
            if advanced_nn_model is not None:
                try:
                    nn_pred = float(advanced_nn_model.predict(features_scaled, verbose=0)[0][0])
                    probabilities['neural_network'] = nn_pred
                    all_probs.append(nn_pred)
                    print(f"  ✓ Neural Network: {nn_pred:.4f}")
                except Exception as e:
                    print(f"  ✗ NN error: {e}")
            
            # Random Forest prediction
            if advanced_rf_model is not None:
                try:
                    rf_pred = float(advanced_rf_model.predict_proba(features_scaled)[0, 1])
                    probabilities['random_forest'] = rf_pred
                    all_probs.append(rf_pred)
                    print(f"  ✓ Random Forest: {rf_pred:.4f}")
                except Exception as e:
                    print(f"  ✗ RF error: {e}")
            
            # XGBoost prediction
            if advanced_xgb_model is not None:
                try:
                    xgb_pred = float(advanced_xgb_model.predict_proba(features_scaled)[0, 1])
                    probabilities['xgboost'] = xgb_pred
                    all_probs.append(xgb_pred)
                    print(f"  ✓ XGBoost: {xgb_pred:.4f}")
                except Exception as e:
                    print(f"  ✗ XGBoost error: {e}")
            
            # Ensemble weighted average
            if len(all_probs) >= 2:
                if len(all_probs) == 3:
                    # All models available - use weighted average
                    ensemble_prob = (0.4 * probabilities.get('neural_network', 0.5) +
                                   0.3 * probabilities.get('random_forest', 0.5) +
                                   0.3 * probabilities.get('xgboost', 0.5))
                else:
                    # Fallback to simple average
                    ensemble_prob = np.mean(all_probs)
            else:
                # Single model available
                ensemble_prob = all_probs[0] if all_probs else 0.5
            
            print(f"  ✓ Ensemble: {ensemble_prob:.4f}")
            
            # Convert to percentage
            pd_probability = ensemble_prob * 100
            
            # Risk classification
            if pd_probability < 20:
                risk = "Very Low Risk"
            elif pd_probability < 35:
                risk = "Low Risk"
            elif pd_probability < 50:
                risk = "Low-Medium Risk"
            elif pd_probability < 60:
                risk = "Medium Risk"
            elif pd_probability < 75:
                risk = "High Risk"
            else:
                risk = "Very High Risk"
            
            # Extract key features for display
            key_features = {}
            for col in features_df.columns:
                if any(x in col for x in ['jitter', 'shimmer', 'hnr', 'f0_', 'spectral', 'energy']):
                    try:
                        key_features[col] = float(features_df[col].values[0])
                    except:
                        pass
            
            # Limit to top features
            key_features = dict(list(key_features.items())[:15])
            
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
            
            print(f"\n✅ Final Result: {risk} ({pd_probability:.1f}%)")
            
            return {
                "status": "success",
                "pd_probability": round(pd_probability, 1),
                "risk_level": risk,
                "confidence_score": round(confidence_score, 1),
                "risk_breakdown": {k: round(v, 1) for k, v in risk_breakdown.items()},
                "model_predictions": {k: round(v*100, 1) for k, v in probabilities.items()},
                "key_features": {k: round(v, 4) for k, v in key_features.items()},
                "disclaimer": "Advanced AI prediction for research only. Not a medical diagnosis.",
                "version": "2.0.0 - Advanced Ensemble"
            }
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_voice_path):
                os.remove(temp_voice_path)

    except Exception as e:
        import traceback
        print(f"❌ Error in prediction: {str(e)}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
