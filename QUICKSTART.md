# Parkinson's Voice Detection - Quick Start Guide

## 🎯 What Was Fixed

Your Parkinson's voice detection model wasn't working because:

1. **Backend was using FAKE predictions** - Random numbers instead of real ML
2. **Poor audio feature extraction** - Dummy MFCC values that don't represent actual voice
3. **Server stability issues** - Model loading was causing hangs
4. **Missing dependencies** - soundfile library wasn't installed

**Now FIXED!** ✓

---

## ✅ What's Working Now

### Real-Time Voice Analysis
- Loads actual trained TensorFlow model (`voice_dnn_1.keras`)
- Extracts **47 real acoustic features** using librosa:
  - Fundamental frequency (F0) with pitch tracking
  - Jitter (pitch variation)
  - Shimmer (amplitude variation)
  - Harmonic-to-Noise Ratio (HNR)
  - Spectral features
  - 40 MFCC coefficients

### Stable Backend Server
- ✓ No hangs during startup
- ✓ Proper async feature extraction
- ✓ Error handling & fallbacks
- ✓ Running on port 8001

---

## 🚀 How to Use

### 1. Start the Backend Server

```bash
cd backend
python main.py
```

**Output should show:**
```
Loading voice model from models/voice_dnn_1.keras...
✓ Voice model loaded successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001
```

### 2. Upload Voice File for Prediction

Using curl:
```bash
curl -X POST "http://localhost:8001/predict" \
  -F "voice=@path/to/voice_sample.wav"
```

Using Python:
```python
import requests

with open("voice_sample.wav", "rb") as f:
    files = {"voice": f}
    response = requests.post("http://localhost:8001/predict", files=files)
    result = response.json()
    print(f"Parkinson's Probability: {result['pd_probability']}%")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Features: {result['key_features']}")
```

### 3. Response Format

```json
{
  "status": "success",
  "pd_probability": 72.3,
  "risk_level": "High",
  "confidence_score": 44.6,
  "risk_breakdown": {
    "Low": 12.5,
    "Medium": 32.4,
    "High": 55.1
  },
  "model_used": "TensorFlow Deep Neural Network",
  "key_features": {
    "Jitter": 0.0082,
    "Shimmer": 0.0234,
    "HNR": 18.234,
    "F0 Mean": 142.5,
    "F0 Std": 22.1,
    "Spectral Centroid": 2150.0,
    "Zero Crossing Rate": 0.082
  },
  "disclaimer": "AI-powered detection for research only. Not a medical diagnosis."
}
```

---

## 📊 Voice Features Extracted

| Feature | Description | What it tells us |
|---------|-------------|------------------|
| **F0 Mean** | Average fundamental frequency | Pitch level |
| **F0 Std** | Pitch variation | Pitch stability |
| **Jitter** | Period-to-period pitch variation | Voice quality indicator |
| **Shimmer** | Amplitude variation | Speech roughness |
| **HNR** | Harmonic-to-Noise Ratio | Periodicity (higher = cleaner voice) |
| **Spectral Centroid** | Center of spectral power | Tonal characteristics |
| **Zero Crossing Rate** | Signal sign changes | Noise/speech ratio |
| **MFCC (0-19)** | Mel-Frequency Cepstral Coeff. | Voice timbre features |

---

## 🔧 Technical Details

### Model Information
- **Model File**: `backend/models/voice_dnn_1.keras`
- **Architecture**: Deep Neural Network
- **Input**: 47 acoustic features
- **Output**: Probability score (0-1)
- **File Size**: 2.5 MB

### Risk Levels
- **Low**: < 45% probability
- **Medium**: 45-65% probability
- **High**: > 65% probability

### System Requirements
- Python 3.8+
- TensorFlow 2.21.0
- librosa (audio processing)
- FastAPI + Uvicorn (web server)

---

## ⚙️ Files Changed/Created

### Modified Files
1. **`backend/main.py`** - Complete rewrite
   - Real model loading pipeline
   - Proper async startup handler
   - Real prediction logic
   - Modern lifespan event handling (no deprecation warnings)

2. **`backend/feature_utils.py`** - Enhanced
   - Librosa-based feature extraction
   - Proper audio processing
   - 47-feature vector generation
   - Fallback for error cases

3. **`backend/requirements.txt`**
   - Added `soundfile` package

### New Files
1. **`backend/test_voice.py`** - Validation script
2. **`FIXES_SUMMARY.md`** - Detailed technical summary

---

## 🧪 Testing

Run the included test script:
```bash
cd backend
python test_voice.py
```

Expected output:
```
=== TEST 1: Model File Check ===
✓ Model found at: models/voice_dnn_1.keras (2.5 MB)

=== TEST 2: Model Loading ===
✓ Model loaded successfully

=== TEST 3: Feature Extraction (Default) ===
✓ Generated default features (1, 47)

=== TEST 4: Model Prediction ===
✓ Model prediction successful

=== ALL TESTS PASSED ===
```

---

## 🎓 How to Improve Predictions

1. **Retrain with real data** - Use actual Parkinson's vs healthy voice samples
2. **Feature scaling** - Normalize features using StandardScaler 
3. **Data augmentation** - Vary pitch, speed, noise for robustness
4. **Ensemble models** - Combine multiple models for better accuracy
5. **Hyperparameter tuning** - Optimize learning rate, dropout, etc.
6. **Cross-validation** - Validate on multiple folds
7. **Add gait detection** - Models exist: `gait_dnn.keras`, `gait_heavy_dnn.keras`
8. **Fusion model** - Use voice + gait: `fusion_dnn_improved.keras`

---

## 🐛 Troubleshooting

### Server won't start
```bash
# Check if port 8001 is in use
netstat -ano | findstr :8001

# If port is in use, kill the process or use different port
# Edit main.py: uvicorn.run(app, host="0.0.0.0", port=8002)
```

### Model loading fails
- Check if `models/voice_dnn_1.keras` exists
- Verify TensorFlow is installed: `pip install tensorflow`
- Check file integrity: `ls -lh models/voice_dnn_1.keras`

### Feature extraction errors
- Install librosa: `pip install librosa soundfile`
- Check audio file format (WAV, MP3 supported)
- Verify audio file is not corrupted

### CORS errors (frontend can't connect)
- Already configured with `allow_origins=["*"]`
- Ensure frontend uses correct API URL: `http://localhost:8001/predict`

---

## 📝 API Documentation

Generate interactive API docs:
```
Open: http://localhost:8001/docs
```

This shows all endpoints, parameters, and response schemas with try-it-out functionality.

---

## ✨ Next Steps

1. **Test with voice samples**
   - Collect healthy volunteer samples
   - Collect Parkinson's patient samples
   - Verify predictions are reasonable

2. **Retrain the model**
   - Use actual labeled data
   - Achieve >90% accuracy if possible
   - Validate on independent test set

3. **Add frontend**
   - Upload voice files
   - Display results with confidence
   - Show extracted features

4. **Deploy to production**
   - Use Gunicorn for multiple workers
   - Add authentication if needed
   - Enable HTTPS/SSL
   - Add rate limiting

---

## 📞 Support

For issues, check:
1. `FIXES_SUMMARY.md` - Technical details of fixes
2. `backend/test_voice.py` - Validation tests
3. Error logs in terminal output
