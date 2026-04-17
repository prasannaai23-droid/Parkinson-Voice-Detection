# Parkinson's Voice Detection - Model Improvement Guide

## Current Status
✓ Model is loading and making predictions
✗ Predictions may not be accurate yet

## Why Predictions May Be Poor

1. **Model trained on limited data** - Original training data may be small or unrepresentative
2. **Feature scaling missing** - Features aren't normalized
3. **Class imbalance** - More healthy samples than Parkinson's (or vice versa)
4. **Feature quality** - Currently using default values in fallback scenarios
5. **Model architecture** - May not be optimal for this task

---

## 🔄 Quick Improvement Strategy (Priority Order)

### Priority 1: Feature Normalization (Immediate - 5 mins)
Add StandardScaler to feature extraction:

```python
# In feature_utils.py, after get_default_voice_features():

from sklearn.preprocessing import StandardScaler

# Create a scaler (could load from saved scaler)
scaler = StandardScaler()

# Scale features before prediction (in main.py):
features_array = scaler.fit_transform(features_df.values.astype(np.float32))
# OR better: load pre-fitted scaler from models/voice_scaler.pkl (if exists)
```

Check if scaler exists:
```bash
ls backend/models/voice_scaler.pkl
```

### Priority 2: Data Collection (1-2 weeks)
Collect real voice samples:
- **Healthy controls**: 50-100 samples
- **Parkinson's patients**: 50-100 samples
- **Formats**: WAV, 16-bit, 44.1kHz minimum
- **Recording**: Clean audio, no background noise

### Priority 3: Model Retraining (2-3 days)
```python
# Sample training pipeline (pseudocode)

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from feature_utils import extract_voice_features

# 1. Load all audio files and extract features
X_features = []
y_labels = []

for audio_file in healthy_samples:
    features = extract_voice_features(audio_file)
    X_features.append(features.values[0])
    y_labels.append(0)  # 0 = healthy

for audio_file in parkinsons_samples:
    features = extract_voice_features(audio_file)
    X_features.append(features.values[0])
    y_labels.append(1)  # 1 = Parkinson's

X = np.array(X_features)
y = np.array(y_labels)

# 2. Normalize features  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for later use!
import joblib
joblib.dump(scaler, 'models/voice_scaler.pkl')

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 4. Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(47,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 5. Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# 6. Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)

# 7. Evaluate
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.3f}")
print(f"Test AUC: {test_auc:.3f}")

# 8. Save model
model.save('models/voice_dnn_retrained.keras')
```

---

## 📊 Validation Metrics to Track

| Metric | Target | How to Calculate |
|--------|--------|-----------------|
| **Accuracy** | > 85% | (TP + TN) / Total |
| **Sensitivity** | > 80% | TP / (TP + FN) - catches Parkinson's |
| **Specificity** | > 80% | TN / (TN + FP) - avoids false alarms |
| **AUC** | > 0.85 | Area under ROC curve |
| **Precision** | > 85% | TP / (TP + FP) - prediction reliability |
| **F1-Score** | > 0.80 | Harmonic mean of precision & recall |

---

## 🔍 Debugging: Check Model Performance

### Current Model Analysis
```python
# Test current model with sample features
from feature_utils import get_default_voice_features
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('models/voice_dnn_1.keras')
features = get_default_voice_features().values.astype(np.float32)

prediction = model.predict(features, verbose=0)
print(f"Raw prediction: {prediction[0][0]:.4f}")
print(f"Probability: {prediction[0][0] * 100:.1f}%")

# Check prediction distribution on test set
predictions = model.predict(X_test, verbose=0)
print(f"Min pred: {predictions.min():.3f}")
print(f"Max pred: {predictions.max():.3f}")
print(f"Mean pred: {predictions.mean():.3f}")
```

### Feature Analysis
```python
# Check if features are on reasonable scale
import pandas as pd

features_sample = extract_voice_features('test_audio.wav')
print(features_sample.describe())

# Should see:
# - No NaN values
# - Reasonable ranges (not all zeros)
# - Mix of small and large values
```

---

## ⚙️ Advanced Improvements

### 1. Data Augmentation
```python
import librosa

def augment_audio(audio_path):
    """Apply augmentations to increase training data"""
    y, sr = librosa.load(audio_path, sr=None)
    
    augmented = []
    
    # Original
    augmented.append(y)
    
    # Pitch shift (±2-3 semitones)
    augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2))
    augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2))
    
    # Time stretch (±10%)
    augmented.append(librosa.effects.time_stretch(y, rate=1.1))
    augmented.append(librosa.effects.time_stretch(y, rate=0.9))
    
    # Add slight noise
    augmented.append(y + np.random.normal(0, 0.001, len(y)))
    
    return augmented
```

### 2. Cross-Validation
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
    X_train = X_scaled[train_idx]
    y_train = y[train_idx]
    X_val = X_scaled[val_idx]
    y_val = y[val_idx]
    
    # Train model on fold...
    # Evaluate on validation set
    # scores.append(val_acc)

print(f"Cross-validation accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

### 3. Ensemble Methods
```python
# Train multiple models and average predictions
models = []
for i in range(3):
    model = build_model()
    model.fit(X_train, y_train, epochs=50)
    models.append(model)

# Predict with ensemble
def ensemble_predict(features):
    predictions = []
    for model in models:
        pred = model.predict(features, verbose=0)
        predictions.append(pred[0][0])
    return np.mean(predictions)
```

### 4. Class Imbalance Handling
```python
# If unbalanced data
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)

# Use in training
model.fit(
    X_train, y_train, 
    class_weight=dict(enumerate(class_weights))
)
```

---

## 📈 Expected Improvements Timeline

| Step | Timeline | Accuracy Improvement |
|------|----------|---------------------|
| Feature normalization | 5 min | +5-10% |
| Collect 50 real samples each | 1 week | +15-25% |
| Retrain with real data | 2 days | +30-40% |
| Add data augmentation | 1 day | +5-10% |
| Fine-tune hyperparameters | 2 days | +3-5% |
| **Estimated Total** | **2 weeks** | **60-80% improvement** |

---

## ✅ Validation Checklist

Before considering model ready:

- [ ] Tested on at least 20 unknown audio samples
- [ ] Accuracy > 85% on test set
- [ ] Sensitivity > 80% (catches Parkinson's cases)
- [ ] Specificity > 80% (avoids false alarms)
- [ ] AUC > 0.85
- [ ] Features properly normalized (StandardScaler)
- [ ] Cross-validation performed (5-fold minimum)
- [ ] Results documented with confusion matrix
- [ ] Edge cases tested (background noise, emotion, age, gender variation)
- [ ] Predictions are reproducible

---

## 🎯 Next Immediate Actions

1. **Today**: Collect some test voice samples
2. **This week**: Run model against real samples to see current accuracy
3. **Next week**: Collect more training data (100+ samples healthy, 100+ Parkinson's)
4. **In 2 weeks**: Retrain and validate new model

The framework is ready - now it needs real data to learn from!
