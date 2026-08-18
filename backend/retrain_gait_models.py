"""
RETRAIN GAIT MODELS
===================
Extracts 93 features from gait sensor data files and trains
Random Forest + DNN models for Parkinson's disease detection.

Data: PhysioNet Gait in Parkinson's Disease (VGRF recordings)
  - GaCo*.txt = healthy controls (label=0)
  - GaPt*.txt = PD patients (label=1)
  - Also supports Ju* and Si* prefixes

Usage:
  python retrain_gait_models.py
"""

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import json
from datetime import datetime

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
GAIT_HEALTHY_DIR = PROJECT_DIR / "data" / "gait" / "healthy"
GAIT_DIAGNOSED_DIR = PROJECT_DIR / "data" / "gait" / "diagnosed"
MODELS_DIR = BACKEND_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MIN_ROWS = 100  # Skip files with fewer rows


def extract_gait_features(filepath):
    """
    Extract 93 features from a gait sensor data file.

    Input format: tab-separated, 19 columns
    - Col 0: Time (seconds)
    - Cols 1-8: Left foot sensors (L1-L8)
    - Cols 9-16: Right foot sensors (R1-R8)
    - Cols 17-18: Total force (left, right)

    Features (93 total):
    - 18 signals x 5 stats (mean, std, max, min, range) = 90
    - stride_time_mean, stride_time_std, left_right_asymmetry = 3
    """
    try:
        try:
            df = pd.read_csv(filepath, sep='\t', header=None, engine='python')
        except Exception:
            df = pd.read_csv(filepath, header=None, engine='python')

        if df.shape[0] < MIN_ROWS:
            return None

        # Skip time column, use sensor data
        if df.shape[1] >= 19:
            data = df.iloc[:, 1:19].values.astype(np.float64)  # 18 columns
        elif df.shape[1] > 1:
            data = df.iloc[:, 1:].values.astype(np.float64)
        else:
            return None

        n_cols = data.shape[1]
        features = []

        # 5 stats per column: mean, std, max, min, range
        for col in range(min(n_cols, 18)):
            col_data = data[:, col]
            col_data = col_data[np.isfinite(col_data)]
            if len(col_data) == 0:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            features.append(float(np.mean(col_data)))
            features.append(float(np.std(col_data)))
            features.append(float(np.max(col_data)))
            features.append(float(np.min(col_data)))
            features.append(float(np.ptp(col_data)))

        # Pad to 90 features if fewer than 18 sensor columns
        while len(features) < 90:
            features.append(0.0)

        # Stride time from total left force (column index 16 in data, original col 17)
        try:
            from scipy import signal as sig
            if n_cols >= 17:
                total_left = data[:, 16]
                total_right = data[:, 17] if n_cols >= 18 else data[:, -1]
            else:
                total_left = data[:, -2] if n_cols >= 2 else data[:, 0]
                total_right = data[:, -1]

            total_left_clean = total_left[np.isfinite(total_left)]
            if len(total_left_clean) > 20:
                threshold = np.mean(total_left_clean) + 0.5 * np.std(total_left_clean)
                peaks, _ = sig.find_peaks(total_left_clean, height=threshold, distance=50)
                if len(peaks) > 1:
                    stride_times = np.diff(peaks) * 0.01  # 100 Hz
                    features.append(float(np.mean(stride_times)))
                    features.append(float(np.std(stride_times)))
                else:
                    features.extend([1.0, 0.1])
            else:
                features.extend([1.0, 0.1])

            # Left/right asymmetry
            mean_left = np.mean(total_left[np.isfinite(total_left)])
            mean_right = np.mean(total_right[np.isfinite(total_right)])
            asymmetry = mean_left / (mean_right + 1e-8)
            features.append(float(asymmetry))
        except Exception:
            features.extend([1.0, 0.1, 1.0])

        # Ensure exactly 93
        while len(features) < 93:
            features.append(0.0)
        features = features[:93]

        return np.array(features, dtype=np.float32)

    except Exception as e:
        print(f"    Error processing {filepath}: {e}")
        return None


def extract_all_features():
    """Extract features from all gait data files."""
    all_features = []
    all_labels = []
    all_files = []

    # Process healthy controls
    healthy_files = sorted(GAIT_HEALTHY_DIR.glob("*.txt"))
    print(f"\nProcessing {len(healthy_files)} healthy control files...")
    for i, f in enumerate(healthy_files, 1):
        print(f"  [{i}/{len(healthy_files)}] {f.name}", end="")
        feat = extract_gait_features(str(f))
        if feat is not None:
            all_features.append(feat)
            all_labels.append(0)
            all_files.append(f.name)
            print(" -> OK")
        else:
            print(" -> SKIPPED (too small)")

    # Process PD patients
    diagnosed_files = sorted(GAIT_DIAGNOSED_DIR.glob("*.txt"))
    print(f"\nProcessing {len(diagnosed_files)} PD patient files...")
    for i, f in enumerate(diagnosed_files, 1):
        print(f"  [{i}/{len(diagnosed_files)}] {f.name}", end="")
        feat = extract_gait_features(str(f))
        if feat is not None:
            all_features.append(feat)
            all_labels.append(1)
            all_files.append(f.name)
            print(" -> OK")
        else:
            print(" -> SKIPPED (too small)")

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    print(f"\nTotal samples: {len(X)}")
    print(f"  Healthy: {np.sum(y == 0)}")
    print(f"  PD: {np.sum(y == 1)}")
    print(f"  Features: {X.shape[1]}")

    return X, y, all_files


def build_gait_dnn(n_features):
    """Build DNN for gait classification."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_models(X, y):
    """Train all gait models."""

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Fit scaler
    print("\n=== Fitting RobustScaler ===")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    scaler_path = MODELS_DIR / "gait_scaler.pkl"
    joblib.dump(scaler, str(scaler_path))
    print(f"  Saved scaler to {scaler_path.name}")

    # Cross-validation
    n_splits = min(5, min(np.sum(y == 0), np.sum(y == 1)))
    if n_splits < 2:
        n_splits = 2
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # ==========================================
    # Train Random Forest
    # ==========================================
    print(f"\n=== Training Random Forest ({n_splits}-fold CV) ===")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    rf_cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        rf.fit(X_train, y_train)
        fold_acc = rf.score(X_val, y_val)
        rf_cv_scores.append(fold_acc)
        print(f"  Fold {fold}: accuracy={fold_acc:.4f}")

    print(f"  CV Mean Accuracy: {np.mean(rf_cv_scores):.4f} +/- {np.std(rf_cv_scores):.4f}")

    # Train final RF on all data
    rf.fit(X_scaled, y)
    rf_path = MODELS_DIR / "gait_rf.pkl"
    joblib.dump(rf, str(rf_path))
    print(f"  Saved RF to {rf_path.name}")

    # ==========================================
    # Train DNN
    # ==========================================
    print(f"\n=== Training DNN ({n_splits}-fold CV) ===")
    dnn_cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_gait_dnn(X_scaled.shape[1])
        model.fit(
            X_train, y_train,
            epochs=150,
            batch_size=16,
            validation_data=(X_val, y_val),
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=20, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
                )
            ]
        )
        val_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
        fold_acc = accuracy_score(y_val, val_pred)
        dnn_cv_scores.append(fold_acc)
        print(f"  Fold {fold}: accuracy={fold_acc:.4f}")

    print(f"  CV Mean Accuracy: {np.mean(dnn_cv_scores):.4f} +/- {np.std(dnn_cv_scores):.4f}")

    # Train final DNN on all data
    final_dnn = build_gait_dnn(X_scaled.shape[1])
    final_dnn.fit(
        X_scaled, y,
        epochs=200,
        batch_size=16,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=30, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss', factor=0.5, patience=15, min_lr=1e-6
            )
        ]
    )
    dnn_path = MODELS_DIR / "gait_heavy_dnn.keras"
    final_dnn.save(str(dnn_path))
    print(f"  Saved DNN to {dnn_path.name}")

    # ==========================================
    # Final Validation
    # ==========================================
    print("\n=== Final Validation ===")
    rf_preds = rf.predict(X_scaled)
    rf_probs = rf.predict_proba(X_scaled)[:, 1]
    dnn_probs = final_dnn.predict(X_scaled, verbose=0).flatten()
    dnn_preds = (dnn_probs > 0.5).astype(int)

    ensemble_prob = 0.5 * rf_probs + 0.5 * dnn_probs
    ensemble_preds = (ensemble_prob > 0.5).astype(int)

    print("\nRandom Forest:")
    print(f"  Accuracy: {accuracy_score(y, rf_preds):.4f}")
    print(f"  AUC-ROC: {roc_auc_score(y, rf_probs):.4f}")

    print("\nDNN:")
    print(f"  Accuracy: {accuracy_score(y, dnn_preds):.4f}")
    print(f"  AUC-ROC: {roc_auc_score(y, dnn_probs):.4f}")

    print("\nEnsemble (RF+DNN):")
    print(f"  Accuracy: {accuracy_score(y, ensemble_preds):.4f}")
    print(f"  AUC-ROC: {roc_auc_score(y, ensemble_prob):.4f}")
    print(f"\nClassification Report (Ensemble):")
    print(classification_report(y, ensemble_preds, target_names=['Healthy', 'Parkinsons']))
    print(f"Confusion Matrix:\n{confusion_matrix(y, ensemble_preds)}")

    # Save metadata
    metadata = {
        "model_version": "gait-v2-retrained",
        "creation_date": datetime.now().isoformat(),
        "feature_count": int(X_scaled.shape[1]),
        "feature_extractor": "93-feature gait sensor extraction",
        "training_samples": int(len(y)),
        "healthy_samples": int(np.sum(y == 0)),
        "pd_samples": int(np.sum(y == 1)),
        "label_mapping": {"0": "Healthy", "1": "Parkinsons"},
        "rf_cv_accuracy": float(np.mean(rf_cv_scores)),
        "dnn_cv_accuracy": float(np.mean(dnn_cv_scores)),
        "ensemble_accuracy": float(accuracy_score(y, ensemble_preds)),
        "ensemble_auc": float(roc_auc_score(y, ensemble_prob)),
    }

    meta_path = MODELS_DIR / "gait_model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {meta_path.name}")

    return metadata


if __name__ == "__main__":
    print("=" * 60)
    print("GAIT MODEL RETRAINING")
    print("=" * 60)

    if not GAIT_HEALTHY_DIR.exists():
        print(f"ERROR: Healthy gait directory not found: {GAIT_HEALTHY_DIR}")
        sys.exit(1)
    if not GAIT_DIAGNOSED_DIR.exists():
        print(f"ERROR: Diagnosed gait directory not found: {GAIT_DIAGNOSED_DIR}")
        sys.exit(1)

    h_count = len(list(GAIT_HEALTHY_DIR.glob("*.txt")))
    d_count = len(list(GAIT_DIAGNOSED_DIR.glob("*.txt")))
    print(f"\nData: {h_count} healthy + {d_count} PD gait files")

    X, y, files = extract_all_features()

    if len(X) < 5:
        print("ERROR: Too few valid samples. Run download_gait_data.py first.")
        sys.exit(1)

    metadata = train_models(X, y)

    print("\n" + "=" * 60)
    print("GAIT RETRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModels saved to: {MODELS_DIR}")
    print("Files created/updated:")
    print("  - gait_rf.pkl (Random Forest)")
    print("  - gait_heavy_dnn.keras (Deep Neural Network)")
    print("  - gait_scaler.pkl (RobustScaler)")
    print("  - gait_model_metadata.json")
