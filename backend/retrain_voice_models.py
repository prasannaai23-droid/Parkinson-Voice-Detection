"""
RETRAIN VOICE MODELS - Fix Inverted Predictions
================================================
Extracts 215 features from HC_AH (healthy) and PD_AH (Parkinson's) voice data,
then trains Neural Network, Random Forest, and XGBoost ensemble models.

Labels:
  0 = Healthy (HC_AH)
  1 = Parkinson's Disease (PD_AH)

Usage:
  python retrain_voice_models.py
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

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from advanced_features import extract_advanced_features

# Paths
PROJECT_DIR = BACKEND_DIR.parent
HC_DIR = PROJECT_DIR / "HC_AH"    # Healthy Controls
PD_DIR = PROJECT_DIR / "PD_AH"    # Parkinson's Disease
MODELS_DIR = BACKEND_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def extract_all_features():
    """Extract 215 features from all voice files."""
    all_features = []
    all_labels = []
    all_files = []

    # Process healthy controls (label=0)
    hc_files = sorted(HC_DIR.glob("*.wav"))
    print(f"\nProcessing {len(hc_files)} healthy control files from HC_AH/")
    for i, f in enumerate(hc_files, 1):
        print(f"  [{i}/{len(hc_files)}] {f.name}")
        df = extract_advanced_features(str(f))
        if df is not None and df.shape[0] > 0 and df.shape[1] == 215:
            all_features.append(df.values[0])
            all_labels.append(0)
            all_files.append(f.name)
        else:
            print(f"    SKIPPED - extraction failed or wrong shape")

    # Process PD patients (label=1)
    pd_files = sorted(PD_DIR.glob("*.wav"))
    print(f"\nProcessing {len(pd_files)} Parkinson's disease files from PD_AH/")
    for i, f in enumerate(pd_files, 1):
        print(f"  [{i}/{len(pd_files)}] {f.name}")
        df = extract_advanced_features(str(f))
        if df is not None and df.shape[0] > 0 and df.shape[1] == 215:
            all_features.append(df.values[0])
            all_labels.append(1)
            all_files.append(f.name)
        else:
            print(f"    SKIPPED - extraction failed or wrong shape")

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    print(f"\nTotal samples: {len(X)}")
    print(f"  Healthy: {np.sum(y == 0)}")
    print(f"  PD: {np.sum(y == 1)}")
    print(f"  Features: {X.shape[1]}")

    return X, y, all_files


def build_nn_model(n_features):
    """Build optimized neural network for voice classification."""
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
    """Train all three ensemble models with cross-validation."""

    # Replace NaN/Inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Fit and save scaler
    print("\n=== Fitting RobustScaler ===")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    scaler_path = MODELS_DIR / "voice_scaler_v2.pkl"
    joblib.dump(scaler, str(scaler_path))
    print(f"  Saved scaler to {scaler_path.name}")

    # Cross-validation setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ==========================================
    # Train Neural Network
    # ==========================================
    print("\n=== Training Neural Network ===")
    nn_cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_nn_model(X_scaled.shape[1])
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
        nn_cv_scores.append(fold_acc)
        print(f"  Fold {fold}: accuracy={fold_acc:.4f}")

    print(f"  CV Mean Accuracy: {np.mean(nn_cv_scores):.4f} +/- {np.std(nn_cv_scores):.4f}")

    # Train final NN on all data
    final_nn = build_nn_model(X_scaled.shape[1])
    final_nn.fit(
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
    nn_path = MODELS_DIR / "voice_nn_v2.keras"
    final_nn.save(str(nn_path))
    print(f"  Saved NN to {nn_path.name}")

    # ==========================================
    # Train Random Forest
    # ==========================================
    print("\n=== Training Random Forest ===")
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
    rf_path = MODELS_DIR / "voice_rf_v2.pkl"
    joblib.dump(rf, str(rf_path))
    print(f"  Saved RF to {rf_path.name}")

    # ==========================================
    # Train XGBoost
    # ==========================================
    print("\n=== Training XGBoost ===")
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=np.sum(y == 0) / max(np.sum(y == 1), 1),
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
        )

        xgb_cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            xgb.fit(X_train, y_train)
            fold_acc = xgb.score(X_val, y_val)
            xgb_cv_scores.append(fold_acc)
            print(f"  Fold {fold}: accuracy={fold_acc:.4f}")

        print(f"  CV Mean Accuracy: {np.mean(xgb_cv_scores):.4f} +/- {np.std(xgb_cv_scores):.4f}")

        # Train final XGB on all data
        xgb.fit(X_scaled, y)
        xgb_path = MODELS_DIR / "voice_xgb_v2.pkl"
        joblib.dump(xgb, str(xgb_path))
        print(f"  Saved XGB to {xgb_path.name}")
    except ImportError:
        print("  XGBoost not installed, skipping")
        xgb = None
        xgb_cv_scores = []

    # ==========================================
    # Final Validation
    # ==========================================
    print("\n=== Final Validation on Full Dataset ===")
    nn_preds_prob = final_nn.predict(X_scaled, verbose=0).flatten()
    nn_preds = (nn_preds_prob > 0.5).astype(int)

    rf_preds_prob = rf.predict_proba(X_scaled)[:, 1]
    rf_preds = rf.predict(X_scaled)

    if xgb is not None:
        xgb_preds_prob = xgb.predict_proba(X_scaled)[:, 1]
        xgb_preds = xgb.predict(X_scaled)
        ensemble_prob = 0.4 * nn_preds_prob + 0.3 * rf_preds_prob + 0.3 * xgb_preds_prob
    else:
        ensemble_prob = 0.5 * nn_preds_prob + 0.5 * rf_preds_prob

    ensemble_preds = (ensemble_prob > 0.5).astype(int)

    print("\nNeural Network:")
    print(f"  Accuracy: {accuracy_score(y, nn_preds):.4f}")
    print(f"  AUC-ROC: {roc_auc_score(y, nn_preds_prob):.4f}")

    print("\nRandom Forest:")
    print(f"  Accuracy: {accuracy_score(y, rf_preds):.4f}")
    print(f"  AUC-ROC: {roc_auc_score(y, rf_preds_prob):.4f}")

    if xgb is not None:
        print("\nXGBoost:")
        print(f"  Accuracy: {accuracy_score(y, xgb_preds):.4f}")
        print(f"  AUC-ROC: {roc_auc_score(y, xgb_preds_prob):.4f}")

    print("\nEnsemble:")
    print(f"  Accuracy: {accuracy_score(y, ensemble_preds):.4f}")
    print(f"  AUC-ROC: {roc_auc_score(y, ensemble_prob):.4f}")
    print(f"\nClassification Report (Ensemble):")
    print(classification_report(y, ensemble_preds, target_names=['Healthy', 'Parkinsons']))
    print(f"Confusion Matrix:\n{confusion_matrix(y, ensemble_preds)}")

    # Save metadata
    metadata = {
        "model_version": "2.2-retrained",
        "creation_date": datetime.now().isoformat(),
        "feature_count": int(X_scaled.shape[1]),
        "feature_extractor": "advanced_features.extract_advanced_features (215 features)",
        "training_samples": int(len(y)),
        "healthy_samples": int(np.sum(y == 0)),
        "pd_samples": int(np.sum(y == 1)),
        "label_mapping": {"0": "Healthy (HC_AH)", "1": "Parkinsons (PD_AH)"},
        "scaler_type": "RobustScaler",
        "nn_cv_accuracy": float(np.mean(nn_cv_scores)),
        "rf_cv_accuracy": float(np.mean(rf_cv_scores)),
        "xgb_cv_accuracy": float(np.mean(xgb_cv_scores)) if xgb_cv_scores else None,
        "ensemble_accuracy": float(accuracy_score(y, ensemble_preds)),
        "ensemble_auc": float(roc_auc_score(y, ensemble_prob)),
        "models": {
            "nn": "voice_nn_v2.keras",
            "rf": "voice_rf_v2.pkl",
            "xgb": "voice_xgb_v2.pkl" if xgb is not None else None,
            "scaler": "voice_scaler_v2.pkl",
        },
        "ensemble_weights": {"nn": 0.4, "rf": 0.3, "xgb": 0.3},
    }

    meta_path = MODELS_DIR / "voice_model_v2_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {meta_path.name}")

    # Save features CSV for reference
    feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
    features_df = pd.DataFrame(X, columns=feature_cols)
    features_df['label'] = y
    features_path = MODELS_DIR / "voice_features_v2.csv"
    features_df.to_csv(str(features_path), index=False)
    print(f"Features saved to {features_path.name}")

    # Copy models to filenames the backend expects
    import shutil as _shutil
    _shutil.copy2(str(MODELS_DIR / 'voice_scaler_v2.pkl'), str(MODELS_DIR / 'unified_scaler_fixed.pkl'))
    _shutil.copy2(str(MODELS_DIR / 'voice_scaler_v2.pkl'), str(MODELS_DIR / 'unified_scaler.pkl'))
    _shutil.copy2(str(MODELS_DIR / 'voice_rf_v2.pkl'), str(MODELS_DIR / 'voice_rf.pkl'))
    if xgb is not None:
        _shutil.copy2(str(MODELS_DIR / 'voice_xgb_v2.pkl'), str(MODELS_DIR / 'ensemble_xgb.pkl'))
    final_nn.save(str(MODELS_DIR / 'unified_advanced_pd_fixed.keras'))
    final_nn.save(str(MODELS_DIR / 'ensemble_nn.keras'))
    print('  Copied models to backend-expected filenames')

    return metadata


if __name__ == "__main__":
    print("=" * 60)
    print("VOICE MODEL RETRAINING")
    print("=" * 60)

    # Verify data directories
    if not HC_DIR.exists():
        print(f"ERROR: HC_AH directory not found at {HC_DIR}")
        sys.exit(1)
    if not PD_DIR.exists():
        print(f"ERROR: PD_AH directory not found at {PD_DIR}")
        sys.exit(1)

    hc_count = len(list(HC_DIR.glob("*.wav")))
    pd_count = len(list(PD_DIR.glob("*.wav")))
    print(f"\nData: {hc_count} healthy + {pd_count} PD voice files")

    # Extract features
    X, y, files = extract_all_features()

    if len(X) < 10:
        print("ERROR: Too few samples extracted. Check audio files.")
        sys.exit(1)

    # Train models
    metadata = train_models(X, y)

    print("\n" + "=" * 60)
    print("RETRAINING COMPLETE")
    print("=" * 60)
    print(f"\nNew models saved to: {MODELS_DIR}")
    print("Files created:")
    print("  - voice_nn_v2.keras (Neural Network)")
    print("  - voice_rf_v2.pkl (Random Forest)")
    print("  - voice_xgb_v2.pkl (XGBoost)")
    print("  - voice_scaler_v2.pkl (RobustScaler)")
    print("  - voice_model_v2_metadata.json")
    print("  - voice_features_v2.csv")
