"""
VOICE MODEL TRAINING v3
=======================
Trains the Parkinson's voice classifier from labelled recordings and writes the
artifacts the backend loads (advanced_scaler.pkl / advanced_rf.pkl /
advanced_xgb.pkl / advanced_metadata.json).

Unlike the previous script, every reported number comes from cross-validation
with the scaler and feature selection fitted inside each fold, so the metrics
reflect unseen data instead of the training set.

Usage:
  python train_voice_models_v3.py --data-dir ../voice
  python train_voice_models_v3.py --features models/voice_features_v2.csv

The data directory must contain two sub-folders of .wav files:
  healthy/ (or HC_AH/)  -> label 0
  pd/      (or PD_AH/)  -> label 1
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models"
sys.path.insert(0, str(BACKEND_DIR))

HEALTHY_DIRS = ("healthy", "HC_AH", "hc")
PD_DIRS = ("pd", "PD_AH", "parkinsons")

N_SPLITS = 5
N_REPEATS = 1
RANDOM_STATE = 42


def find_class_dir(root: Path, names) -> Path:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise SystemExit(f"No {names[0]}/ folder under {root}")


def extract_dataset(data_dir: Path, cache: Path):
    """Extract 215 acoustic features per recording, caching the result."""
    from advanced_features import extract_advanced_features

    rows, labels, files = [], [], []
    for label, names in ((0, HEALTHY_DIRS), (1, PD_DIRS)):
        class_dir = find_class_dir(data_dir, names)
        wavs = sorted(class_dir.glob("*.wav"))
        print(f"\n{class_dir.name}: {len(wavs)} files (label={label})")
        for i, wav in enumerate(wavs, 1):
            try:
                df = extract_advanced_features(str(wav))
            except Exception as e:
                print(f"  [{i}/{len(wavs)}] SKIPPED {wav.name}: {e}")
                continue
            if df is None or df.shape[0] == 0 or df.shape[1] != 215:
                print(f"  [{i}/{len(wavs)}] SKIPPED {wav.name}")
                continue
            rows.append(df.values[0])
            labels.append(label)
            files.append(wav.name)
            if i % 10 == 0:
                print(f"  [{i}/{len(wavs)}] extracted")

    X = np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(labels, dtype=np.int32)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cached = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    cached["label"] = y
    cached["file"] = files
    cached.to_csv(cache, index=False)
    print(f"\nCached features to {cache}")
    return X, y


def load_features_csv(path: Path):
    df = pd.read_csv(path)
    y = df["label"].to_numpy(dtype=np.int32)
    X = df.drop(columns=[c for c in ("label", "file") if c in df.columns]).to_numpy(dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), y


def candidate_models(n_pos, n_neg):
    """Classifiers worth trying on ~80 samples with 215 features."""
    scale_pos_weight = n_neg / max(n_pos, 1)
    models = {
        "logistic": LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000),
        "random_forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=100, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.6, reg_lambda=2.0, scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
    except ImportError:
        print("xgboost not installed - skipping that candidate")
    return models


def build_pipeline(clf, k_features):
    """Scaling and feature selection live inside the pipeline to avoid leakage."""
    return Pipeline([
        ("scaler", RobustScaler()),
        ("select", SelectKBest(f_classif, k=k_features)),
        ("clf", clf),
    ])


def evaluate(pipeline, X, y):
    """Out-of-fold probabilities averaged over repeated stratified CV."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                 random_state=RANDOM_STATE)
    aucs, accs, sens, specs = [], [], [], []
    oof_sum = np.zeros(len(y))

    for train_idx, test_idx in cv.split(X, y):
        pipeline.fit(X[train_idx], y[train_idx])
        prob = pipeline.predict_proba(X[test_idx])[:, 1]
        oof_sum[test_idx] += prob
        pred = (prob >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[test_idx], prob))
        accs.append(accuracy_score(y[test_idx], pred))
        tn, fp, fn, tp = confusion_matrix(y[test_idx], pred, labels=[0, 1]).ravel()
        sens.append(tp / max(tp + fn, 1))
        specs.append(tn / max(tn + fp, 1))

    return {
        "auc": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "sensitivity": float(np.mean(sens)),
        "specificity": float(np.mean(specs)),
        "oof_prob": oof_sum / N_REPEATS,
    }


def best_threshold(y, prob):
    """Threshold maximising Youden's J on out-of-fold probabilities."""
    best_j, best_t = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        j = tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t, best_j


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="folder containing healthy/ and pd/ wav files")
    parser.add_argument("--features", type=Path, default=None,
                        help="pre-extracted features CSV (skips audio processing)")
    parser.add_argument("--k-features", type=int, default=40,
                        help="number of features kept by SelectKBest")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cache = MODELS_DIR / "voice_features_v3.csv"

    if args.features:
        X, y = load_features_csv(args.features)
        print(f"Loaded {X.shape[0]} samples x {X.shape[1]} features from {args.features}")
    elif args.data_dir:
        X, y = extract_dataset(args.data_dir, cache)
    elif cache.exists():
        X, y = load_features_csv(cache)
        print(f"Loaded cached features from {cache}")
    else:
        raise SystemExit("Pass --data-dir or --features")

    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    print(f"\nDataset: {len(y)} samples ({n_neg} healthy, {n_pos} Parkinson's), "
          f"{X.shape[1]} features")
    print(f"Validation: {N_SPLITS}-fold stratified CV repeated {N_REPEATS}x\n")

    results = {}
    for name, clf in candidate_models(n_pos, n_neg).items():
        pipeline = build_pipeline(clf, args.k_features)
        scores = evaluate(pipeline, X, y)
        results[name] = scores
        print(f"{name:<15} AUC {scores['auc']:.3f}+/-{scores['auc_std']:.3f}  "
              f"acc {scores['accuracy']:.3f}  sens {scores['sensitivity']:.3f}  "
              f"spec {scores['specificity']:.3f}")

    ranked = sorted(results, key=lambda n: results[n]["auc"], reverse=True)
    ensemble_members = ranked[:3]
    ensemble_prob = np.mean([results[n]["oof_prob"] for n in ensemble_members], axis=0)
    ensemble_auc = float(roc_auc_score(y, ensemble_prob))
    ensemble_acc = float(accuracy_score(y, (ensemble_prob >= 0.5).astype(int)))
    print(f"\nensemble({', '.join(ensemble_members)}) "
          f"out-of-fold AUC {ensemble_auc:.3f}  acc {ensemble_acc:.3f}")

    threshold, youden = best_threshold(y, ensemble_prob)
    print(f"Best decision threshold {threshold:.2f} (Youden J {youden:.3f})")

    # Refit the chosen members on all data. The scaler is saved separately
    # because the backend scales features before calling the models.
    scaler = RobustScaler().fit(X)
    X_scaled = scaler.transform(X)
    inner_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    saved = {}
    all_models = candidate_models(n_pos, n_neg)
    for name in ensemble_members:
        model = Pipeline([
            ("select", SelectKBest(f_classif, k=args.k_features)),
            ("clf", CalibratedClassifierCV(all_models[name], cv=inner_cv, method="sigmoid")),
        ])
        model.fit(X_scaled, y)
        filename = {"random_forest": "advanced_rf.pkl", "xgboost": "advanced_xgb.pkl"}.get(
            name, f"advanced_{name}.pkl")
        joblib.dump(model, MODELS_DIR / filename)
        saved[name] = filename
        print(f"Saved {name} -> {filename}")

    joblib.dump(scaler, MODELS_DIR / "advanced_scaler.pkl")
    print("Saved scaler -> advanced_scaler.pkl")

    metadata = {
        "model_version": "3.0",
        "creation_date": datetime.now().isoformat(),
        "feature_count": int(X.shape[1]),
        "features_selected": args.k_features,
        "feature_extractor": "advanced_features.extract_advanced_features (215 features)",
        "training_samples": int(len(y)),
        "healthy_samples": n_neg,
        "pd_samples": n_pos,
        "label_mapping": {"0": "Healthy", "1": "Parkinsons"},
        "scaler_type": "RobustScaler",
        "calibration": "sigmoid (Platt) inside a 5-fold inner CV",
        "validation": f"{N_SPLITS}-fold stratified CV repeated {N_REPEATS}x",
        "cross_validated_scores": {
            name: {k: v for k, v in scores.items() if k != "oof_prob"}
            for name, scores in results.items()
        },
        "ensemble_members": ensemble_members,
        "ensemble_auc": ensemble_auc,
        "ensemble_accuracy": ensemble_acc,
        "decision_threshold": threshold,
        "models": saved,
    }
    with open(MODELS_DIR / "advanced_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata -> {MODELS_DIR / 'advanced_metadata.json'}")


if __name__ == "__main__":
    main()
