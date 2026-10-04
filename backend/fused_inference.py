"""Conservative multimodal voice and gait probability fusion."""

from __future__ import annotations

from typing import Any

import numpy as np


class FusionInputError(ValueError):
    """Raised when one modality cannot produce a trustworthy probability."""


def gait_highlights(features: list[float]) -> list[dict[str, str]]:
    """Describe the measured 12-feature gait vector used by the video model."""
    ankle_mean, ankle_std, ankle_range = features[0:3]
    knee_mean, knee_std, knee_range = features[3:6]
    wrist_mean, wrist_std, wrist_range = features[6:9]
    variability = ankle_std / max(abs(ankle_mean), 1e-6)
    return [
        {"name": "Stride variability", "value": f"{variability * 100:.1f}%", "signal": "higher" if variability > 0.2 else "stable"},
        {"name": "Stride range", "value": f"{ankle_range:.3f}", "signal": "wide" if ankle_range > 0.15 else "compact"},
        {"name": "Knee stability", "value": f"{knee_std:.3f}", "signal": "variable" if knee_std > 0.08 else "steady"},
        {"name": "Arm movement range", "value": f"{wrist_range:.3f}", "signal": "wide" if wrist_range > 0.15 else "compact"},
    ]


def _probability(model: Any, features: np.ndarray) -> float:
    if not hasattr(model, "predict_proba"):
        raise FusionInputError("The gait model does not expose calibrated probabilities")
    values = model.predict_proba(features)
    if values.ndim != 2 or values.shape[1] < 2:
        raise FusionInputError("The gait model returned an invalid probability shape")
    probability = float(values[0, 1])
    if not np.isfinite(probability):
        raise FusionInputError("The gait model returned a non-finite probability")
    return float(np.clip(probability, 0.0, 1.0))


def fuse_probabilities(voice_probability: float, gait_probability: float) -> dict[str, Any]:
    """Fuse two probabilities and abstain when their evidence conflicts."""
    voice_probability = float(np.clip(voice_probability, 0.0, 1.0))
    gait_probability = float(np.clip(gait_probability, 0.0, 1.0))
    fused_probability = (voice_probability + gait_probability) / 2.0
    same_side = (voice_probability >= 0.5) == (gait_probability >= 0.5)
    agreement = abs(voice_probability - gait_probability) <= 0.25

    if same_side and agreement:
        prediction = "Parkinson's Risk" if fused_probability >= 0.5 else "Healthy"
        decision = "classified"
    else:
        prediction = "Inconclusive"
        decision = "abstain"

    return {
        "voice_probability": round(voice_probability * 100.0, 1),
        "gait_probability": round(gait_probability * 100.0, 1),
        "fused_probability": round(fused_probability * 100.0, 1),
        "prediction": prediction,
        "decision": decision,
        "agreement": round((1.0 - abs(voice_probability - gait_probability)) * 100.0, 1),
    }


def predict_gait_probability(model: Any, features: list[float]) -> float:
    """Validate the extracted 12-feature gait vector and score it."""
    values = np.asarray(features, dtype=np.float64).reshape(1, -1)
    if values.shape[1] != 12 or not np.isfinite(values).all() or np.all(values == 0):
        raise FusionInputError("Gait pose extraction did not produce valid features")
    return _probability(model, values)
