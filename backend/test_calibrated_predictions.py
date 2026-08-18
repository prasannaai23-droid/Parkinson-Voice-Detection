import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

import joblib
from advanced_features import extract_advanced_features
from live_inference import predict_live

MODELS_DIR = BACKEND_DIR / "models"
models = {}
for name in ["advanced_svm_rbf.pkl", "advanced_rf.pkl", "advanced_logistic.pkl"]:
    p = MODELS_DIR / name
    if p.exists():
        key = name.replace("advanced_", "").replace(".pkl", "")
        models[key] = joblib.load(p)

print("Loaded models:", list(models.keys()))

hc_dir = BACKEND_DIR.parent / "HC_AH"
pd_dir = BACKEND_DIR.parent / "PD_AH"

print("\n================ HEALTHY RECORDINGS (Target: <= 25%) ================")
healthy_risks = []
for f in sorted(os.listdir(hc_dir))[:10]:
    p = hc_dir / f
    res = predict_live(str(p), models, extract_advanced_features)
    bm = res["biomarkers"]
    risk = res["pd_probability"]
    healthy_risks.append(risk)
    print(f"{f[:24]}: Risk={risk:5.1f}% | Jitter={bm['jitter_raw']*100:.3f}% | Shimmer={bm['shimmer_raw']*100:5.2f}% | HNR={bm['hnr_raw']:5.1f}dB | Ensemble={res['ensemble_prob']:.3f}")

print(f"\nHealthy average risk: {sum(healthy_risks)/len(healthy_risks):.1f}% (Max: {max(healthy_risks):.1f}%)")

print("\n================ PARKINSONS RECORDINGS (Target: >= 70%) ================")
pd_risks = []
for f in sorted(os.listdir(pd_dir))[:10]:
    p = pd_dir / f
    res = predict_live(str(p), models, extract_advanced_features)
    bm = res["biomarkers"]
    risk = res["pd_probability"]
    pd_risks.append(risk)
    print(f"{f[:24]}: Risk={risk:5.1f}% | Jitter={bm['jitter_raw']*100:.3f}% | Shimmer={bm['shimmer_raw']*100:5.2f}% | HNR={bm['hnr_raw']:5.1f}dB | Ensemble={res['ensemble_prob']:.3f}")

print(f"\nParkinson's average risk: {sum(pd_risks)/len(pd_risks):.1f}% (Min: {min(pd_risks):.1f}%)")
