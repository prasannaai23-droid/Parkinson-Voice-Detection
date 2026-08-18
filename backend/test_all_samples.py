import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

import main_advanced
from advanced_features import extract_advanced_features
from live_inference import predict_live

main_advanced.load_advanced_models()
models = main_advanced.advanced_models
threshold = main_advanced.decision_threshold
print(f"\nLoaded {len(models)} models: {list(models.keys())}")

hc_dir = BACKEND_DIR.parent / "HC_AH"
pd_dir = BACKEND_DIR.parent / "PD_AH"

print("\n================ HEALTHY RECORDINGS (Target: <= 25%) ================")
hc_risks = []
for f in sorted(os.listdir(hc_dir)):
    p = hc_dir / f
    res = predict_live(str(p), models, extract_advanced_features, decision_threshold=threshold)
    bm = res["biomarkers"]
    risk = res["pd_probability"]
    hc_risks.append(risk)
    print(f"{f[:22]}: Risk={risk:5.1f}% | Jitter={bm['jitter_raw']*100:.3f}% | Shimmer={bm['shimmer_raw']*100:5.2f}% | HNR={bm['hnr_raw']:5.1f}dB | Ens={res['ensemble_prob']:.3f}")

print(f"\nALL HEALTHY (N={len(hc_risks)}): Average Risk = {sum(hc_risks)/len(hc_risks):.1f}% | Max = {max(hc_risks):.1f}% | % <= 25%: {sum(1 for r in hc_risks if r <= 25.0)/len(hc_risks)*100:.1f}%")

print("\n================ PARKINSONS RECORDINGS (Target: >= 70%) ================")
pd_risks = []
for f in sorted(os.listdir(pd_dir)):
    p = pd_dir / f
    res = predict_live(str(p), models, extract_advanced_features, decision_threshold=threshold)
    bm = res["biomarkers"]
    risk = res["pd_probability"]
    pd_risks.append(risk)
    print(f"{f[:22]}: Risk={risk:5.1f}% | Jitter={bm['jitter_raw']*100:.3f}% | Shimmer={bm['shimmer_raw']*100:5.2f}% | HNR={bm['hnr_raw']:5.1f}dB | Ens={res['ensemble_prob']:.3f}")

print(f"\nALL PARKINSONS (N={len(pd_risks)}): Average Risk = {sum(pd_risks)/len(pd_risks):.1f}% | Min = {min(pd_risks):.1f}% | % >= 70%: {sum(1 for r in pd_risks if r >= 70.0)/len(pd_risks)*100:.1f}%")
