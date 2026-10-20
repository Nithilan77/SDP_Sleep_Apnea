"""MESA ensembles over saved subject-grouped OOF predictions (data/mesa/cache/*/preds.npz; 220 subjects, 156,826 30 s epochs).
Second level: grouped 10-fold CV over subjects; weights/stacker fitted on training subjects' OOF predictions only
(same indirect-influence caveat as src/ensemble/run_ensemble.py). Wearable-deployable combos exclude SpO2 (not on our hardware).
    python3 -m src.ensemble.mesa_ensemble"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
C = Path("data/mesa/cache"); OUT = Path("results/ensemble")
M = {"ecg": "ecg_baseline", "headline": "canet/headline", "resp": "canet/resp_only", "spo2": "canet/spo2_only", "ecg_spo2": "canet/ecg_spo2",
     "ceiling": "canet/ceiling_spo2", "ecg_spo2_lstm": "canet_lstm/ecg_spo2_lstm", "all_lstm": "canet_lstm/ecg_resp_spo2_lstm"}
COMBOS = [("headline",), ("ecg", "headline"), ("ecg", "resp"), ("ecg", "headline", "resp"), ("ceiling",), ("ecg_spo2", "headline"),
          ("ceiling", "ecg_spo2", "headline"), ("ceiling", "all_lstm")]
d = {k: np.load(C / v / "preds.npz", allow_pickle=True) for k, v in M.items()}
ref = d["headline"]
for k, z in d.items():
    assert (z["subject"] == ref["subject"]).all() and (z["epoch"] == ref["epoch"]).all() and (z["y"] == ref["y"]).all(), k
y = ref["y"].astype(int); g = ref["subject"]; P = {k: np.clip(z["p"].astype(float), 1e-4, 1 - 1e-4) for k, z in d.items()}
def met(p, t=0.5):
    pr = p > t; tp = (pr & (y == 1)).sum(); tn = (~pr & (y == 0)).sum()
    return dict(acc=(tp + tn) / len(y), sens=tp / (y == 1).sum(), spec=tn / (y == 0).sum(), auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p))
def boot_auc(p, q, n=300, rng=np.random.default_rng(3)):
    subs = np.unique(g); ix = {s: np.where(g == s)[0] for s in subs}; r = []
    for _ in range(n):
        sel = np.concatenate([ix[s] for s in rng.choice(subs, len(subs))]); r.append(roc_auc_score(y[sel], p[sel]) - roc_auc_score(y[sel], q[sel]))
    return np.percentile(r, [2.5, 97.5])
rows = []; out = {}
for k in M: rows.append(dict(model=k, strategy="single", **met(P[k]))); out[(k,)] = P[k]
for combo in COMBOS:
    if len(combo) == 1: continue
    X = np.column_stack([logit(P[k]) for k in combo])
    soft = np.mean([P[k] for k in combo], 0); stk = np.zeros(len(y))
    for tr, te in GroupKFold(10).split(X, y, g):
        stk[te] = LogisticRegression(C=1.0, class_weight="balanced").fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    for st, p in (("soft_avg", soft), ("stacking", stk)):
        rows.append(dict(model="+".join(combo), strategy=st, **met(p))); out[("+".join(combo), st)] = p
R = pd.DataFrame(rows); R.to_csv(OUT / "mesa_ensemble_metrics.csv", index=False); print(R.round(3).to_string())
cmp = []
for a, b in ((("ecg", "headline", "resp"), "soft_avg"), (("ecg", "headline"), "stacking"), (("ceiling", "ecg_spo2", "headline"), "stacking")):
    n = "+".join(a); best_single = max([k for k in a], key=lambda k: roc_auc_score(y, P[k]))
    lo, hi = boot_auc(out[(n, b)], P[best_single]); cmp.append(dict(ensemble=f"{n}|{b}", vs=best_single, d_auroc=roc_auc_score(y, out[(n, b)]) - roc_auc_score(y, P[best_single]), lo=lo, hi=hi))
C_ = pd.DataFrame(cmp); C_.to_csv(OUT / "mesa_ensemble_paired.csv", index=False); print(C_.round(3))
