"""Experiment B -- feature-based baseline: gradient boosting (sklearn HistGradientBoosting) on per-minute HRV/CVHR features.

Same minutes, same 35-fold LOSO, same inner-validation protocol as expA_multiseed (5 seeded validation records per fold; early stopping
and threshold chosen on them; test record used once; scaling not needed for trees). Feature sets:
  within : Phase-2 CVHR (minus hr_delta_prev) + HRV banks (time/freq/nonlinear) + RR/amp channel summary stats  -- same information as the CNN input
  +prev  : within + hr_delta_prev (uses the previous minute; causal but not the same input as the CNN)
Columns of the HRV-bank csv are joined on (record, minute); minutes lacking features keep NaN (HistGB handles NaN natively).

    python3 -m src.ecg.expB_hrv_gbm --seeds 0 1 2
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ecg.cnn_apnea_final import restrict_to_phase2_minutes
from src.ecg.expA_multiseed import bal_thr
from src.eval.splits import leave_one_subject_out
OUT = Path("results/expA_multiseed")


def seq_stats(x):  # x: (n, 2, 60) raw
    f = []
    for c in range(2):
        s = x[:, c]; d = np.diff(s, axis=1)
        f += [s.mean(1), s.std(1), s.min(1), s.max(1), np.median(np.abs(s - np.median(s, 1, keepdims=True)), 1), np.abs(d).mean(1), d.std(1),
              (np.sign(d[:, 1:]) != np.sign(d[:, :-1])).mean(1)]
    return np.stack(f, 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); a = ap.parse_args()
    ds = dict(np.load("results/phase3_cnn_v2/sequences.npz", allow_pickle=True)); mask = restrict_to_phase2_minutes(ds)
    base = pd.DataFrame(dict(record=ds["record"][mask], minute=ds["minute"][mask])); base["y"] = (ds["label"][mask] == "A").astype(int)
    p2 = pd.read_csv("results/phase2_cvhr_baseline/features.csv").drop(columns=["label", "n_beats"])
    hb = pd.read_csv("results/feature_study/hrv_bank_features.csv").drop(columns=["label", "n_beats", "sdnn_ms", "rmssd_ms", "pnn50", "mean_hr"])
    df = base.merge(p2, on=["record", "minute"], how="left").merge(hb, on=["record", "minute"], how="left")
    assert len(df) == len(base) == mask.sum()
    S = seq_stats(ds["sequence"][mask]); sc = [f"s{i}" for i in range(S.shape[1])]; df[sc] = S
    prev = ["hr_delta_prev"]; within = [c for c in df.columns if c not in ("record", "minute", "y") + tuple(prev)]
    sets = {"within": within, "plusprev": within + prev}
    recs = sorted(df.record.unique()); rows = []
    for fs_name, cols in sets.items():
        for name in ("hgb", "logreg"):
            for seed in a.seeds:
                t0 = time.time()
                for k, (tr, te) in enumerate(leave_one_subject_out(recs), 1):
                    r = np.random.default_rng(1000 * seed + k); val = list(r.choice(tr, 5, replace=False)); fit_r = [x for x in tr if x not in val]
                    fi, vi, ei = df.record.isin(fit_r).values, df.record.isin(val).values, (df.record == te).values
                    X = df[cols].values
                    if name == "hgb":
                        m = HGB(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, early_stopping=False, class_weight="balanced", random_state=seed)
                        # pick iteration count on validation AUROC via staged predictions
                        m.fit(X[fi], df.y.values[fi])
                        best, bi = -1, 0
                        for i, pv in enumerate(m.staged_predict_proba(X[vi])):
                            au = roc_auc_score(df.y.values[vi], pv[:, 1])
                            if au > best: best, bi = au, i
                        gen = m.staged_predict_proba
                        pva = list(gen(X[vi]))[bi][:, 1]; pte = list(gen(X[ei]))[bi][:, 1]
                    else:
                        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
                        m.fit(X[fi], df.y.values[fi]); pva = m.predict_proba(X[vi])[:, 1]; pte = m.predict_proba(X[ei])[:, 1]; bi = 0
                    thr = bal_thr(df.y.values[vi], pva)
                    rows.append(pd.DataFrame(dict(model=f"{name}_{fs_name}", seed=seed, record=df.record.values[ei], minute=df.minute.values[ei], y_true=df.y.values[ei], y_prob=pte, thr_val=thr, best_epoch=bi + 1)))
                print(f"[{name}_{fs_name}] seed {seed} done {time.time()-t0:.0f}s", flush=True)
                pd.concat(rows, ignore_index=True).to_csv(OUT / "tabular_predictions.csv", index=False)


if __name__ == "__main__":
    main()
