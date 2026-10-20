"""Phase 3 ensembles over the saved leave-one-subject-out out-of-fold predictions of Experiments A/B.

Base predictions: results/expA_multiseed/*_predictions.csv (35 records x 16,949 minutes x 3 seeds; identical population
verified below). Each base prediction for record r came from a model that never saw r (inner-validation early stopping,
thresholds and normalisation fitted on the other records).

Second-level protocol ("LOSO over OOF"): for held-out record r, ensemble weights / Platt calibrators / stacking
coefficients / decision threshold are fitted ONLY on the OOF predictions and labels of the other 34 records, then applied
frozen to r. Residual caveat (documented, unavoidable without 35x34 nested retraining): the base prediction of a training
record j came from a model trained on a set that included r, so r's labels had an indirect, weak, 1-fold-removed influence on
the second-level inputs. With <= 7 second-level parameters this influence is small, but it is not zero; the result is therefore
reported as 'OOF-stacked LOSO', not as fully nested.

    python3 -m src.ensemble.run_ensemble
"""
import itertools, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
A = Path("results/expA_multiseed"); OUT = Path("results/ensemble")
BASE = {"cnn": ("cnn", "cnn"), "tcn": ("tcn", "tcn"), "gru": ("gru", "gru"), "lstm": ("lstm", "lstm"),
        "hgb": ("tabular", "hgb_within"), "logreg": ("tabular", "logreg_within")}
SEEDS = [0, 1, 2]
PRESPEC = [("cnn", "tcn"), ("cnn", "hgb"), ("tcn", "hgb"), ("cnn", "tcn", "hgb"), ("cnn", "tcn", "gru", "lstm"),
           ("hgb", "logreg"), ("cnn", "tcn", "hgb", "logreg"), ("cnn", "tcn", "gru", "lstm", "hgb", "logreg")]
EPS = 1e-4

def load():
    out, keys = {}, None
    for name, (f, mod) in BASE.items():
        d = pd.read_csv(A / f"{f}_predictions.csv"); d = d[d.model == mod]
        for s in SEEDS:
            g = d[d.seed == s].sort_values(["record", "minute"]).reset_index(drop=True)
            k = g[["record", "minute", "y_true"]]
            if keys is None: keys = k
            assert k.equals(keys), f"population mismatch {name} seed {s}"
            out[(name, s)] = g.y_prob.values
    return keys, out

def metrics(y, p, thr=0.5):
    pred = p > thr; tp = (pred & (y == 1)).sum(); tn = (~pred & (y == 0)).sum(); fp = (pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum()
    se, sp = tp / max(tp + fn, 1), tn / max(tn + fp, 1); pr = tp / max(tp + fp, 1)
    pc = np.clip(p, EPS, 1 - EPS); bins = np.minimum((pc * 10).astype(int), 9); ece = 0.0
    for b in range(10):
        m = bins == b
        if m.any(): ece += m.mean() * abs(pc[m].mean() - y[m].mean())
    return dict(acc=(tp + tn) / len(y), sens=se, spec=sp, prec=pr, f1=2 * pr * se / max(pr + se, 1e-9), auroc=roc_auc_score(y, p),
                auprc=average_precision_score(y, p), brier=brier_score_loss(y, pc), ece=ece, tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))

def best_thr(y, p):
    grid = np.arange(0.2, 0.81, 0.02); bal = [((p > t)[y == 1].mean() + (~(p > t))[y == 0].mean()) / 2 for t in grid]
    return float(grid[int(np.argmax(bal))])

def fit_weights(P, y):                     # softmax weights minimising log-loss of the weighted probability mean
    k = P.shape[1]
    def f(a):
        w = np.exp(a - a.max()); w /= w.sum(); q = np.clip(P @ w, EPS, 1 - EPS); return -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()
    a = minimize(f, np.zeros(k), method="L-BFGS-B").x; w = np.exp(a - a.max()); return w / w.sum()

def second_level(strategy, Ptr, ytr, Pte):
    """returns (prob_te, threshold_from_train) for one fold; fitted on training records only."""
    k = Ptr.shape[1]
    if strategy == "hard_vote":
        return (Pte > 0.5).mean(1), 0.5 - 1e-9                          # fraction of models voting apnea; decision > 0.5 = strict majority
    if strategy == "soft_avg": q = Pte.mean(1); qt = Ptr.mean(1)
    elif strategy == "weighted_avg": w = fit_weights(Ptr, ytr); q = Pte @ w; qt = Ptr @ w
    elif strategy == "calibrated_avg":
        cal = [LogisticRegression(C=1e6).fit(logit(np.clip(Ptr[:, i:i + 1], EPS, 1 - EPS)), ytr) for i in range(k)]
        f = lambda P: np.mean([c.predict_proba(logit(np.clip(P[:, i:i + 1], EPS, 1 - EPS)))[:, 1] for i, c in enumerate(cal)], 0)
        q, qt = f(Pte), f(Ptr)
    elif strategy == "stacking":
        lr = LogisticRegression(C=1.0).fit(logit(np.clip(Ptr, EPS, 1 - EPS)), ytr)
        q = lr.predict_proba(logit(np.clip(Pte, EPS, 1 - EPS)))[:, 1]; qt = lr.predict_proba(logit(np.clip(Ptr, EPS, 1 - EPS)))[:, 1]
    return q, best_thr(ytr, qt)

STRATS = ["hard_vote", "soft_avg", "weighted_avg", "calibrated_avg", "stacking"]

def run_combo(combo, strategy, keys, preds, seed):
    y = keys.y_true.values; rec = keys.record.values; P = np.column_stack([preds[(m, seed)] for m in combo])
    q = np.zeros(len(y)); thr = np.zeros(len(y))
    for r in np.unique(rec):
        te = rec == r; tr = ~te
        q[te], t = second_level(strategy, P[tr], y[tr], P[te]); thr[te] = t
    return q, thr

def main():
    t0 = time.time(); keys, preds = load(); y = keys.y_true.values; rec = keys.record.values
    print("population identical across 6 base models x 3 seeds:", len(keys), "minutes,", keys.record.nunique(), "records")
    rows, probs = [], {}
    for m in BASE:                                          # individual models
        for s in SEEDS:
            probs[(m, "single", s)] = (preds[(m, s)], np.full(len(y), 0.5))
    combos = [(m,) for m in BASE] + PRESPEC
    for combo in PRESPEC:
        for st in STRATS:
            for s in SEEDS: probs[("+".join(combo), st, s)] = run_combo(combo, st, keys, preds, s)
        print("done", combo, f"{time.time()-t0:.0f}s", flush=True)
    for (name, st, s), (q, thr) in probs.items():
        for rule, t in (("thr0.5", 0.5 if st != "hard_vote" else 0.5 - 1e-9), ("thr_train", thr)):
            r = metrics(y, q, t) if rule == "thr0.5" else metrics(y, q, thr)
            r.update(model=name, strategy=st, seed=s, rule=rule); rows.append(r)
    R = pd.DataFrame(rows); R.to_csv(OUT / "per_seed_metrics.csv", index=False)
    cols = ["acc", "sens", "spec", "prec", "f1", "auroc", "auprc", "brier", "ece"]
    S = R.groupby(["model", "strategy", "rule"])[cols].agg(["mean", "std"]); S.to_csv(OUT / "summary_mean_sd.csv")
    # seed-averaged probabilities saved for bootstrap / dashboard
    sav = keys.copy()
    for (name, st, s), (q, thr) in probs.items(): sav[f"{name}|{st}|s{s}"] = q
    sav.to_csv(OUT / "oof_ensemble_predictions.csv.gz", index=False, float_format="%.5f")
    json.dump({"runtime_s": time.time() - t0, "prespecified_combos": ["+".join(c) for c in PRESPEC], "strategies": STRATS}, open(OUT / "run_meta.json", "w"))
    print(S.xs("thr0.5", level="rule")[[("acc", "mean"), ("auroc", "mean")]].round(3).sort_values(("auroc", "mean"), ascending=False).head(25))

if __name__ == "__main__":
    main()
