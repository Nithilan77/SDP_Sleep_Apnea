"""Phase 3 analysis of the saved OOF ensemble predictions: re-verification, record-clustered paired bootstrap vs the strongest
single model, per-record comparison, latency/size, plots.   python3 -m src.ensemble.report

Primary (pre-designated, NOT chosen by test score) ensemble: cnn+tcn+hgb soft average (no fitted second-level parameters).
All other combinations are exploratory; the best-by-test-accuracy ensemble is shown but flagged as selection-optimistic."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ensemble.run_ensemble import metrics, load, BASE, SEEDS

OUT = Path("results/ensemble"); NB = 1000; rng = np.random.default_rng(7)


def col(df, name, strat, s): return df[f"{name}|{strat}|s{s}"].values


def main():
    df = pd.read_csv(OUT / "oof_ensemble_predictions.csv.gz"); y = df.y_true.values; rec = df.record.values
    # 1. independent re-verification of the stored per-seed metrics from the stored predictions
    ps = pd.read_csv(OUT / "per_seed_metrics.csv"); bad = 0; n = 0
    for _, r in ps[ps.rule == "thr0.5"].iterrows():
        if r.strategy not in ("single", "soft_avg"): continue
        t = 0.5; m = metrics(y, col(df, r.model, r.strategy, int(r.seed)), t); n += 1
        bad += abs(m["acc"] - r.acc) > 1e-4 or abs(m["auroc"] - r.auroc) > 1e-4
    print(f"re-verified {n} rows from saved predictions, mismatches: {bad}")
    # 2. paired record-clustered bootstrap
    cands = {"hgb (strongest single)": ("hgb", "single"), "cnn (historical primary arch)": ("cnn", "single"),
             "PRIMARY ens: cnn+tcn+hgb soft": ("cnn+tcn+hgb", "soft_avg"), "cnn+hgb soft": ("cnn+hgb", "soft_avg"),
             "cnn+tcn+hgb weighted": ("cnn+tcn+hgb", "weighted_avg"), "cnn+tcn+hgb+logreg soft (best AUROC)": ("cnn+tcn+hgb+logreg", "soft_avg"),
             "cnn+tcn+hgb stacking": ("cnn+tcn+hgb", "stacking")}
    recs = np.unique(rec); idx = {r: np.where(rec == r)[0] for r in recs}
    P = {k: [col(df, m, s_, s) for s in SEEDS] for k, (m, s_) in cands.items()}
    def stats(sel, k):   # seed-mean accuracy and AUROC on the bootstrap sample
        a, u = [], []
        for p in P[k]:
            a.append(((p[sel] > 0.5) == (y[sel] == 1)).mean()); u.append(roc_auc_score(y[sel], p[sel]))
        return np.mean(a), np.mean(u)
    ref = "hgb (strongest single)"; boots = {k: [] for k in cands}; refb = []
    for b in range(NB):
        pick = rng.choice(recs, len(recs)); sel = np.concatenate([idx[r] for r in pick])
        if y[sel].min() == y[sel].max(): continue
        refb.append(stats(sel, ref))
        for k in cands: boots[k].append(stats(sel, k))
    refb = np.array(refb); rows = []
    for k in cands:
        full = np.arange(len(y)); a0, u0 = stats(full, k); ra0, ru0 = stats(full, ref); bb = np.array(boots[k])
        d = bb - refb; ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
        rows.append(dict(model=k, acc=a0, auroc=u0, d_acc_vs_hgb=a0 - ra0, d_acc_lo=ci(d[:, 0])[0], d_acc_hi=ci(d[:, 0])[1],
                         d_auroc_vs_hgb=u0 - ru0, d_auroc_lo=ci(d[:, 1])[0], d_auroc_hi=ci(d[:, 1])[1]))
    R = pd.DataFrame(rows); R.to_csv(OUT / "paired_vs_best_single.csv", index=False); print(R.round(3).to_string())
    # 3. per-record: seed-mean accuracy hgb vs primary ensemble
    pr = []
    for r in recs:
        i = idx[r]; row = dict(record=r, n=len(i), apnea_frac=y[i].mean())
        for k in ("hgb (strongest single)", "PRIMARY ens: cnn+tcn+hgb soft", "cnn (historical primary arch)"):
            row[k] = np.mean([((p[i] > 0.5) == (y[i] == 1)).mean() for p in P[k]])
        pr.append(row)
    pr = pd.DataFrame(pr); pr["ens_minus_hgb"] = pr["PRIMARY ens: cnn+tcn+hgb soft"] - pr["hgb (strongest single)"]
    pr.to_csv(OUT / "per_record_primary_vs_hgb.csv", index=False)
    print("records improved/worse (|d|>0.01):", (pr.ens_minus_hgb > .01).sum(), (pr.ens_minus_hgb < -.01).sum(), "of", len(pr))
    # 4. error overlap / disagreement between base models (seed 0)
    err = {m: ((col(df, m, "single", 0) > 0.5) != (y == 1)) for m in BASE}; names = list(BASE)
    ov = pd.DataFrame([[(err[a] & err[b]).sum() / max(err[a].sum(), 1) for b in names] for a in names], index=names, columns=names)
    ov.to_csv(OUT / "error_overlap_seed0.csv"); print(ov.round(2))
    # 5. plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].errorbar(range(len(R)), R.d_acc_vs_hgb, yerr=[R.d_acc_vs_hgb - R.d_acc_lo, R.d_acc_hi - R.d_acc_vs_hgb], fmt="o"); ax[0].axhline(0, color="k", lw=.7)
    ax[0].set_xticks(range(len(R))); ax[0].set_xticklabels(R.model, rotation=40, ha="right", fontsize=7); ax[0].set_title("Accuracy @0.5 minus HGB (95% record-bootstrap CI)", fontsize=9)
    ax[1].bar(pr.record, pr.ens_minus_hgb, color=np.where(pr.ens_minus_hgb > 0, "#2E7D32", "#C62828")); ax[1].tick_params(axis="x", rotation=90, labelsize=6)
    ax[1].set_title("Per-record accuracy: primary ensemble minus HGB", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig_ensemble_vs_best_single.png", dpi=130)


if __name__ == "__main__":
    main()


def latency():
    """CPU inference latency per 1000 one-minute windows (random weights: latency does not depend on weight values)."""
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from src.ecg.expA_multiseed import make
    torch.set_num_threads(1); X = torch.randn(1000, 2, 60); rows = []
    for kind in ("cnn", "tcn", "gru", "lstm"):
        m = make(kind).eval(); npar = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            m(X); t = time.perf_counter(); [m(X) for _ in range(5)]; dt = (time.perf_counter() - t) / 5
        rows.append(dict(model=kind, params=npar, ms_per_1000_windows_cpu1=1e3 * dt))
    F = np.random.randn(5000, 90); yy = (np.random.rand(5000) > .6).astype(int)
    h = HGB(max_iter=300, learning_rate=.05, max_leaf_nodes=15).fit(F, yy); h.predict_proba(F[:10]); t = time.perf_counter(); [h.predict_proba(F[:1000]) for _ in range(5)]
    rows.append(dict(model="hgb (excl. HRV feature extraction)", params=None, ms_per_1000_windows_cpu1=1e3 * (time.perf_counter() - t) / 5))
    L = pd.DataFrame(rows); L.to_csv(OUT / "latency_cpu.csv", index=False); print(L.round(2))


if __name__ == "__main__" and "--latency" in sys.argv:
    latency()
