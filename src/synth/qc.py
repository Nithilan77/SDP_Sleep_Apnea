"""Quality control of a synthetic dataset against the measured in-house specification.
    python3 -m src.synth.qc --dataset data/synthetic/medium --n 40 --out results/synth/qc_medium
Outputs: manifest summary, per-feature real-vs-synthetic comparison (coverage of real p5-p95, KS), domain-classifier AUROC (can a model tell real from synthetic?),
diversity / near-duplicate check, figures (PSD, example overlays)."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import welch, butter, sosfiltfilt
from scipy.stats import ks_2samp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ingest.sensortile import load_sensortile
from src.synth import spec as SP

FEATS = ["qvar_sd", "hf_noise_sd_lsb", "wander_sd_lsb", "hr_bpm", "breath_band_rms_mg", "acc_noise_rms_mg_20_60hz", "grav_mag_mg", "qrs_fwhm_ms", "template_peak_lsb"]


def window_feats(rec, win_s=60.0, n_max=60, seed=0):
    fs = rec.fs_measured; q = rec.qvar.astype(float); a = rec.accel_mg; n = int(win_s * fs); rng = np.random.default_rng(seed); rows = []
    starts = rng.choice(np.arange(0, len(q) - n, n), size=min(n_max, max(1, (len(q) - n) // n)), replace=False)
    sos_hf = butter(3, [40, 100], "band", fs=fs, output="sos"); sos_w = butter(2, 0.5, "low", fs=fs, output="sos"); sos_b = butter(3, [0.08, 0.6], "band", fs=fs, output="sos")
    for s in starts:
        x = q[s:s + n]; m = np.linalg.norm(a[s:s + n], axis=1)
        rows.append([x.std(), sosfiltfilt(sos_hf, x).std(), sosfiltfilt(sos_w, x).std(), sosfiltfilt(sos_b, m - np.median(m)).std(), np.median(m), a[s:s + n].std(0).mean(), len(np.unique(x))])
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--n", type=int, default=40); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); D = Path(a.dataset)
    man = [json.loads(l) for l in open(D / "manifest.jsonl")]
    s = dict(n=len(man), valid=sum(m["validation"]["valid"] for m in man), scenarios=pd.Series([m["scenario"] for m in man]).value_counts().to_dict(), tiles=pd.Series([m["tile"] for m in man]).value_counts().to_dict(),
             quality=pd.Series([m["profile"]["quality"] for m in man]).value_counts().to_dict(), failed_required=pd.Series([f for m in man for f in m["validation"]["failed_required"]]).value_counts().to_dict(),
             optional_failed=pd.Series([f for m in man for f in m["validation"]["optional_failed"]]).value_counts().to_dict(), n_events=sum(len(m["events"]) for m in man),
             unique_head_hashes=len({m["head_sha256_16"] for m in man}), total_hours=sum(m["duration_s"] for m in man) / 3600)
    # recording-level features: same code as the real spec
    syn = [m for m in man if m["tile"] == "ECG" and m["validation"]["valid"]][:a.n]; rows = []
    for m in syn:
        r = SP.analyse(D / m["file"]) if (D / m["file"]).suffix == ".csv" else None
        if r: rows.append(r)
    syn_df = pd.DataFrame(rows); real = pd.read_csv("results/synth/spec/per_recording_table.csv"); real = real[real.cardiac_prominence > 20]
    cmp = []
    for f in FEATS:
        rv, sv = real[f].dropna().astype(float), syn_df[f].dropna().astype(float)
        if len(rv) < 3 or len(sv) < 3: continue
        lo, hi = np.percentile(rv, [5, 95]); ks = ks_2samp(rv, sv)
        cmp.append(dict(feature=f, real_p5=lo, real_median=rv.median(), real_p95=hi, synth_p5=np.percentile(sv, 5), synth_median=sv.median(), synth_p95=np.percentile(sv, 95),
                        frac_synth_inside_real_range=float(((sv >= lo) & (sv <= hi)).mean()), ks_stat=ks.statistic, ks_p=ks.pvalue, n_real=len(rv), n_synth=len(sv)))
    C = pd.DataFrame(cmp); C.to_csv(out / "real_vs_synth_features.csv", index=False)
    # diversity: standardised recording-level feature distance
    Z = syn_df[FEATS].astype(float).dropna(axis=1, how="any"); Z = (Z - Z.mean()) / (Z.std() + 1e-9); dist = np.linalg.norm(Z.values[:, None] - Z.values[None], axis=2); np.fill_diagonal(dist, np.inf)
    s["diversity"] = dict(n_compared=len(Z), min_pairwise_dist=float(dist.min()), median_nn_dist=float(np.median(dist.min(1))), near_duplicates_dist_lt_0p3=int((dist.min(1) < 0.3).sum()))
    # domain classifier on 60 s windows (grouped by recording)
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    X, y, g = [], [], []
    real_files = [p for p in SP.FILES if p.exists() and p.stat().st_size > 0]
    for i, p in enumerate(real_files):
        try:
            rec = load_sensortile(p)
            if rec.duration_s < 600: continue
            w = window_feats(rec, n_max=40, seed=i); X.append(w); y += [0] * len(w); g += [f"R{i}"] * len(w)
        except Exception: pass
    for j, m in enumerate(syn):
        rec = load_sensortile(D / m["file"]); w = window_feats(rec, n_max=20, seed=j); X.append(w); y += [1] * len(w); g += [f"S{j}"] * len(w)
    X = np.vstack(X); y = np.array(y); g = np.array(g); P = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g): P[te] = HGB(max_iter=100).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    names = ["qvar_sd", "hf_noise", "wander", "breath_band_rms", "grav_mag", "acc_axis_sd", "qvar_unique"]
    imp = {}
    for k, nme in enumerate(names):
        Xs = X.copy(); Xs[:, k] = np.random.default_rng(0).permutation(Xs[:, k]); Ps = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(Xs, y, g): Ps[te] = HGB(max_iter=100).fit(X[tr], y[tr]).predict_proba(Xs[te])[:, 1]
        imp[nme] = float(roc_auc_score(y, P) - roc_auc_score(y, Ps))
    s["domain_classifier"] = dict(auroc_real_vs_synthetic=float(roc_auc_score(y, P)), n_real_windows=int((y == 0).sum()), n_synth_windows=int((y == 1).sum()), perm_importance_auroc_drop=imp,
                                  reading="0.5 = indistinguishable; values near 1 mean the synthetic windows carry artificial cues a model could exploit")
    # figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    r0 = [p for p in real_files if p.name.endswith("ECG_07_10_2026.csv")][0]; rr = load_sensortile(r0); sr = load_sensortile(D / syn[0]["file"])
    for rec, lab, c in ((rr, "real (session-1 ECG tile)", "k"), (sr, f"synthetic ({syn[0]['id']})", "tab:red")):
        f, Pw = welch(rec.qvar - rec.qvar.mean(), rec.fs_measured, nperseg=int(8 * rec.fs_measured)); ax[0].semilogy(f, Pw, label=lab, color=c, lw=.8)
        t0 = int(120 * rec.fs_measured); ax[1].plot(np.arange(int(8 * rec.fs_measured)) / rec.fs_measured, rec.qvar[t0:t0 + int(8 * rec.fs_measured)], color=c, lw=.7, label=lab)
        f2, P2 = welch(np.linalg.norm(rec.accel_mg, axis=1) - np.median(np.linalg.norm(rec.accel_mg, axis=1)), rec.fs_measured, nperseg=int(60 * rec.fs_measured)); ax[2].semilogy(f2, P2, color=c, lw=.8)
    ax[0].set_xlim(0, 60); ax[0].set_title("QVAR PSD"); ax[0].legend(fontsize=7); ax[1].set_title("QVAR 8 s excerpt (LSB)"); ax[2].set_xlim(0, 3); ax[2].set_title("|accel| PSD (mg²/Hz)")
    fig.tight_layout(); fig.savefig(out / "fig_real_vs_synth.png", dpi=120)
    json.dump(s, open(out / "qc_summary.json", "w"), indent=1, default=float); print(json.dumps(s, indent=1, default=float)); print(C.round(2).to_string())


if __name__ == "__main__":
    main()
