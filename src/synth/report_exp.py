"""Aggregate the ECG expanded-data matrix (results/synth/exp/cnn__<config>__s<seed>_predictions.csv).  python3 -m src.synth.report_exp
Per config: pooled per-minute metrics per seed (thr 0.5 and validation-chosen threshold) -> mean/SD over the seeds actually available; paired record-clustered bootstrap
of the seed-averaged probabilities vs the `baseline` config run through the same code. Also writes mean inner-validation AUROC (training-side) per config for ratio selection."""
from __future__ import annotations
import glob, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
EXP = Path("results/synth/exp"); OUT = Path("results/synth")
rng = np.random.default_rng(11)


def load():
    d = {}
    for f in sorted(glob.glob(str(EXP / "cnn__*__s*_predictions.csv"))):
        m = re.match(r".*cnn__(.+)__s(\d+)_predictions.csv", f); cfg, s = m.group(1), int(m.group(2)); d.setdefault(cfg, {})[s] = pd.read_csv(f)
    return d


def pooled(df, use_val=False):
    y = df.y_true.values; p = df.y_prob.values; thr = df.thr_val.values if use_val else 0.5; pr = p > thr
    tp, tn, fp, fn = (pr & (y == 1)).sum(), (~pr & (y == 0)).sum(), (pr & (y == 0)).sum(), (~pr & (y == 1)).sum()
    return dict(acc=(tp + tn) / len(y), sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1), auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p))


def main():
    D = load(); rows = []; key = ["record", "minute"]
    for cfg, seeds in D.items():
        ms = [pooled(df) for df in seeds.values()]; mv = [pooled(df, True) for df in seeds.values()]; vs = [df.groupby("record").val_auc.first().mean() for df in seeds.values()]
        r = dict(config=cfg, n_seeds=len(seeds), seeds=",".join(map(str, sorted(seeds))))
        for k in ("acc", "sens", "spec", "auroc", "auprc"): r[f"{k}_mean"] = np.mean([m[k] for m in ms]); r[f"{k}_sd"] = np.std([m[k] for m in ms], ddof=1) if len(ms) > 1 else np.nan
        r["acc_valthr_mean"] = np.mean([m["acc"] for m in mv]); r["val_auc_inner_mean"] = float(np.mean(vs)); rows.append(r)
    S = pd.DataFrame(rows).sort_values("auroc_mean", ascending=False); S.to_csv(OUT / "exp_summary.csv", index=False)
    if "baseline" in D:
        base = pd.concat(D["baseline"].values()).groupby(key + ["y_true"], as_index=False).y_prob.mean(); recs = base.record.values; ur = np.unique(recs); idx = {r: np.where(recs == r)[0] for r in ur}; pb = []
        for cfg, seeds in D.items():
            if cfg == "baseline": continue
            c = pd.concat(seeds.values()).groupby(key + ["y_true"], as_index=False).y_prob.mean()
            m = base.merge(c, on=key + ["y_true"], suffixes=("_b", "_c")); recs = m.record.values; ur = np.unique(recs); idx = {r: np.where(recs == r)[0] for r in ur}; d_a, d_u = [], []
            for _ in range(500):
                sel = np.concatenate([idx[r] for r in rng.choice(ur, len(ur))]); y = m.y_true.values[sel]
                if y.min() == y.max(): continue
                d_a.append((((m.y_prob_c.values[sel] > .5) == (y == 1)).mean() - ((m.y_prob_b.values[sel] > .5) == (y == 1)).mean())); d_u.append(roc_auc_score(y, m.y_prob_c.values[sel]) - roc_auc_score(y, m.y_prob_b.values[sel]))
            yy = m.y_true.values
            pb.append(dict(config=cfg, d_acc=((m.y_prob_c > .5) == (yy == 1)).mean() - ((m.y_prob_b > .5) == (yy == 1)).mean(), d_acc_lo=np.percentile(d_a, 2.5), d_acc_hi=np.percentile(d_a, 97.5),
                           d_auroc=roc_auc_score(yy, m.y_prob_c) - roc_auc_score(yy, m.y_prob_b), d_auroc_lo=np.percentile(d_u, 2.5), d_auroc_hi=np.percentile(d_u, 97.5),
                           n_seeds=len(seeds), seed_avg_prob="yes"))
        P = pd.DataFrame(pb); P.to_csv(OUT / "exp_paired_vs_baseline.csv", index=False); print(P.round(3).to_string())
    print(S[["config", "n_seeds", "acc_mean", "acc_sd", "sens_mean", "spec_mean", "auroc_mean", "auroc_sd", "auprc_mean", "val_auc_inner_mean"]].round(3).to_string())
    # ratio curve
    rr = [(0, "baseline")] + [(float(c[6:]), c) for c in D if re.fullmatch(r"synthR[\d.]+", c)]
    if len(rr) > 1:
        rr.sort(); fig, ax = plt.subplots(1, 2, figsize=(10, 3.6)); T = S.set_index("config")
        for a_, k in zip(ax, ("auroc", "acc")):
            a_.errorbar([r for r, _ in rr], [T.loc[c, f"{k}_mean"] for _, c in rr], yerr=[T.loc[c, f"{k}_sd"] if T.loc[c, "n_seeds"] > 1 else 0 for _, c in rr], marker="o"); a_.set_xlabel("synthetic : real training minutes"); a_.set_ylabel(k + " (LOSO, real held-out)")
        fig.tight_layout(); fig.savefig(OUT / "fig_ratio_curve.png", dpi=120)


if __name__ == "__main__":
    main()


def ensemble_of_expanded(top_k: int = 3):
    """Pre-specified: the top_k NON-baseline configs by mean inner-validation AUROC (training-side, real validation records only) are soft-averaged seed-by-seed
    (equal weights, thr 0.5) and compared with the baseline and with the best single expanded model on the same minutes. No test labels used for the choice."""
    D = load(); S = pd.read_csv(OUT / "exp_summary.csv"); S = S[(S.config != "baseline") & (S.n_seeds >= 3) & (S.config != "synth_only")]
    pick = list(S.sort_values("val_auc_inner_mean", ascending=False).config.head(top_k)); key = ["record", "minute", "y_true"]; rows = []
    for s in sorted(D["baseline"]):
        m = D[pick[0]][s][key + ["y_prob"]].rename(columns={"y_prob": pick[0]})
        for c in pick[1:]: m = m.merge(D[c][s][key + ["y_prob"]].rename(columns={"y_prob": c}), on=key)
        m["y_prob"] = m[pick].mean(1); m["thr_val"] = 0.5; m["val_auc"] = np.nan; rows.append(dict(seed=s, **pooled(m)))
    R = pd.DataFrame(rows); R.to_csv(OUT / "exp_ensemble_expanded.csv", index=False)
    json.dump(dict(members=pick, rule=f"top-{top_k} non-baseline configs by mean inner-validation AUROC", mean=R.drop(columns="seed").mean().to_dict(), sd=R.drop(columns="seed").std().to_dict()), open(OUT / "exp_ensemble_expanded.json", "w"), indent=1)
    print("expanded-data ensemble", pick, R.drop(columns="seed").mean().round(3).to_dict())


if __name__ == "__main__" and "--ensemble" in sys.argv:
    ensemble_of_expanded()
