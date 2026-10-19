"""Aggregate Experiment A/B predictions -> results/expA_multiseed/summary.{json,md}. Pooled per-minute metrics per seed, mean +/- SD over seeds,
paired record-level bootstrap (resampling the 35 records) of the seed-averaged difference vs the CNN."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path("results/expA_multiseed")
parts = [pd.read_csv(p) for p in sorted(OUT.glob("*_predictions.csv"))]
df = pd.concat(parts, ignore_index=True)

def m(y, pred):
    tp = ((pred == 1) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    se, sp = tp / max(tp + fn, 1), tn / max(tn + fp, 1); pr = tp / max(tp + fp, 1)
    return dict(acc=(tp + tn) / len(y), sens=se, spec=sp, f1=2 * pr * se / max(pr + se, 1e-9), bal=(se + sp) / 2, tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))

rows = []
for (mod, seed), g in df.groupby(["model", "seed"]):
    y, p = g.y_true.values, g.y_prob.values
    for rule, pred in (("thr0.5", (p > .5).astype(int)), ("valthr", (p > g.thr_val.values).astype(int))):
        r = m(y, pred); r.update(model=mod, seed=seed, rule=rule, auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p))
        gg = g.assign(pr=pred).groupby("record").agg(t=("y_true", "mean"), q=("pr", "mean"))
        isapnea = ~gg.index.str.startswith("c"); r["per_recording_correct"] = int((((gg.q > .05) == isapnea)).sum())
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv(OUT / "per_seed_metrics.csv", index=False)
cols = ["acc", "sens", "spec", "f1", "auroc", "auprc", "per_recording_correct"]
agg = R.groupby(["model", "rule"])[cols].agg(["mean", "std"]); agg.to_csv(OUT / "summary_mean_sd.csv")

# paired record-bootstrap: seed-averaged probability, AUROC and accuracy at val thresholds
base = "cnn"; recs = np.array(sorted(df.record.unique())); rng = np.random.default_rng(0)
key = df.groupby(["model", "record", "minute"]).agg(p=("y_prob", "mean"), y=("y_true", "first"), t=("thr_val", "mean")).reset_index()
piv = {mod: g.set_index(["record", "minute"]) for mod, g in key.groupby("model")}
idx = {r: np.flatnonzero(piv[base].reset_index().record.values == r) for r in recs}
def stat(mod, ii):
    g = piv[mod].iloc[ii]; return roc_auc_score(g.y, g.p), ((g.p > g.t) == g.y).mean()
cmp = []
for mod in piv:
    if mod == base: continue
    assert piv[mod].index.equals(piv[base].index)
    d = []
    for _ in range(300):
        ii = np.concatenate([idx[r] for r in rng.choice(recs, len(recs))]); a, b = stat(mod, ii), stat(base, ii); d.append((a[0] - b[0], a[1] - b[1]))
    d = np.array(d); cmp.append(dict(model=mod, d_auroc_vs_cnn=float(np.mean(d[:, 0])), d_auroc_ci=np.percentile(d[:, 0], [2.5, 97.5]).round(3).tolist(),
                                   d_acc_vs_cnn=float(np.mean(d[:, 1])), d_acc_ci=np.percentile(d[:, 1], [2.5, 97.5]).round(3).tolist()))
pd.DataFrame(cmp).to_csv(OUT / "paired_vs_cnn.csv", index=False)
with open(OUT / "summary.md", "w") as f:
    f.write("# Experiment A/B -- clean protocol, LOSO(35), 3 seeds, pooled per-minute metrics (mean +/- SD over seeds)\n\n")
    f.write(agg.round(3).to_markdown() + "\n\n## Paired record-bootstrap vs CNN (seed-averaged probs, validation thresholds)\n\n" + pd.DataFrame(cmp).round(3).to_markdown(index=False) + "\n")
print(open(OUT / "summary.md").read())
