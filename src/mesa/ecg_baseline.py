"""MESA Track B -- STEP 1: ECG-only baseline (the floor CANet must beat).

Cardiac stream only (RR interval + R-peak amplitude) -> 1D-CNN -> P(respiratory event) per 30 s
epoch (apnea OR hypopnea vs normal). Subject-independent grouped 5-fold CV over the 220 usable
subjects; weighted BCE; threshold and early stopping chosen on an inner validation split of the
TRAINING subjects only (the test fold is never used for any choice).

Design choices (fixed, not tuned on test folds):
  * Evaluation/training epochs = scored SLEEP epochs (N1/N2/N3/REM) with >= MIN_BEATS valid beats.
    AHI is a per-sleep-hour index and sleep staging is the sister track's job, so wake is out of
    scope here (caveat: a deployed system would need its own sleep/wake gate).
  * Context window = target epoch +/- CONTEXT neighbours (CONTEXT=2 -> 150 s, 300 samples at 2 Hz).
  * Per-subject label-free normalisation (see cardiac_stream.py).

    python -m src.mesa.ecg_baseline            # from project root; needs the cardiac cache
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.mesa import cardiac_stream as C  # noqa: E402

log = logging.getLogger("ecg_baseline")

OUT = Path("results/mesa/ecg_baseline")
PRED_CACHE = Path("data/mesa/cache/ecg_baseline")  # per-epoch predictions carry mesaids -> gitignored
CFG = dict(context=2, grid_hz=C.GRID_HZ, min_beats=10, n_folds=5, split_seed=0, inner_val_frac=0.15,
           max_epochs=25, patience=5, batch_size=512, lr=1e-3, weight_decay=1e-4, dropout=0.3,
           loss="BCEWithLogits, pos_weight=n_neg/n_pos (train fold)", threshold="Youden J on inner-val subjects",
           target="resp_event (apnea|hypopnea) vs normal, sleep epochs only")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- data
def load_dataset():
    ids = C.usable_ids()
    Xs, ys, sub, ep, stats = [], [], [], [], []
    for k, mid in enumerate(ids):
        d = C.load_subject(mid)
        sleep = np.isin(d["stage"], C.SLEEP_STAGES)
        keep = sleep & (d["n_beats"] >= CFG["min_beats"])
        tgt = np.flatnonzero(keep)
        stats.append((int(sleep.sum()), int(keep.sum())))
        if not len(tgt):
            continue
        w = C.epoch_windows(np.nan_to_num(d["rr"]), np.nan_to_num(d["amp"], nan=1.0), tgt, CFG["context"])
        Xs.append(w.astype(np.float32)); ys.append(d["resp_event"][tgt]); sub.append(np.full(len(tgt), k)); ep.append(tgt)
    X = np.concatenate(Xs)
    X[:, 0] = np.clip(X[:, 0], -1, 1)
    X[:, 1] = np.clip(X[:, 1] - 1, -3, 3)
    st = np.array(stats)
    log.info("subjects %d | sleep epochs %d -> kept %d (%.2f%% dropped for < %d beats)", len(ids), st[:, 0].sum(),
             st[:, 1].sum(), 100 * (1 - st[:, 1].sum() / st[:, 0].sum()), CFG["min_beats"])
    return ids, X, np.concatenate(ys).astype(np.float32), np.concatenate(sub), np.concatenate(ep), st


# ----------------------------------------------------------------------------- model
class CardiacCNN(nn.Module):
    """RRCNN scaled up for a 150 s two-channel window: 3 conv blocks, avg+max pooling over time."""

    def __init__(self, n_ch: int = 2, dropout: float = 0.3):
        super().__init__()

        def blk(i, o, k, pool):
            return [nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU()] + ([nn.MaxPool1d(2)] if pool else [])

        self.features = nn.Sequential(*blk(n_ch, 32, 7, True), *blk(32, 64, 7, True), *blk(64, 64, 5, False))
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        z = self.features(x)
        return self.head(torch.cat([z.mean(-1), z.amax(-1)], 1)).squeeze(-1)


@torch.no_grad()
def predict(model, X, bs=4096):
    model.eval()
    out = [torch.sigmoid(model(torch.as_tensor(X[i:i + bs], device=DEV))).cpu().numpy() for i in range(0, len(X), bs)]
    return np.concatenate(out)


def train_fold(Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed)
    model = CardiacCNN(dropout=CFG["dropout"]).to(DEV)
    pw = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], device=DEV)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    Xg, yg = torch.as_tensor(Xtr, device=DEV), torch.as_tensor(ytr, device=DEV)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for e in range(CFG["max_epochs"]):
        model.train()
        order = torch.as_tensor(rng.permutation(len(Xg)), device=DEV)
        for i in range(0, len(order), CFG["batch_size"]):
            b = order[i:i + CFG["batch_size"]]
            opt.zero_grad(); loss_fn(model(Xg[b]), yg[b]).backward(); opt.step()
        ap = average_precision_score(yva, predict(model, Xva))
        log.info("    ep %2d val AUPRC %.4f", e, ap)
        if ap > best:
            best, bad, best_state = ap, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= CFG["patience"]:
                break
    model.load_state_dict(best_state)
    return model, best, e + 1


# ----------------------------------------------------------------------------- metrics
def binary_report(y, p, thr):
    yhat = (p >= thr).astype(int); y = y.astype(int)
    tp = int(((y == 1) & (yhat == 1)).sum()); tn = int(((y == 0) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum()); fn = int(((y == 1) & (yhat == 0)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return dict(n=len(y), prevalence=float(y.mean()), threshold=float(np.mean(thr)), tp=tp, tn=tn, fp=fp, fn=fn,
                sensitivity=sens, specificity=spec, precision=prec,
                f1=2 * prec * sens / max(prec + sens, 1e-12), balanced_acc=(sens + spec) / 2,
                auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)))


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def ahi_correlation(ids, sub, y, p, thr_per_epoch, st):
    """Subject-level: predicted event-epoch rate per sleep hour vs NSRR AHI. NO SpO2 is used, so this
    cannot reproduce desaturation-based AHI definitions; read it as a screening-correlation only."""
    audit = pd.read_csv(C.AUDIT_CSV, dtype={"mesaid": str}).set_index("mesaid")
    rows = []
    for k, mid in enumerate(ids):
        m = sub == k
        if not m.any():
            continue
        hrs = audit.loc[mid, "sleep_h"]
        rows.append(dict(pred_rate=float((p[m] >= thr_per_epoch[m]).sum() / hrs),
                         label_rate=float(y[m].sum() / hrs), prob_rate=float(p[m].sum() / hrs),
                         ahi_a0h3=audit.loc[mid, "ahi_a0h3"], ahi_a0h4=audit.loc[mid, "ahi_a0h4"],
                         ahi_xml_all=audit.loc[mid, "ahi_xml_all"]))
    df = pd.DataFrame(rows)
    res = {"n_subjects": len(df), "caveat": "no SpO2 input; rate = predicted event-epochs per scored sleep hour"}
    for tgt in ("ahi_a0h3", "ahi_a0h4", "ahi_xml_all"):
        for src in ("pred_rate", "prob_rate", "label_rate"):
            res[f"{src}_vs_{tgt}"] = dict(pearson=float(pearsonr(df[src], df[tgt])[0]), spearman=float(spearmanr(df[src], df[tgt])[0]))
    for cut in (5, 15, 30):
        yy = (df.ahi_a0h4 >= cut).astype(int)
        if 0 < yy.sum() < len(yy):
            res[f"auroc_pred_rate_detects_a0h4_ge{cut}"] = float(roc_auc_score(yy, df.pred_rate))
    return res, df


# ----------------------------------------------------------------------------- main
def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(parents=True, exist_ok=True); PRED_CACHE.mkdir(parents=True, exist_ok=True)
    ids, X, y, sub, ep, st = load_dataset()
    log.info("X %s | prevalence %.4f | device %s", X.shape, y.mean(), DEV)

    n_sub = len(ids)
    perm = np.random.default_rng(CFG["split_seed"]).permutation(n_sub)
    fold_of = np.empty(n_sub, int)
    for f, chunk in enumerate(np.array_split(perm, CFG["n_folds"])):
        fold_of[chunk] = f

    p_all = np.full(len(y), np.nan, np.float32); thr_all = np.full(len(y), np.nan)
    fold_rows = []
    for f in range(CFG["n_folds"]):
        tr_sub = np.flatnonzero(fold_of != f); te_sub = np.flatnonzero(fold_of == f)
        rng = np.random.default_rng(100 + f)
        va_sub = rng.choice(tr_sub, int(round(CFG["inner_val_frac"] * len(tr_sub))), replace=False)
        tr_sub = np.setdiff1d(tr_sub, va_sub)
        assert not (set(tr_sub) & set(te_sub)) and not (set(va_sub) & set(te_sub)) and not (set(tr_sub) & set(va_sub))
        tr, va, te = (np.isin(sub, s) for s in (tr_sub, va_sub, te_sub))
        log.info("fold %d: subjects train/val/test %d/%d/%d | epochs %d/%d/%d | prev %.3f/%.3f/%.3f", f, len(tr_sub),
                 len(va_sub), len(te_sub), tr.sum(), va.sum(), te.sum(), y[tr].mean(), y[va].mean(), y[te].mean())
        model, best_ap, n_ep = train_fold(X[tr], y[tr], X[va], y[va], seed=f)
        thr = youden_threshold(y[va], predict(model, X[va]))
        p = predict(model, X[te]); p_all[te] = p; thr_all[te] = thr
        r = binary_report(y[te], p, thr); r.update(fold=f, n_test_subjects=len(te_sub), val_auprc=float(best_ap), epochs_run=n_ep)
        fold_rows.append(r)
        log.info("  fold %d TEST: sens %.3f spec %.3f AUROC %.3f AUPRC %.3f F1 %.3f (thr %.3f)", f, r["sensitivity"],
                 r["specificity"], r["auroc"], r["auprc"], r["f1"], thr)

    assert not np.isnan(p_all).any()
    fold_df = pd.DataFrame(fold_rows).set_index("fold")
    pooled = binary_report(y, p_all, thr_all)  # per-fold thresholds applied to each fold's own test epochs
    cols = ["sensitivity", "specificity", "precision", "f1", "balanced_acc", "auroc", "auprc"]
    summary = {"pooled_all_test_epochs": pooled,
               "per_fold_mean": fold_df[cols].mean().to_dict(), "per_fold_std": fold_df[cols].std().to_dict(),
               "chance_auprc_equals_prevalence": float(y.mean())}
    ahi, pts = ahi_correlation(ids, sub, y, p_all, thr_all, st)

    fold_df.to_csv(OUT / "fold_metrics.csv")
    pd.DataFrame([[pooled["tn"], pooled["fp"]], [pooled["fn"], pooled["tp"]]], index=["true_normal", "true_event"],
                 columns=["pred_normal", "pred_event"]).to_csv(OUT / "confusion_matrix.csv")
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    (OUT / "ahi_correlation.json").write_text(json.dumps(ahi, indent=2))
    pts.rename_axis("idx").to_csv(OUT / "ahi_points_anon.csv")  # one row per subject, no ids
    cfg = dict(CFG, n_subjects=n_sub, n_epochs=int(len(y)), input_shape=list(X.shape[1:]), device=DEV,
               fold_sizes=[int((fold_of == f).sum()) for f in range(CFG["n_folds"])], model=str(CardiacCNN()),
               torch=torch.__version__)
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))
    np.savez_compressed(PRED_CACHE / "preds.npz", subject=np.array(ids)[sub], epoch=ep, y=y, p=p_all, thr=thr_all)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        for f in range(CFG["n_folds"]):
            m = np.isin(sub, np.flatnonzero(fold_of == f)); fpr, tpr, _ = roc_curve(y[m], p_all[m]); ax[0].plot(fpr, tpr, lw=1, label=f"fold {f}")
        ax[0].plot([0, 1], [0, 1], "k:", lw=.8); ax[0].set(xlabel="1 - specificity", ylabel="sensitivity", title=f"ROC (pooled AUROC {pooled['auroc']:.3f})"); ax[0].legend(fontsize=7)
        for i, (t, nm) in enumerate((("ahi_a0h3", "NSRR a0h3"), ("ahi_a0h4", "NSRR a0h4")), 1):
            ax[i].scatter(pts[t], pts.pred_rate, s=10, alpha=.6); ax[i].set(xlabel=nm, ylabel="predicted event-epochs / sleep h",
                title=f"r={ahi[f'pred_rate_vs_{t}']['pearson']:.2f}, rho={ahi[f'pred_rate_vs_{t}']['spearman']:.2f} (no SpO2)")
        fig.tight_layout(); fig.savefig(OUT / "ecg_baseline_overview.png", dpi=120)
    except Exception as e:  # plotting must never kill the run
        log.warning("plot failed: %s", e)

    log.info("POOLED: sens %.3f spec %.3f AUROC %.3f AUPRC %.3f F1 %.3f | per-fold AUROC %.3f +/- %.3f", pooled["sensitivity"],
             pooled["specificity"], pooled["auroc"], pooled["auprc"], pooled["f1"], summary["per_fold_mean"]["auroc"], summary["per_fold_std"]["auroc"])
    log.info("AHI corr (pred_rate vs a0h3 / a0h4): r=%.3f / %.3f", ahi["pred_rate_vs_ahi_a0h3"]["pearson"], ahi["pred_rate_vs_ahi_a0h4"]["pearson"])


if __name__ == "__main__":
    main()
