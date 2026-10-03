"""MESA Track B -- STEP 2/3: CANet, dual-stream cross-attention fusion of cardiac + respiratory effort.

Streams (all windows = target 30 s epoch +/- 2 neighbours = 150 s, per-subject label-free normalised):
  cardiac : RR + R-amplitude @ 2 Hz (300 samples)  -- EXACTLY the ECG-only baseline's input tensor
  resp    : Thor + Abdo belts @ 32 Hz (4800 samples)
  spo2    : SpO2 @ 1 Hz (150 samples)  -- 'ceiling' variant ONLY, not available on our wearable
Each stream -> multi-scale 1D-CNN (parallel kernels 3/7/15) -> 75 tokens of width 64 -> bidirectional
cross-modal attention (each stream's tokens query the other stream(s)' tokens) -> GAP+GMP -> MLP.

Same epochs, same grouped 5-fold split, same inner-validation / threshold discipline as
src/mesa/ecg_baseline.py (imported, not re-implemented, where possible); the test fold is never used
for any choice. Predictions are cached per variant so results/mesa/canet/head_to_head.* can be
rebuilt for the identical epochs.

    python -m src.mesa.canet --variant headline          # cardiac + resp, cross-attention  (the claim)
    python -m src.mesa.canet --variant ceiling_spo2      # + SpO2, NOT deployable
    python -m src.mesa.canet --variant resp_only | concat_fusion      # ablations
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.mesa import cardiac_stream as C  # noqa: E402
from src.mesa import ecg_baseline as B  # noqa: E402
from src.mesa import respiratory_stream as R  # noqa: E402

log = logging.getLogger("canet")
OUT = Path("results/mesa/canet")
CACHE = Path("data/mesa/cache/canet")  # per-epoch predictions carry mesaids -> gitignored
DEV = B.DEV
CTX = B.CFG["context"]
VARIANTS = {  # name -> (streams, fusion)
    "headline": (("cardiac", "resp"), "cross"),
    "ceiling_spo2": (("cardiac", "resp", "spo2"), "cross"),
    "resp_only": (("resp",), "concat"),
    "concat_fusion": (("cardiac", "resp"), "concat"),
}
D_MODEL, N_HEADS, N_TOKENS = 64, 4, 75
STREAM_SPEC = {  # in_channels, stem stride (0 = none), pooling per block -> every stream ends at 75 tokens
    "cardiac": dict(c_in=2, stem=0, pools=(2, 2)),
    "resp": dict(c_in=2, stem=4, pools=(4, 4)),
    "spo2": dict(c_in=1, stem=0, pools=(2, 1)),
}
TRAIN_CFG = {k: B.CFG[k] for k in ("max_epochs", "patience", "batch_size", "lr", "weight_decay", "dropout")}


# ----------------------------------------------------------------------------- model
class MSBlock(nn.Module):
    """Multi-scale conv block: parallel kernels {3,7,15}, concatenated, BN-ReLU, max-pool."""

    def __init__(self, c_in, c_branch, pool):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(c_in, c_branch, k, padding=k // 2) for k in (3, 7, 15))
        self.bn = nn.BatchNorm1d(3 * c_branch)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()

    def forward(self, x):
        return self.pool(torch.relu(self.bn(torch.cat([c(x) for c in self.convs], 1))))


class StreamEncoder(nn.Module):
    def __init__(self, c_in, stem, pools):
        super().__init__()
        layers, c = [], c_in
        if stem:
            layers += [nn.Conv1d(c, 16, 15, stride=stem, padding=7), nn.BatchNorm1d(16), nn.ReLU()]
            c = 16
        layers += [MSBlock(c, 16, pools[0]), MSBlock(48, 24, pools[1]), nn.Conv1d(72, D_MODEL, 1)]
        self.net = nn.Sequential(*layers)
        self.pos = nn.Parameter(torch.zeros(1, N_TOKENS, D_MODEL))

    def forward(self, x):                       # (B, c, L) -> (B, 75, d)
        z = self.net(x).transpose(1, 2)
        assert z.shape[1] == N_TOKENS, z.shape
        return z + self.pos


class CrossLayer(nn.Module):
    """Every stream's tokens attend to the concatenation of the OTHER streams' tokens.
    With two streams this is exactly bidirectional cross-modal attention (A<-B and B<-A)."""

    def __init__(self, names, dropout=0.1):
        super().__init__()
        self.names = names
        self.attn = nn.ModuleDict({n: nn.MultiheadAttention(D_MODEL, N_HEADS, dropout=dropout, batch_first=True) for n in names})
        self.ln_q = nn.ModuleDict({n: nn.LayerNorm(D_MODEL) for n in names})
        self.ln_kv = nn.ModuleDict({n: nn.LayerNorm(D_MODEL) for n in names})
        self.ln_f = nn.ModuleDict({n: nn.LayerNorm(D_MODEL) for n in names})
        self.ffn = nn.ModuleDict({n: nn.Sequential(nn.Linear(D_MODEL, 2 * D_MODEL), nn.GELU(), nn.Dropout(dropout),
                                                   nn.Linear(2 * D_MODEL, D_MODEL)) for n in names})

    def forward(self, t):
        out = {}
        for n in self.names:
            kv = self.ln_kv[n](torch.cat([t[m] for m in self.names if m != n], 1))
            q = self.ln_q[n](t[n])
            h = t[n] + self.attn[n](q, kv, kv, need_weights=False)[0]
            out[n] = h + self.ffn[n](self.ln_f[n](h))
        return out


class CANet(nn.Module):
    def __init__(self, streams, fusion="cross", dropout=0.3):
        super().__init__()
        self.streams, self.fusion = tuple(streams), fusion
        self.enc = nn.ModuleDict({s: StreamEncoder(**STREAM_SPEC[s]) for s in self.streams})
        self.cross = CrossLayer(self.streams) if fusion == "cross" and len(self.streams) > 1 else None
        self.head = nn.Sequential(nn.Linear(2 * D_MODEL * len(self.streams), 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, xs):
        t = {s: self.enc[s](xs[s]) for s in self.streams}
        if self.cross is not None:
            t = self.cross(t)
        z = torch.cat([torch.cat([t[s].mean(1), t[s].amax(1)], 1) for s in self.streams], 1)
        return self.head(z).squeeze(-1)


# ----------------------------------------------------------------------------- data on GPU
class GpuData:
    """All streams resident on the GPU; windows are gathered per batch by (subject, epoch) index."""

    def __init__(self, ids, X_card, sub, ep, streams):
        self.streams = streams
        self.card = torch.as_tensor(X_card, device=DEV)  # baseline tensor, reused unchanged
        self.sub, self.ep = sub, ep
        self.arr, self.start, self.win = {}, {}, {}
        for name, key, hz in (("resp", "belts", R.BELT_HZ), ("spo2", "spo2", R.SPO2_HZ)):
            if name not in streams:
                continue
            spe, pad = int(C.EPOCH_S * hz), int(C.EPOCH_S * hz) * CTX
            chunks, offs, o = [], [], 0
            for mid in ids:
                a = R.load_subject(mid)[key]
                a = a[None] if a.ndim == 1 else a
                a = np.pad(a, ((0, 0), (pad, pad)))
                chunks.append(a); offs.append(o); o += a.shape[1]
            self.arr[name] = torch.as_tensor(np.concatenate(chunks, 1), device=DEV)  # fp16 (c, T)
            self.start[name] = torch.as_tensor(np.array(offs)[sub] + ep * spe, device=DEV)
            self.win[name] = (2 * CTX + 1) * spe
            log.info("%s stream on GPU: %s fp16 (%.2f GB)", name, tuple(self.arr[name].shape), self.arr[name].numel() * 2 / 1e9)

    def batch(self, idx):                     # idx: LongTensor on DEV
        out = {}
        if "cardiac" in self.streams:
            out["cardiac"] = self.card[idx]
        for name in self.arr:
            g = self.start[name][idx][:, None] + torch.arange(self.win[name], device=DEV)[None]
            out[name] = self.arr[name][:, g].permute(1, 0, 2).float()
        return out


@torch.no_grad()
def predict(model, data, rows, bs=1024):
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        idx = torch.as_tensor(rows[i:i + bs], device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(model(data.batch(idx)).float()).cpu().numpy())
    return np.concatenate(out)


def train_fold(data, y, tr, va, streams, fusion, seed):
    torch.manual_seed(seed)
    model = CANet(streams, fusion, dropout=TRAIN_CFG["dropout"]).to(DEV)
    ytr = y[tr]
    pw = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], device=DEV)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN_CFG["lr"], weight_decay=TRAIN_CFG["weight_decay"])
    yg = torch.as_tensor(y, device=DEV)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for e in range(TRAIN_CFG["max_epochs"]):
        model.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), TRAIN_CFG["batch_size"]):
            b = torch.as_tensor(order[i:i + TRAIN_CFG["batch_size"]], device=DEV)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(data.batch(b))
            loss_fn(logits.float(), yg[b]).backward(); opt.step()
        ap = average_precision_score(y[va], predict(model, data, va))
        log.info("    ep %2d val AUPRC %.4f", e, ap)
        if ap > best:
            best, bad, best_state = ap, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= TRAIN_CFG["patience"]:
                break
    model.load_state_dict(best_state)
    return model, best, e + 1


# ----------------------------------------------------------------------------- head-to-head
def _subtypes(subjects, epochs):
    out = np.empty(len(subjects), dtype=object)
    for mid in np.unique(subjects):
        m = subjects == mid
        out[m] = C.load_subject(mid)["subtype"][epochs[m]]
    return out


def _load_preds(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def write_head_to_head(n_boot=200, seed=0):
    """CANet variants vs the ECG-only baseline on the IDENTICAL epochs (own per-fold thresholds)."""
    base = _load_preds(B.PRED_CACHE / "preds.npz")
    models = {"ECG-only baseline (CardiacCNN)": base}
    labels = {"headline": "CANet (ECG + effort)  [HEADLINE]", "concat_fusion": "ablation: ECG + effort, concat (no attention)",
              "resp_only": "ablation: effort only", "ceiling_spo2": "CEILING: CANet + SpO2 (not on our wearable)"}
    for v in VARIANTS:
        p = CACHE / v / "preds.npz"
        if p.exists():
            z = _load_preds(p)
            assert np.array_equal(z["subject"], base["subject"]) and np.array_equal(z["epoch"], base["epoch"]) and np.array_equal(z["y"], base["y"]), "epoch set differs from baseline"
            models[labels[v]] = z
    y, subj = base["y"].astype(int), base["subject"]
    st = _subtypes(subj, base["epoch"])
    rows = []
    for name, z in models.items():
        yhat = z["p"] >= z["thr"]
        r = dict(model=name, auroc=roc_auc_score(y, z["p"]), auprc=average_precision_score(y, z["p"]),
                 sens_event=yhat[y == 1].mean(), spec_normal=(~yhat[y == 0]).mean(),
                 f1=2 * (yhat & (y == 1)).sum() / (yhat.sum() + (y == 1).sum()))
        for s in ("obstructive", "central", "hypopnea"):
            r[f"sens_{s}"] = yhat[st == s].mean()
        rows.append(r)
    tab = pd.DataFrame(rows)
    # paired subject-level bootstrap of the delta vs baseline
    uniq = np.unique(subj); idx_of = {u: np.flatnonzero(subj == u) for u in uniq}
    rng = np.random.default_rng(seed); deltas = {n: [] for n in models if n != "ECG-only baseline (CardiacCNN)"}
    for _ in range(n_boot):
        ii = np.concatenate([idx_of[u] for u in rng.choice(uniq, len(uniq))])
        yb = y[ii]
        ab, pb = roc_auc_score(yb, base["p"][ii]), average_precision_score(yb, base["p"][ii])
        for n in deltas:
            deltas[n].append((roc_auc_score(yb, models[n]["p"][ii]) - ab, average_precision_score(yb, models[n]["p"][ii]) - pb))
    ci = {n: np.percentile(np.array(d), [2.5, 97.5], axis=0) for n, d in deltas.items()}
    tab["d_auroc"] = [np.nan if n not in ci else tab.set_index("model").loc[n, "auroc"] - tab.auroc[0] for n in tab.model]
    tab["d_auroc_ci95"] = ["" if n not in ci else f"[{ci[n][0, 0]:+.3f}, {ci[n][1, 0]:+.3f}]" for n in tab.model]
    tab["d_auprc"] = [np.nan if n not in ci else tab.set_index("model").loc[n, "auprc"] - tab.auprc[0] for n in tab.model]
    tab["d_auprc_ci95"] = ["" if n not in ci else f"[{ci[n][0, 1]:+.3f}, {ci[n][1, 1]:+.3f}]" for n in tab.model]
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "head_to_head.csv", index=False)
    md = ["# CANet vs ECG-only baseline -- identical epochs, subject-independent grouped 5-fold CV", "",
          f"n_epochs={len(y)}, n_subjects={len(uniq)}, event prevalence={y.mean():.3f}. Each model uses its own inner-val Youden "
          f"threshold per fold. Deltas are vs the baseline; CI = paired bootstrap over subjects ({n_boot} reps).", "",
          tab.to_markdown(index=False, floatfmt=".3f"), ""]
    (OUT / "head_to_head.md").write_text("\n".join(md))
    return tab


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, default="headline")
    ap.add_argument("--only_report", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.only_report:
        print(write_head_to_head().to_string()); return
    streams, fusion = VARIANTS[a.variant]
    out = OUT if a.variant == "headline" else OUT / a.variant
    out.mkdir(parents=True, exist_ok=True); (CACHE / a.variant).mkdir(parents=True, exist_ok=True)

    ids, Xc, y, sub, ep, st = B.load_dataset()          # same 220 subjects, same epoch mask as the baseline
    base = _load_preds(B.PRED_CACHE / "preds.npz")
    assert np.array_equal(base["epoch"], ep) and np.array_equal(base["y"], y) and np.array_equal(base["subject"], np.array(ids)[sub])
    n_sub = len(ids)
    perm = np.random.default_rng(B.CFG["split_seed"]).permutation(n_sub)       # identical folds to the baseline
    fold_of = np.empty(n_sub, int)
    for f, chunk in enumerate(np.array_split(perm, B.CFG["n_folds"])):
        fold_of[chunk] = f
    for f in range(B.CFG["n_folds"]):                    # folds provably identical: baseline thr is constant per fold
        thr_f = np.unique(base["thr"][np.isin(sub, np.flatnonzero(fold_of == f))])
        assert len(thr_f) == 1
    data = GpuData(ids, Xc, sub, ep, streams)
    log.info("variant %s | streams %s | fusion %s | X %d epochs", a.variant, streams, fusion, len(y))
    n_params = sum(p.numel() for p in CANet(streams, fusion).parameters()); log.info("parameters: %d", n_params)

    p_all = np.full(len(y), np.nan, np.float32); thr_all = np.full(len(y), np.nan); fold_rows = []
    for f in range(B.CFG["n_folds"]):
        tr_sub = np.flatnonzero(fold_of != f); te_sub = np.flatnonzero(fold_of == f)
        rng = np.random.default_rng(100 + f)             # identical inner split to the baseline
        va_sub = rng.choice(tr_sub, int(round(B.CFG["inner_val_frac"] * len(tr_sub))), replace=False)
        tr_sub = np.setdiff1d(tr_sub, va_sub)
        assert not (set(tr_sub) & set(te_sub)) and not (set(va_sub) & set(te_sub)) and not (set(tr_sub) & set(va_sub))
        tr, va, te = (np.flatnonzero(np.isin(sub, s)) for s in (tr_sub, va_sub, te_sub))
        log.info("fold %d: train/val/test epochs %d/%d/%d", f, len(tr), len(va), len(te))
        model, best_ap, n_ep = train_fold(data, y, tr, va, streams, fusion, seed=f)
        thr = B.youden_threshold(y[va], predict(model, data, va))
        p = predict(model, data, te); p_all[te] = p; thr_all[te] = thr
        r = B.binary_report(y[te], p, thr); r.update(fold=f, n_test_subjects=len(te_sub), val_auprc=float(best_ap), epochs_run=n_ep)
        fold_rows.append(r)
        log.info("  fold %d TEST: sens %.3f spec %.3f AUROC %.3f AUPRC %.3f F1 %.3f (thr %.3f)", f, r["sensitivity"], r["specificity"], r["auroc"], r["auprc"], r["f1"], thr)
        del model; torch.cuda.empty_cache()

    assert not np.isnan(p_all).any()
    fold_df = pd.DataFrame(fold_rows).set_index("fold")
    pooled = B.binary_report(y, p_all, thr_all)
    cols = ["sensitivity", "specificity", "precision", "f1", "balanced_acc", "auroc", "auprc"]
    summary = {"variant": a.variant, "pooled_all_test_epochs": pooled, "per_fold_mean": fold_df[cols].mean().to_dict(),
               "per_fold_std": fold_df[cols].std().to_dict(), "chance_auprc_equals_prevalence": float(y.mean())}
    ahi, pts = B.ahi_correlation(ids, sub, y, p_all, thr_all, st)
    fold_df.to_csv(out / "fold_metrics.csv")
    pd.DataFrame([[pooled["tn"], pooled["fp"]], [pooled["fn"], pooled["tp"]]], index=["true_normal", "true_event"],
                 columns=["pred_normal", "pred_event"]).to_csv(out / "confusion_matrix.csv")
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    (out / "ahi_correlation.json").write_text(json.dumps(ahi, indent=2))
    pts.rename_axis("idx").to_csv(out / "ahi_points_anon.csv")
    cfg = dict(TRAIN_CFG, variant=a.variant, streams=list(streams), fusion=fusion, n_params=n_params, d_model=D_MODEL, n_heads=N_HEADS,
               tokens=N_TOKENS, kernels=[3, 7, 15], context=CTX, n_subjects=n_sub, n_epochs=int(len(y)), split_seed=B.CFG["split_seed"],
               inner_val_frac=B.CFG["inner_val_frac"], loss=B.CFG["loss"], threshold=B.CFG["threshold"], amp="bf16 autocast",
               fold_sizes=[int((fold_of == f).sum()) for f in range(B.CFG["n_folds"])], torch=torch.__version__, device=DEV)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    np.savez_compressed(CACHE / a.variant / "preds.npz", subject=np.array(ids)[sub], epoch=ep, y=y, p=p_all, thr=thr_all)
    log.info("POOLED %s: sens %.3f spec %.3f AUROC %.3f AUPRC %.3f F1 %.3f | per-fold AUROC %.3f +/- %.3f", a.variant, pooled["sensitivity"],
             pooled["specificity"], pooled["auroc"], pooled["auprc"], pooled["f1"], summary["per_fold_mean"]["auroc"], summary["per_fold_std"]["auroc"])
    log.info("AHI corr (pred_rate vs a0h3 / a0h4): r=%.3f / %.3f", ahi["pred_rate_vs_ahi_a0h3"]["pearson"], ahi["pred_rate_vs_ahi_a0h4"]["pearson"])
    print(write_head_to_head().to_string())


if __name__ == "__main__":
    main()
