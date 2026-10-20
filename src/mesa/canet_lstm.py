"""MESA Track B -- STEP 4: CNN-LSTM (temporal model) with ECG + SpO2 (+ optional effort belts).

Drop-in sibling of src/mesa/canet.py. Same 220 subjects, same sleep-epoch mask, same grouped 5-fold
split, same inner-val early stopping / Youden threshold, same metrics -> the pooled predictions line up
row-for-row with the ECG-only baseline and CANet, so results/mesa/canet_lstm/ is directly comparable.

What is new vs CANet (which classifies one 150 s window at a time):
  * every 30 s epoch is encoded by a small multi-scale 1D-CNN per stream (cardiac / spo2 / resp)
  * a BIDIRECTIONAL LSTM runs over L consecutive epochs (default L=20 = 10 min) and emits one logit per
    epoch -> the model sees the whole apnea -> desaturation -> recovery sequence, incl. the 20-30 s SpO2 lag
  * the SpO2 encoder is a separate module; its embedding is randomly zeroed during training
    (--spo2-drop, default 0.1) so the model still works if the oximeter drops out; the test report also
    gives results with SpO2 masked.
  * all signals live on the GPU as flat fp16 tensors; chunks are gathered by index (no DataLoader).

Data: needs the existing caches only (python -m src.mesa.cardiac_stream, python -m src.mesa.respiratory_stream).

    python -m src.mesa.canet_lstm --variant ecg_lstm              # ECG only + LSTM
    python -m src.mesa.canet_lstm --variant ecg_spo2_lstm         # ECG + SpO2 + LSTM   <- your target
    python -m src.mesa.canet_lstm --variant ecg_spo2_lstm --L 1   # ablation: same net, NO temporal context
    python -m src.mesa.canet_lstm --variant ecg_resp_spo2_lstm    # everything (ceiling)
    python -m src.mesa.canet_lstm --variant ecg_spo2_lstm --final # fit on all subjects, save for transfer
    python -m src.mesa.canet_lstm --report                        # comparison table incl. paired bootstrap
    python -m src.mesa.canet_lstm --variant ecg_spo2_lstm --smoke 12   # 12-subject, 2-fold, 1-epoch dry run

Finished folds are cached, so after a dropped session just re-run the same command (use --fresh to redo).
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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.mesa import canet as N  # noqa: E402  (MSBlock, label maps, prediction loader)
from src.mesa import cardiac_stream as C  # noqa: E402
from src.mesa import ecg_baseline as B  # noqa: E402
from src.mesa import respiratory_stream as R  # noqa: E402

log = logging.getLogger("canet_lstm")
OUT = Path("results/mesa/canet_lstm")
CACHE = Path("data/mesa/cache/canet_lstm")          # per-epoch predictions + weights carry mesaids -> gitignored
FINAL_DIR = Path("data/mesa/cache/final_models")
DEV = B.DEV

VARIANTS = {
    "ecg_lstm": ("cardiac",),
    "ecg_spo2_lstm": ("cardiac", "spo2"),
    "ecg_resp_lstm": ("cardiac", "resp"),
    "ecg_resp_spo2_lstm": ("cardiac", "resp", "spo2"),
}
SPE = {"cardiac": int(C.EPOCH_S * C.GRID_HZ), "resp": int(C.EPOCH_S * R.BELT_HZ), "spo2": int(C.EPOCH_S * R.SPO2_HZ)}  # 60 / 960 / 30
SPEC = {  # c_in, stem stride, pooling per block -> every stream ends at 15 tokens per 30 s epoch
    "cardiac": dict(c_in=2, stem=0, pools=(2, 2)),
    "resp": dict(c_in=2, stem=4, pools=(4, 4)),
    "spo2": dict(c_in=1, stem=0, pools=(2, 1)),
}
D_EMB, D_FUSE, HID = 32, 128, 64
HP = dict(L=20, batch_chunks=32, max_epochs=40, patience=6, lr=1e-3, weight_decay=1e-4, dropout=0.3, spo2_drop=0.1, clip=1.0)


# ----------------------------------------------------------------------------- model
class EpochEncoder(nn.Module):
    """One 30 s epoch of one stream -> vector (2*D_EMB): multi-scale CNN, then mean+max pooling over time."""

    def __init__(self, c_in, stem, pools, d=D_EMB):
        super().__init__()
        layers, c = [], c_in
        if stem:
            layers += [nn.Conv1d(c, 16, 15, stride=stem, padding=7), nn.BatchNorm1d(16), nn.ReLU()]
            c = 16
        layers += [N.MSBlock(c, 16, pools[0]), N.MSBlock(48, 24, pools[1]), nn.Conv1d(72, d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):                                   # (N, c, T)
        z = self.net(x)
        return torch.cat([z.mean(-1), z.amax(-1)], 1)


class CNNLSTM(nn.Module):
    def __init__(self, streams, dropout=0.3, spo2_drop=0.1):
        super().__init__()
        self.streams, self.spo2_drop = tuple(streams), spo2_drop
        self.enc = nn.ModuleDict({s: EpochEncoder(**SPEC[s]) for s in self.streams})   # separate -> freezable / loadable
        self.fuse = nn.Sequential(nn.Linear(2 * D_EMB * len(self.streams), D_FUSE), nn.LayerNorm(D_FUSE), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(D_FUSE, HID, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(2 * HID + D_FUSE, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, xs, mask_spo2=False):                 # xs[s]: (B, L, c, T) -> logits (B, L)
        embs = []
        for s in self.streams:
            x = xs[s]
            b, l = x.shape[:2]
            e = self.enc[s](x.flatten(0, 1)).view(b, l, -1)
            if s == "spo2":
                if mask_spo2:
                    e = torch.zeros_like(e)
                elif self.training and self.spo2_drop > 0:      # modality dropout: whole chunk loses SpO2
                    e = e * (torch.rand(b, 1, 1, device=e.device) >= self.spo2_drop).to(e.dtype)
            embs.append(e)
        f = self.fuse(torch.cat(embs, -1))
        h, _ = self.lstm(f)
        return self.head(torch.cat([h, f], -1)).squeeze(-1)


# ----------------------------------------------------------------------------- data (flat tensors on the GPU)
class SeqData:
    """All subjects concatenated per stream; subject boundaries are epoch-aligned, so a chunk of L epochs
    starting at global epoch g is the contiguous slice [g*spe, (g+L)*spe). Chunks never cross subjects."""

    def __init__(self, ids, streams, L):
        self.streams, self.L, self.ids = tuple(streams), L, list(ids)
        need_r = any(s in streams for s in ("resp", "spo2"))
        arrs = {s: [] for s in streams}
        ys, keeps, subs, n_ep = [], [], [], []
        for k, mid in enumerate(ids):
            d = C.load_subject(mid)
            n = len(d["stage"])
            r = R.load_subject(mid) if need_r else None
            if r is not None:
                n = min(n, int(r["n_epochs"]))
            if n < L:
                raise ValueError(f"subject index {k} has only {n} epochs < L={L}; lower --L")
            sleep = np.isin(d["stage"][:n], C.SLEEP_STAGES)
            keeps.append(sleep & (d["n_beats"][:n] >= B.CFG["min_beats"]))   # SAME mask as ecg_baseline.load_dataset
            ys.append(d["resp_event"][:n].astype(np.float32)); subs.append(np.full(n, k)); n_ep.append(n)
            if "cardiac" in streams:
                rr = np.clip(np.nan_to_num(d["rr"][:n * SPE["cardiac"]]), -1, 1)
                amp = np.clip(np.nan_to_num(d["amp"][:n * SPE["cardiac"]], nan=1.0) - 1, -3, 3)
                arrs["cardiac"].append(np.stack([rr, amp]))
            if "resp" in streams:
                arrs["resp"].append(r["belts"][:, :n * SPE["resp"]])
            if "spo2" in streams:
                arrs["spo2"].append(r["spo2"][None, :n * SPE["spo2"]])
            for s in streams:
                assert arrs[s][-1].shape[1] == n * SPE[s], (s, k, arrs[s][-1].shape, n)
        self.arr = {s: torch.as_tensor(np.concatenate(a, 1).astype(np.float16), device=DEV) for s, a in arrs.items()}
        for s, a in self.arr.items():
            log.info("%s stream on %s: %s fp16 (%.2f GB)", s, DEV, tuple(a.shape), a.numel() * 2 / 1e9)
        self.n = np.array(n_ep)
        self.eoff = np.concatenate([[0], np.cumsum(self.n)])
        self.E = int(self.eoff[-1])
        self.sub_np, self.keep_np, self.y_np = np.concatenate(subs), np.concatenate(keeps), np.concatenate(ys)
        self.y, self.keep = torch.as_tensor(self.y_np, device=DEV), torch.as_tensor(self.keep_np, device=DEV)
        self._kc = np.concatenate([[0], np.cumsum(self.keep_np)])
        self._ar = torch.arange(L, device=DEV)

    def batch(self, g):                                     # g: LongTensor (B,) of global start epochs
        xs = {}
        for s in self.streams:
            spe = SPE[s]
            idx = g[:, None] * spe + torch.arange(self.L * spe, device=DEV)[None]
            a = self.arr[s][:, idx]                         # (c, B, L*spe)
            xs[s] = a.reshape(a.shape[0], g.numel(), self.L, spe).permute(1, 2, 0, 3).float()
        e = g[:, None] + self._ar[None]
        return xs, self.y[e], self.keep[e]

    def rows_for(self, subs):                               # kept (scored) epochs of these subjects, ascending global index
        return np.flatnonzero(np.isin(self.sub_np, subs) & self.keep_np)

    def train_starts(self, subs, stride, offset):
        out = [self.eoff[k] + np.arange(offset % stride, self.n[k] - self.L + 1, stride) for k in subs]
        g = np.concatenate(out)
        return g[(self._kc[g + self.L] - self._kc[g]) > 0]   # skip chunks with no scored epoch

    def eval_starts(self, subs):
        stride, out = max(1, self.L // 2), []
        for k in subs:
            s = np.arange(0, self.n[k] - self.L + 1, stride)
            if s[-1] != self.n[k] - self.L:
                s = np.append(s, self.n[k] - self.L)
            out.append(self.eoff[k] + s)
        return np.concatenate(out)


def amp_ctx(use_amp=True):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda" and use_amp))


@torch.no_grad()
def predict_rows(model, data, subs, mask_spo2=False, bs=64, use_amp=True):
    """Overlapping chunks (stride L/2), logits averaged with a triangular weight so chunk centres count most.
    Returns (rows, prob) for the scored epochs of `subs`."""
    model.eval()
    L = data.L
    w = (1.05 - (2 * torch.arange(L, device=DEV) / max(L - 1, 1) - 1).abs()) if L > 1 else torch.ones(1, device=DEV)
    acc, wsum = torch.zeros(data.E, device=DEV), torch.zeros(data.E, device=DEV)
    starts = torch.as_tensor(data.eval_starts(subs), device=DEV)
    for i in range(0, len(starts), bs):
        g = starts[i:i + bs]
        xs, _, _ = data.batch(g)
        with amp_ctx(use_amp):
            lg = model(xs, mask_spo2=mask_spo2).float()
        e = (g[:, None] + data._ar[None]).flatten()
        acc.index_add_(0, e, (lg * w).flatten()); wsum.index_add_(0, e, w.expand_as(lg).flatten())
    rows = data.rows_for(subs)
    return rows, torch.sigmoid(acc[rows] / wsum[rows].clamp_min(1e-9)).cpu().numpy()


def train_model(data, tr_sub, va_sub, streams, seed, hp, use_amp=True):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = CNNLSTM(streams, hp["dropout"], hp["spo2_drop"]).to(DEV)
    tr_rows, va_rows = data.rows_for(tr_sub), data.rows_for(va_sub)
    ytr, yva = data.y_np[tr_rows], data.y_np[va_rows]
    pw = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], device=DEV)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    stride = max(1, data.L // 2)
    best, best_state, bad = -1.0, None, 0
    for e in range(hp["max_epochs"]):
        model.train()
        order = rng.permutation(data.train_starts(tr_sub, stride, int(rng.integers(stride))))
        for i in range(0, len(order), hp["batch_chunks"]):
            xs, yc, kc = data.batch(torch.as_tensor(order[i:i + hp["batch_chunks"]], device=DEV))
            with amp_ctx(use_amp):
                lg = model(xs)
            loss = (loss_fn(lg.float(), yc) * kc).sum() / kc.sum().clamp_min(1)     # loss only on scored sleep epochs
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp["clip"]); opt.step()
        ap = average_precision_score(yva, predict_rows(model, data, va_sub, use_amp=use_amp)[1])
        log.info("    ep %2d val AUPRC %.4f", e, ap)
        if ap > best:
            best, bad, best_state = ap, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    return model, best, e + 1


# ----------------------------------------------------------------------------- report vs baseline / CANet
def report(n_boot=200, seed=0):
    base = N._load_preds(B.PRED_CACHE / "preds.npz")
    BASE = "ECG-only baseline (CardiacCNN)"
    models = {BASE: base}
    for v, lab in (("headline", "CANet ECG+effort"), ("ceiling_spo2", "CANet ECG+effort+SpO2 (ceiling)")):
        p = N.CACHE / v / "preds.npz"
        if p.exists():
            models[lab] = N._load_preds(p)
    if CACHE.exists():
        for d in sorted(CACHE.iterdir()):
            if (d / "preds.npz").exists():
                models["CNN-LSTM " + d.name] = N._load_preds(d / "preds.npz")
    y, subj = base["y"].astype(int), base["subject"]
    rows = []
    for name, z in models.items():
        assert np.array_equal(z["subject"], base["subject"]) and np.array_equal(z["epoch"], base["epoch"]) and np.array_equal(z["y"], base["y"]), f"{name}: epoch set differs"
        yhat = z["p"] >= z["thr"]
        rows.append(dict(model=name, auroc=roc_auc_score(y, z["p"]), auprc=average_precision_score(y, z["p"]), sens=yhat[y == 1].mean(),
                         spec=(~yhat[y == 0]).mean(), f1=2 * (yhat & (y == 1)).sum() / (yhat.sum() + (y == 1).sum())))
    tab = pd.DataFrame(rows)
    uniq = np.unique(subj); idx_of = {u: np.flatnonzero(subj == u) for u in uniq}
    rng = np.random.default_rng(seed); dl = {n: [] for n in models if n != BASE}
    for _ in range(n_boot):
        ii = np.concatenate([idx_of[u] for u in rng.choice(uniq, len(uniq))]); yb = y[ii]
        ab, pb = roc_auc_score(yb, base["p"][ii]), average_precision_score(yb, base["p"][ii])
        for n in dl:
            dl[n].append((roc_auc_score(yb, models[n]["p"][ii]) - ab, average_precision_score(yb, models[n]["p"][ii]) - pb))
    ci = {n: np.percentile(np.array(v), [2.5, 97.5], axis=0) for n, v in dl.items()}
    tab["d_auroc_vs_baseline"] = [f"[{ci[n][0, 0]:+.3f}, {ci[n][1, 0]:+.3f}]" if n in ci else "" for n in tab.model]
    tab["d_auprc_vs_baseline"] = [f"[{ci[n][0, 1]:+.3f}, {ci[n][1, 1]:+.3f}]" if n in ci else "" for n in tab.model]
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "head_to_head.csv", index=False)
    (OUT / "head_to_head.md").write_text(f"# CNN-LSTM vs baselines -- identical epochs, grouped 5-fold CV, paired subject bootstrap ({n_boot} reps)\n\n"
                                         + tab.to_markdown(index=False, floatfmt=".3f") + "\n")
    return tab


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, default="ecg_spo2_lstm")
    ap.add_argument("--L", type=int, default=HP["L"], help="epochs per LSTM sequence (1 = no temporal context)")
    ap.add_argument("--spo2-drop", type=float, default=HP["spo2_drop"])
    ap.add_argument("--max-epochs", type=int, default=HP["max_epochs"])
    ap.add_argument("--batch-chunks", type=int, default=HP["batch_chunks"])
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore cached folds")
    ap.add_argument("--final", action="store_true", help="fit on all subjects (15%% held out for early stopping) and save weights")
    ap.add_argument("--smoke", type=int, default=0, help="N subjects, 2 folds, 1 epoch: dry run, writes nothing under results/")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.report:
        print(report().to_string()); return

    streams = VARIANTS[a.variant]
    hp = dict(HP, L=a.L, spo2_drop=a.spo2_drop, max_epochs=1 if a.smoke else a.max_epochs, batch_chunks=a.batch_chunks)
    name = a.variant if a.L == HP["L"] else f"{a.variant}_L{a.L}"
    use_amp = not a.no_amp
    out, cache = (OUT / name, CACHE / name) if not a.smoke else (Path("/tmp/canet_lstm_smoke/out"), Path("/tmp/canet_lstm_smoke/cache"))
    out.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)

    ids = C.usable_ids()
    if a.smoke:
        ids = ids[:a.smoke]
    data = SeqData(ids, streams, a.L)
    rows = np.flatnonzero(data.keep_np); sub = data.sub_np[rows]; ep = rows - data.eoff[sub]; y = data.y_np[rows]
    if not a.smoke:                                         # identical epoch set to baseline / CANet, or stop now
        base = N._load_preds(B.PRED_CACHE / "preds.npz")
        assert np.array_equal(base["subject"], np.array(ids)[sub]) and np.array_equal(base["epoch"], ep) and np.array_equal(base["y"], y), \
            "scored-epoch set differs from the ECG-only baseline"
    n_sub, n_folds = len(ids), (2 if a.smoke else B.CFG["n_folds"])
    n_params = sum(p.numel() for p in CNNLSTM(streams).parameters())
    log.info("variant %s | streams %s | L=%d | %d subjects | %d scored epochs | params %d | device %s", name, streams, a.L, n_sub, len(y), n_params, DEV)

    if a.final:
        rng = np.random.default_rng(999)
        va_sub = rng.choice(n_sub, int(round(B.CFG["inner_val_frac"] * n_sub)), replace=False)
        tr_sub = np.setdiff1d(np.arange(n_sub), va_sub)
        model, best_ap, n_run = train_model(data, tr_sub, va_sub, streams, 999, hp, use_amp)
        vr, vp = predict_rows(model, data, va_sub, use_amp=use_amp)
        thr = B.youden_threshold(data.y_np[vr], vp)
        FINAL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state=model.state_dict(), thr=thr, val_auprc=best_ap, epochs=n_run, kind=name, streams=streams, L=a.L, hp=hp,
                        norm=dict(cardiac="RR/median-1 clip[-1,1]; amp/median|amp|-1 clip[-3,3]; grid 2 Hz",
                                  resp="band 0.03-1.5 Hz, /(1.4826 MAD), clip +-6; 32 Hz",
                                  spo2="(x-95)/5 clip[-8,2]; 1 Hz; invalid(<50/NaN) forward-filled")),
                   FINAL_DIR / f"{name}.pt")
        log.info("final %s: val AUPRC %.4f thr %.3f -> %s", name, best_ap, thr, FINAL_DIR / f"{name}.pt"); return

    perm = np.random.default_rng(B.CFG["split_seed"]).permutation(n_sub)
    fold_of = np.empty(n_sub, int)
    for f, chunk in enumerate(np.array_split(perm, n_folds)):
        fold_of[chunk] = f
    p_all = np.full(len(y), np.nan, np.float32); pm_all = p_all.copy(); thr_all = np.full(len(y), np.nan); fold_rows = []
    for f in range(n_folds):
        te_sub = np.flatnonzero(fold_of == f); tr_sub = np.flatnonzero(fold_of != f)
        rng = np.random.default_rng(100 + f)                # identical inner split to baseline / CANet
        va_sub = rng.choice(tr_sub, int(round(B.CFG["inner_val_frac"] * len(tr_sub))), replace=False)
        tr_sub = np.setdiff1d(tr_sub, va_sub)
        assert not (set(tr_sub) & set(te_sub)) and not (set(va_sub) & set(te_sub)) and not (set(tr_sub) & set(va_sub))
        te_pos = np.searchsorted(rows, data.rows_for(te_sub))
        fp = cache / f"fold{f}.npz"
        if fp.exists() and not a.fresh:
            z = np.load(fp); p, pm, thr = z["p"], z["p_masked"], float(z["thr"])
            best_ap, n_run = float(z["val_auprc"]), int(z["epochs_run"]); log.info("fold %d: loaded cached result", f)
        else:
            log.info("fold %d: train/val/test subjects %d/%d/%d", f, len(tr_sub), len(va_sub), len(te_sub))
            model, best_ap, n_run = train_model(data, tr_sub, va_sub, streams, f, hp, use_amp)
            vr, vp = predict_rows(model, data, va_sub, use_amp=use_amp)
            thr = B.youden_threshold(data.y_np[vr], vp)
            _, p = predict_rows(model, data, te_sub, use_amp=use_amp)
            pm = predict_rows(model, data, te_sub, mask_spo2=True, use_amp=use_amp)[1] if "spo2" in streams else np.full_like(p, np.nan)
            np.savez_compressed(fp, p=p, p_masked=pm, thr=thr, val_auprc=best_ap, epochs_run=n_run)
            torch.save(model.state_dict(), cache / f"fold{f}.pt")
            del model; torch.cuda.empty_cache()
        p_all[te_pos], pm_all[te_pos], thr_all[te_pos] = p, pm, thr
        r = B.binary_report(y[te_pos], p, thr); r.update(fold=f, n_test_subjects=len(te_sub), val_auprc=float(best_ap), epochs_run=n_run)
        fold_rows.append(r)
        log.info("  fold %d TEST: sens %.3f spec %.3f AUROC %.3f AUPRC %.3f F1 %.3f (thr %.3f)", f, r["sensitivity"], r["specificity"], r["auroc"], r["auprc"], r["f1"], thr)

    assert not np.isnan(p_all).any()
    fold_df = pd.DataFrame(fold_rows).set_index("fold")
    pooled = B.binary_report(y, p_all, thr_all)
    cols = ["sensitivity", "specificity", "precision", "f1", "balanced_acc", "auroc", "auprc"]
    summary = {"variant": name, "pooled_all_test_epochs": pooled, "per_fold_mean": fold_df[cols].mean().to_dict(),
               "per_fold_std": fold_df[cols].std().to_dict(), "chance_auprc_equals_prevalence": float(y.mean())}
    if "spo2" in streams:
        summary["spo2_masked_at_test"] = dict(auroc=float(roc_auc_score(y, pm_all)), auprc=float(average_precision_score(y, pm_all)))
    fold_df.to_csv(out / "fold_metrics.csv")
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    (out / "config.json").write_text(json.dumps(dict(hp, variant=name, streams=list(streams), n_params=n_params, d_emb=D_EMB, d_fuse=D_FUSE, lstm_hidden=HID,
                                                     n_subjects=n_sub, n_epochs=int(len(y)), split_seed=B.CFG["split_seed"], inner_val_frac=B.CFG["inner_val_frac"],
                                                     loss="BCEWithLogits pos_weight=n_neg/n_pos, masked to scored sleep epochs", threshold=B.CFG["threshold"],
                                                     torch=torch.__version__, device=DEV), indent=2))
    np.savez_compressed(cache / "preds.npz", subject=np.array(ids)[sub], epoch=ep, y=y, p=p_all, thr=thr_all)
    log.info("POOLED %s: sens %.3f spec %.3f AUROC %.3f AUPRC %.3f F1 %.3f | per-fold AUROC %.3f +/- %.3f", name, pooled["sensitivity"], pooled["specificity"],
             pooled["auroc"], pooled["auprc"], pooled["f1"], summary["per_fold_mean"]["auroc"], summary["per_fold_std"]["auroc"])
    if "spo2_masked_at_test" in summary:
        log.info("  with SpO2 masked at test: AUROC %.3f AUPRC %.3f", *summary["spo2_masked_at_test"].values())
    try:
        ahi, pts = B.ahi_correlation(ids, sub, y, p_all, thr_all, None)
        (out / "ahi_correlation.json").write_text(json.dumps(ahi, indent=2)); pts.rename_axis("idx").to_csv(out / "ahi_points_anon.csv")
        log.info("AHI corr (pred_rate vs a0h3 / a0h4): r=%.3f / %.3f", ahi["pred_rate_vs_ahi_a0h3"]["pearson"], ahi["pred_rate_vs_ahi_a0h4"]["pearson"])
    except Exception as e:  # predictions are already saved; never lose a long run to the AHI summary
        log.warning("AHI correlation skipped: %s", e)
    if not a.smoke:
        print(report().to_string())


if __name__ == "__main__":
    main()
