"""Phase 3b -- vanilla RNN and LSTM on the SAME task/protocol as the adopted 1D-CNN.

Task: per-minute apnea (A/N) from the 2-channel (RR-interval, R-peak amplitude) 60-step sequence
(results/phase3_cnn_v2/sequences.npz), the 16,949 Phase-2 minutes, 35 labelled Apnea-ECG records,
leave-one-subject-out (35 folds), per-fold normalisation from the training fold only, 30 epochs, Adam 1e-3, batch 128,
class-balanced BCE pos_weight = n_neg/n_pos, seed 0 -- identical to src/ecg/cnn_apnea_final.py (CNN: 0.812/0.756/0.846).
Only the architecture differs. Both recurrent nets use hidden=32, 1 layer, last hidden state -> Linear(32,16)-ReLU-Linear(16,1);
grad-norm clipping 1.0 (both; standard for recurrent nets). No tuning against test folds. One seed.

    python3 -m src.ecg.rnn_lstm [--arch rnn|lstm|both] [--final]
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ecg.cnn_apnea_final import load_or_build_sequences, restrict_to_phase2_minutes, per_recording_evaluation  # noqa: E402
from src.eval.metrics import binary_metrics, format_metrics  # noqa: E402
from src.eval.splits import leave_one_subject_out  # noqa: E402

OUT = Path("results/phase3_rnn_lstm")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH, LR, SEED, HIDDEN, CLIP = 30, 128, 1e-3, 0, 32, 1.0


class RecurrentApnea(nn.Module):
    """x: (batch, 2, 60) channels-first (same tensor as the CNN) -> logits (batch,)."""
    def __init__(self, arch: str, n_channels: int = 2, hidden: int = HIDDEN, pool: str = "last", bidir: bool = False):
        super().__init__()
        self.arch, self.pool = arch, pool
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM}[arch]
        self.rnn = rnn_cls(n_channels, hidden, num_layers=1, batch_first=True, bidirectional=bidir)
        self.head = nn.Sequential(nn.Linear(hidden * (2 if bidir else 1), 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, _ = self.rnn(x.transpose(1, 2))          # (b, 60, H*dirs)
        z = out[:, -1] if self.pool == "last" else out.mean(1)   # 'last' (primary) == final hidden state for 1-layer unidirectional
        return self.head(z).squeeze(-1)


def n_params(m): return int(sum(p.numel() for p in m.parameters()))


def train(arch, X, y, seed=SEED, epochs=EPOCHS, lr=LR, **kw):
    torch.manual_seed(seed)
    m = RecurrentApnea(arch, **kw).to(DEV)
    pw = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)], dtype=torch.float32, device=DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr); lf = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt = torch.tensor(X, dtype=torch.float32, device=DEV); yt = torch.tensor(y, dtype=torch.float32, device=DEV)
    rng = np.random.default_rng(seed); m.train()
    for _ in range(epochs):
        order = rng.permutation(len(Xt))
        for s in range(0, len(Xt), BATCH):
            i = torch.tensor(order[s:s + BATCH], device=DEV)
            opt.zero_grad(); lf(m(Xt[i]), yt[i]).backward()
            nn.utils.clip_grad_norm_(m.parameters(), CLIP); opt.step()
    return m


@torch.no_grad()
def predict(m, X):
    m.eval()
    return torch.sigmoid(m(torch.tensor(X, dtype=torch.float32, device=DEV))).cpu().numpy()


def run_loso(arch, ds, mask, **kw):
    recs = sorted(np.unique(ds["record"][mask])); seq = ds["sequence"][mask]
    lab = (ds["label"][mask] == "A").astype(np.float32); rc = ds["record"][mask]; mn = ds["minute"][mask]
    preds, t0 = [], time.time()
    for k, (tr, te) in enumerate(leave_one_subject_out(recs), 1):
        ti, ei = np.isin(rc, tr), rc == te
        mu = seq[ti].mean(axis=(0, 2), keepdims=True); sd = seq[ti].std(axis=(0, 2), keepdims=True)
        m = train(arch, (seq[ti] - mu) / sd, lab[ti], **kw)
        p = predict(m, (seq[ei] - mu) / sd)
        preds.append(pd.DataFrame(dict(record=rc[ei], minute=mn[ei], y_true=lab[ei].astype(int), y_pred=(p > .5).astype(int), y_prob=p)))
        print(f"[{arch}] fold {k:2d}/{len(recs)} {te} n={ei.sum()} {time.time()-t0:.0f}s", flush=True)
    return pd.concat(preds, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arch", default="both"); ap.add_argument("--final", action="store_true")
    a = ap.parse_args(); archs = ["rnn", "lstm"] if a.arch == "both" else [a.arch]
    ds = load_or_build_sequences(); mask = restrict_to_phase2_minutes(ds)
    OUT.mkdir(parents=True, exist_ok=True); summary = {}
    for arch in archs:
        pr = run_loso(arch, ds, mask); pr.to_csv(OUT / f"{arch}_per_minute_predictions.csv", index=False)
        m = binary_metrics(pr["y_true"], pr["y_pred"]); rec = per_recording_evaluation(pr)
        print(f"\n=== {arch.upper()} LOSO ===\n{format_metrics(m)}\nper-recording {int(rec.correct.sum())}/{len(rec)}")
        rec.to_csv(OUT / f"{arch}_per_recording.csv", index=False)
        summary[arch] = dict(metrics={k: float(v) for k, v in m.items() if np.isscalar(v)}, per_recording_correct=int(rec.correct.sum()),
                             n_records=len(rec), params=n_params(RecurrentApnea(arch)), hidden=HIDDEN, epochs=EPOCHS, seed=SEED)
        json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
        if a.final:   # deployable: all 35 records, no held-out set (same status as the CNN artifact)
            seq, lab = ds["sequence"][mask], (ds["label"][mask] == "A").astype(np.float32)
            mu = seq.mean(axis=(0, 2), keepdims=True); sd = seq.std(axis=(0, 2), keepdims=True)
            mdl = train(arch, (seq - mu) / sd, lab); d = OUT / f"deployable_{arch}"; d.mkdir(exist_ok=True)
            torch.save(mdl.cpu().state_dict(), d / "model_state_dict.pt")
            json.dump(dict(mean=mu.flatten().tolist(), std=sd.flatten().tolist(), arch=arch, hidden=HIDDEN), open(d / "norm_stats.json", "w"))


if __name__ == "__main__":
    main()
