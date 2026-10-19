"""Experiment A -- controlled multi-seed comparison of CNN / LSTM / GRU / TCN on the Phase-3 task.

Task/data identical to src/ecg/cnn_apnea_final.py: per-minute apnea, 2-channel (RR, R-amp) x 60 steps, 16,949 minutes, 35 records, LOSO.
What differs from the historical runs (which are left untouched): a *clean* protocol with no test-fold selection.
  * per outer fold: 5 of the 34 training records are held out (seeded) as an inner validation set
  * normalisation + class weight from the remaining training records only
  * early stopping = epoch with best validation AUROC (checkpoint restore); threshold = max validation balanced accuracy
  * the test record is used once, for prediction. Both the validation-chosen threshold and the fixed 0.5 threshold are reported.
  * hyperparameters below were fixed a priori (not tuned); same optimiser/batch/epochs for every model.

    python3 -m src.ecg.expA_multiseed --model cnn --seeds 0 1 2
"""
from __future__ import annotations
import argparse, json, sys, time, copy
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ecg.cnn_apnea_final import load_or_build_sequences, restrict_to_phase2_minutes
from src.ecg.cnn_model import RRCNN
from src.eval.splits import leave_one_subject_out

OUT = Path("results/expA_multiseed"); DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH, LR, CLIP, N_VAL = 40, 128, 1e-3, 1.0, 5


class Rec(nn.Module):
    def __init__(self, kind, hidden=64):
        super().__init__()
        self.rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[kind](2, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x): return self.head(self.rnn(x.transpose(1, 2))[0].mean(1)).squeeze(-1)


class Block(nn.Module):
    def __init__(self, cin, cout, d):
        super().__init__(); self.pad = 2 * d
        self.c1 = nn.Conv1d(cin, cout, 3, dilation=d); self.c2 = nn.Conv1d(cout, cout, 3, dilation=d)
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
    def forward(self, x):
        h = torch.relu(self.c1(nn.functional.pad(x, (self.pad, 0))))
        h = torch.relu(self.c2(nn.functional.pad(h, (self.pad, 0))))
        return torch.relu(h + self.res(x))


class TCN(nn.Module):
    def __init__(self, ch=24):
        super().__init__()
        self.net = nn.Sequential(Block(2, ch, 1), Block(ch, ch, 2), Block(ch, ch, 4), Block(ch, ch, 8))
        self.head = nn.Sequential(nn.Linear(ch, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x): return self.head(self.net(x).mean(-1)).squeeze(-1)


def make(kind):
    return {"cnn": lambda: RRCNN(60, 2), "lstm": lambda: Rec("lstm"), "gru": lambda: Rec("gru"), "tcn": TCN}[kind]()


def probs(m, X):
    m.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(X), 2048): out.append(torch.sigmoid(m(X[s:s + 2048])).cpu().numpy())
    return np.concatenate(out)


def bal_thr(y, p):
    best, bt = -1, .5
    for t in np.linspace(.05, .95, 91):
        pr = p > t; se = (pr & (y == 1)).sum() / max((y == 1).sum(), 1); sp = (~pr & (y == 0)).sum() / max((y == 0).sum(), 1)
        if se + sp > best: best, bt = se + sp, t
    return float(bt)


def fit(kind, Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed); m = make(kind).to(DEV)
    pw = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], dtype=torch.float32, device=DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR); lf = nn.BCEWithLogitsLoss(pos_weight=pw)
    rng = np.random.default_rng(seed); best, bs, ep_best = -1, None, 0
    for ep in range(EPOCHS):
        m.train(); order = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), BATCH):
            i = torch.tensor(order[s:s + BATCH], device=DEV)
            opt.zero_grad(); lf(m(Xtr[i]), ytr[i]).backward(); nn.utils.clip_grad_norm_(m.parameters(), CLIP); opt.step()
        a = roc_auc_score(yva.cpu().numpy(), probs(m, Xva))
        if a > best: best, bs, ep_best = a, copy.deepcopy(m.state_dict()), ep + 1
    m.load_state_dict(bs); return m, best, ep_best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args(); OUT.mkdir(exist_ok=True)
    ds = load_or_build_sequences(); ds = dict(np.load("results/phase3_cnn_v2/sequences.npz", allow_pickle=True))
    mask = restrict_to_phase2_minutes(ds)
    seq, lab, rc, mn = ds["sequence"][mask], (ds["label"][mask] == "A").astype(np.float32), ds["record"][mask], ds["minute"][mask]
    assert seq.shape[1] == 2
    recs = sorted(np.unique(rc)); nparams = sum(p.numel() for p in make(a.model).parameters())
    rows = []
    for seed in a.seeds:
        t0 = time.time()
        for k, (tr, te) in enumerate(leave_one_subject_out(recs), 1):
            r = np.random.default_rng(1000 * seed + k); val = list(r.choice(tr, N_VAL, replace=False)); fit_r = [x for x in tr if x not in val]
            fi, vi, ei = np.isin(rc, fit_r), np.isin(rc, val), rc == te
            mu = seq[fi].mean(axis=(0, 2), keepdims=True); sd = seq[fi].std(axis=(0, 2), keepdims=True)
            T = lambda idx: torch.tensor((seq[idx] - mu) / sd, dtype=torch.float32, device=DEV)
            m, vauc, ep = fit(a.model, T(fi), torch.tensor(lab[fi], device=DEV), T(vi), torch.tensor(lab[vi], device=DEV), seed)
            thr = bal_thr(lab[vi], probs(m, T(vi))); p = probs(m, T(ei))
            rows.append(pd.DataFrame(dict(model=a.model, seed=seed, record=rc[ei], minute=mn[ei], y_true=lab[ei].astype(int), y_prob=p, thr_val=thr, best_epoch=ep)))
            print(f"[{a.model} s{seed}] fold {k}/35 {te} valAUC={vauc:.3f} ep={ep} thr={thr:.2f} {time.time()-t0:.0f}s", flush=True)
        pd.concat(rows, ignore_index=True).to_csv(OUT / f"{a.model}_predictions.csv", index=False)
    json.dump(dict(model=a.model, params=int(nparams), epochs=EPOCHS, batch=BATCH, lr=LR, n_val_records=N_VAL, seeds=a.seeds), open(OUT / f"{a.model}_config.json", "w"))


if __name__ == "__main__":
    main()
