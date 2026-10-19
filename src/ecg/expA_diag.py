"""Diagnostic for the CNN gap between the historical 0.812 and the Exp-A clean protocol (0.753 @0.5). Not model selection: it only decomposes WHY.
  H0 : historical recipe exactly (src.ecg.cnn_apnea_final: train_cnn, 34 training records, 30 epochs, last epoch, thr 0.5), seeds 0,1,2  -> reproduces 0.812 for seed 0
  H1 : same, but 29 training records (the 5 Exp-A validation records removed, same seeded draw), 30 epochs, last epoch, thr 0.5
All outputs to results/expA_multiseed/diag_*.csv. Test records are never used for any choice."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ecg.cnn_apnea_final import restrict_to_phase2_minutes
from src.ecg.cnn_model import train_cnn, predict_cnn
from src.eval.splits import leave_one_subject_out
torch.set_num_threads(4)
ds = dict(np.load("results/phase3_cnn_v2/sequences.npz", allow_pickle=True)); mask = restrict_to_phase2_minutes(ds)
seq, lab, rc, mn = ds["sequence"][mask], (ds["label"][mask] == "A").astype(np.float32), ds["record"][mask], ds["minute"][mask]
recs = sorted(np.unique(rc)); rows = []
for seed in (0, 1, 2):
    t0 = time.time()
    for k, (tr, te) in enumerate(leave_one_subject_out(recs), 1):
        r = np.random.default_rng(1000 * seed + k); val = list(r.choice(tr, 5, replace=False)); fit29 = [x for x in tr if x not in val]
        for name, trr in (("H0_34rec", tr), ("H1_29rec", fit29)):
            ti, ei = np.isin(rc, trr), rc == te
            mu = seq[ti].mean(axis=(0, 2), keepdims=True); sd = seq[ti].std(axis=(0, 2), keepdims=True)
            m = train_cnn((seq[ti] - mu) / sd, lab[ti], seq_len=60, epochs=30, seed=seed)
            p = predict_cnn(m, (seq[ei] - mu) / sd)
            rows.append(pd.DataFrame(dict(proto=name, seed=seed, record=rc[ei], minute=mn[ei], y_true=lab[ei].astype(int), y_prob=p)))
        print(f"s{seed} fold {k}/35 {te} {time.time()-t0:.0f}s", flush=True)
    pd.concat(rows).to_csv("results/expA_multiseed/diag_cnn_historical_recipe.csv", index=False)
