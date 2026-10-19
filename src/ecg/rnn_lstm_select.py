"""Config selection for the recurrent models on a FIXED 7-record validation split (never the LOSO test protocol itself).

Train on the other 28 records, score balanced accuracy on VAL. Grid: pooling {last, mean} x bidirectional {no, yes} x
epochs {30, 80} x hidden {32, 64}, per architecture. The winner per architecture is then re-run through the full 35-fold LOSO
(src.ecg.rnn_lstm_v2). CAVEAT (stated in the report): the 7 VAL records are also LOSO test folds, so the v2 LOSO number is
mildly optimistic; the as-specified primary run (src.ecg.rnn_lstm, no selection) is the unbiased comparison."""
import itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ecg.rnn_lstm import *  # noqa

VAL = ["a05", "a10", "a15", "b02", "c03", "c06", "c09"]
ds = load_or_build_sequences(); mask = restrict_to_phase2_minutes(ds)
seq = ds["sequence"][mask]; lab = (ds["label"][mask] == "A").astype(np.float32); rc = ds["record"][mask]
vi = np.isin(rc, VAL); ti = ~vi
mu = seq[ti].mean(axis=(0, 2), keepdims=True); sd = seq[ti].std(axis=(0, 2), keepdims=True)
rows = []
for arch in ("rnn", "lstm"):
    for pool, bd, ep, h in itertools.product(("last", "mean"), (False, True), (30, 80), (32, 64)):
        m = train(arch, (seq[ti] - mu) / sd, lab[ti], epochs=ep, hidden=h, pool=pool, bidir=bd)
        p = predict(m, (seq[vi] - mu) / sd) >= .5; y = lab[vi] == 1
        bal = 0.5 * (p[y].mean() + (~p[~y]).mean()); acc = (p == y).mean()
        rows.append(dict(arch=arch, pool=pool, bidir=bd, epochs=ep, hidden=h, val_bal_acc=float(bal), val_acc=float(acc)))
        print(rows[-1], flush=True)
best = {a: max((r for r in rows if r["arch"] == a), key=lambda r: r["val_bal_acc"]) for a in ("rnn", "lstm")}
Path("results/phase3_rnn_lstm").mkdir(exist_ok=True, parents=True)
json.dump(dict(val_records=VAL, grid=rows, best=best), open("results/phase3_rnn_lstm/selection.json", "w"), indent=1)
print("BEST", best)
