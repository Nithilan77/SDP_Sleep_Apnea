"""Train the three FINAL MESA models used for the in-house transfer study (MESA only -- MESA is the
training distribution; in-house recordings are never used for any fitting).

The CV runs (src/mesa/ecg_baseline.py, src/mesa/canet.py) did not keep weights, so here each model is
refit with the identical recipe on all 220 usable subjects, holding out a random 15% of SUBJECTS as an
inner validation set for early stopping and the Youden threshold (same discipline as the CV folds):
    ecg_only   : CardiacCNN (the MESA ECG-only baseline)
    resp_only  : CANet effort-only ablation (Thor+Abdo)
    headline   : CANet cardiac + resp, bidirectional cross-attention
Weights go to data/mesa/cache/final_models/ (gitignored, MESA-derived).

    python -m src.mesa.train_final
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.mesa import canet as N  # noqa: E402
from src.mesa import ecg_baseline as B  # noqa: E402

log = logging.getLogger("train_final")
OUT = Path("data/mesa/cache/final_models")
SEED = 999


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    ids, Xc, y, sub, ep, st = B.load_dataset()
    n_sub = len(ids)
    rng = np.random.default_rng(SEED)
    va_sub = rng.choice(n_sub, int(round(B.CFG["inner_val_frac"] * n_sub)), replace=False)
    tr_sub = np.setdiff1d(np.arange(n_sub), va_sub)
    tr, va = (np.flatnonzero(np.isin(sub, s)) for s in (tr_sub, va_sub))
    log.info("final fit: %d train / %d val subjects, %d / %d epochs", len(tr_sub), len(va_sub), len(tr), len(va))

    log.info("== ecg_only (CardiacCNN)")
    m, ap, n_ep = B.train_fold(Xc[tr], y[tr], Xc[va], y[va], seed=SEED)
    thr = B.youden_threshold(y[va], B.predict(m, Xc[va]))
    torch.save(dict(state=m.state_dict(), thr=thr, val_auprc=ap, epochs=n_ep, kind="ecg_only"), OUT / "ecg_only.pt")
    log.info("   val AUPRC %.4f thr %.3f", ap, thr)

    for name, (streams, fusion) in (("resp_only", N.VARIANTS["resp_only"]), ("headline", N.VARIANTS["headline"])):
        log.info("== %s %s", name, streams)
        data = N.GpuData(ids, Xc, sub, ep, streams)
        m, ap, n_ep = N.train_fold(data, y, tr, va, streams, fusion, seed=SEED)
        thr = N.B.youden_threshold(y[va], N.predict(m, data, va))
        torch.save(dict(state=m.state_dict(), thr=thr, val_auprc=ap, epochs=n_ep, kind=name, streams=streams, fusion=fusion),
                   OUT / f"{name}.pt")
        log.info("   val AUPRC %.4f thr %.3f", ap, thr)
        del data, m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
