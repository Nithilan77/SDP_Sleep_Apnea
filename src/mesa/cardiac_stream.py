"""MESA cardiac stream: EKG -> R-peaks -> (RR, R-amplitude) on a uniform time grid, per subject.

Everything is in TIME units: RR in seconds, grid at GRID_HZ, epochs are the MESA 30 s label grid
(src/ingest/mesa.label_grid). Per-subject normalisation is label-free (median RR, median |amp|),
so it needs no train/test bookkeeping.

Cache (gitignored, holds subject ids): data/mesa/cache/cardiac/<mesaid>.npz
    rr, amp          float32 (n_grid,)   normalised RR / amplitude on the GRID_HZ grid
    n_beats          int32   (n_epochs,) valid beats per 30 s epoch
    resp_event, stage, apnea ...         label grid columns, one value per epoch

    python -m src.mesa.cardiac_stream           # build the cache for all usable subjects
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ecg.rpeaks import RR_MAX_S, RR_MIN_S  # noqa: E402
from src.ingest import mesa as M  # noqa: E402

log = logging.getLogger("mesa_cardiac")

CACHE_DIR = M.MESA_ROOT / "cache" / "cardiac"
GRID_HZ = 2.0
EPOCH_S = M.EPOCH_S
SLEEP_STAGES = ("N1", "N2", "N3", "R")
AUDIT_CSV = Path("results/mesa/audit/subjects.csv")  # holds mesaids -> local only, never committed


def usable_ids() -> list[str]:
    """The 220 usable subjects, using the audit's own funnel definition."""
    from src.ingest.mesa_audit import usable_funnel

    d = pd.read_csv(AUDIT_CSV, dtype={"mesaid": str})
    _, mask = usable_funnel(d)
    return d.loc[mask, "mesaid"].tolist()


def _detect(ecg: np.ndarray, fs: float):
    cleaned = nk.ecg_clean(ecg, sampling_rate=int(fs))
    _, info = nk.ecg_peaks(cleaned, sampling_rate=int(fs))
    pk = np.asarray(info["ECG_R_Peaks"])
    return pk / fs, cleaned[pk]


def build_subject(mesaid: str) -> Path:
    out = CACHE_DIR / f"{mesaid}.npz"
    psg = M.load_psg(mesaid, channels={"ecg": M.CHANNELS["ecg"]})
    ch = psg.channels["ecg"]
    t_pk, amp_pk = _detect(ch.x, ch.fs)
    grid = M.label_grid(M.parse_events(mesaid), psg.duration_s)

    rr = np.diff(t_pk)
    t_rr, amp = t_pk[1:], amp_pk[1:]
    ok = (rr >= RR_MIN_S) & (rr <= RR_MAX_S)          # implausible RR (missed/extra beat) -> dropped
    t_ok, rr_ok, amp_ok = t_rr[ok], rr[ok], amp[ok]

    n_ep = len(grid)
    n_beats = np.histogram(t_ok, bins=np.arange(n_ep + 1) * EPOCH_S)[0].astype(np.int32)

    tg = np.arange(0.0, n_ep * EPOCH_S, 1.0 / GRID_HZ)
    if len(t_ok) < 10:
        rr_g = np.full(len(tg), np.nan, np.float32); amp_g = rr_g.copy()
    else:
        rr_g = np.interp(tg, t_ok, rr_ok).astype(np.float32)
        a_med = np.median(np.abs(amp_ok)) or 1.0
        amp_g = np.interp(tg, t_ok, amp_ok / a_med).astype(np.float32)
        rr_g = rr_g / np.float32(np.median(rr_ok)) - 1.0   # relative RR deviation from subject median
    np.savez_compressed(out, rr=rr_g, amp=amp_g, n_beats=n_beats,
                        resp_event=grid.resp_event.to_numpy(np.int8),
                        apnea=grid.apnea.to_numpy(np.int8),
                        subtype=grid.subtype.to_numpy(str), stage=grid.stage.to_numpy(str),
                        median_rr_s=np.float32(np.median(rr_ok)) if len(rr_ok) else np.float32(np.nan),
                        pct_rr_dropped=np.float32(100 * (1 - ok.mean())) if len(ok) else np.float32(100))
    return out


def build_all(n_jobs: int = 10) -> None:
    from joblib import Parallel, delayed

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ids = [i for i in usable_ids() if not (CACHE_DIR / f"{i}.npz").exists()]
    log.info("building cardiac cache for %d subjects", len(ids))
    Parallel(n_jobs=n_jobs, verbose=5)(delayed(build_subject)(i) for i in ids)


def load_subject(mesaid: str) -> dict:
    z = np.load(CACHE_DIR / f"{mesaid}.npz")
    return {k: z[k] for k in z.files}


def epoch_windows(rr: np.ndarray, amp: np.ndarray, targets: np.ndarray, context: int) -> np.ndarray:
    """(n_targets, 2, (2*context+1)*EPOCH_S*GRID_HZ) windows centred on each target epoch.
    Edges are padded with the subject-median value (RR 0, amplitude 1 after normalising)."""
    spe = int(EPOCH_S * GRID_HZ)
    pad = context * spe
    x = np.stack([np.pad(rr, pad, mode="constant"), np.pad(amp, pad, mode="constant", constant_values=1.0)])
    idx = (targets[:, None] * spe) + np.arange((2 * context + 1) * spe)[None, :]
    return x[:, idx].transpose(1, 0, 2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_all()
