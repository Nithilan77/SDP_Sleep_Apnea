"""MESA respiratory-effort stream (Thor + Abdo belts @32 Hz) and the SpO2 'ceiling' stream (1 Hz).

Label-free per-subject preprocessing (needs no train/test bookkeeping):
  belts: zero-phase band-pass 0.03-1.5 Hz (drift out, breathing in), divide by the night's robust
         scale (1.4826 * MAD), clip +/-CLIP. Amplitude is deliberately NOT normalised per epoch, so
         effort cessation (low amplitude) stays visible.
  SpO2:  invalid (<50 or NaN) forward-filled, (x - 95) / 5, clipped to [-8, 2].
Arrays are trimmed to n_epochs * 30 s so they line up with the cardiac cache / label grid.

Cache (gitignored, holds mesaids): data/mesa/cache/resp/<mesaid>.npz  (float16)

    python -m src.mesa.respiratory_stream
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingest import mesa as M  # noqa: E402
from src.mesa import cardiac_stream as C  # noqa: E402

log = logging.getLogger("mesa_resp")

CACHE_DIR = M.MESA_ROOT / "cache" / "resp"
BELT_HZ = 32.0
SPO2_HZ = 1.0
BAND = (0.03, 1.5)
CLIP = 6.0


def _belt(ch: M.Channel, n_samples: int) -> np.ndarray:
    assert abs(ch.fs - BELT_HZ) < 1e-6, f"unexpected belt fs {ch.fs}"
    x = np.nan_to_num(ch.x[:n_samples].astype(np.float64))
    if len(x) < n_samples:
        x = np.pad(x, (0, n_samples - len(x)))
    x = sosfiltfilt(butter(2, BAND, btype="band", fs=BELT_HZ, output="sos"), x)
    scale = 1.4826 * np.median(np.abs(x - np.median(x)))
    return np.clip(x / scale if scale > 1e-9 else np.zeros_like(x), -CLIP, CLIP).astype(np.float16)


def _spo2(ch: M.Channel, n_samples: int) -> np.ndarray:
    assert abs(ch.fs - SPO2_HZ) < 1e-6, f"unexpected spo2 fs {ch.fs}"
    x = ch.x[:n_samples].astype(np.float64)
    x = np.pad(x, (0, max(0, n_samples - len(x))), constant_values=np.nan)
    bad = ~np.isfinite(x) | (x < 50)
    idx = np.maximum.accumulate(np.where(~bad, np.arange(len(x)), -1))
    x = np.where(idx >= 0, x[np.maximum(idx, 0)], 95.0)   # leading invalid -> 95
    return np.clip((x - 95.0) / 5.0, -8, 2).astype(np.float16)


def build_subject(mesaid: str) -> Path:
    out = CACHE_DIR / f"{mesaid}.npz"
    psg = M.load_psg(mesaid, channels={k: M.CHANNELS[k] for k in ("thor", "abdo", "spo2")})
    n_ep = int(psg.duration_s // C.EPOCH_S)
    nb, ns = int(n_ep * C.EPOCH_S * BELT_HZ), int(n_ep * C.EPOCH_S * SPO2_HZ)
    belts = np.stack([_belt(psg.channels["thor"], nb), _belt(psg.channels["abdo"], nb)])
    np.savez_compressed(out, belts=belts, spo2=_spo2(psg.channels["spo2"], ns), n_epochs=np.int32(n_ep))
    return out


def build_all(n_jobs: int = 10) -> None:
    from joblib import Parallel, delayed

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ids = [i for i in C.usable_ids() if not (CACHE_DIR / f"{i}.npz").exists()]
    log.info("building respiratory cache for %d subjects", len(ids))
    Parallel(n_jobs=n_jobs, verbose=5)(delayed(build_subject)(i) for i in ids)


def load_subject(mesaid: str) -> dict:
    z = np.load(CACHE_DIR / f"{mesaid}.npz")
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_all()
