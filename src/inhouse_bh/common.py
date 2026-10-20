"""Breath-hold (Track B) signal pipeline: SensorTile CSV/TXT -> 3-axis breathing-band trace @32 Hz -> windows -> features / tensors.

Window = WIN_S seconds of the 0.08-0.6 Hz band-passed 3-axis accelerometer (g), scaled by a PER-RECORDING robust scale (median 10 s RMS over the whole recording:
uses no labels, so it is also applicable at inference time).  Labels come from results/inhouse_bh/annotations.csv; windows that touch an `uncertain` interval
or lie outside any labelled interval are NOT created.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import kurtosis
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ingest.sensortile import load_sensortile

HZ, WIN_S, STRIDE_S = 32, 10.0, 2.0
W = int(WIN_S * HZ)
ANN = Path("results/inhouse_bh/annotations.csv")
PATHS = {"S1_ECGTILE": Path("data/recordings/breath_hold/user_MukuBreathHold_ECG_07_10_2026.csv")}
FEATURES = ["rms_mag", "rms_x", "rms_y", "rms_z", "env_p10", "env_p50", "env_p90", "band_frac", "peak_frac", "spec_entropy", "peak_hz", "zcr", "kurt", "rng"]


def trace32(path: Path, band=(0.08, 0.6)):
    """-> (n,3) float32 band-passed accelerometer in g at 32 Hz, duration, fs (empirical timestamps)."""
    r = load_sensortile(path); t = (r.timestamp_us - r.timestamp_us[0]) / 1e6; a = r.accel_mg / 1000.0
    sos = butter(3, band, btype="band", fs=r.fs_measured, output="sos"); xb = sosfiltfilt(sos, a - np.median(a, 0), axis=0)
    tg = np.arange(0, t[-1], 1 / HZ); x = np.stack([np.interp(tg, t, xb[:, i]) for i in range(3)], 1).astype(np.float32)
    return x, float(r.duration_s), float(r.fs_measured)


def rec_scale(x: np.ndarray) -> float:
    mag = np.linalg.norm(x, axis=1); n = len(mag) // W
    rms = np.sqrt((mag[:n * W].reshape(n, W) ** 2).mean(1)); return float(np.median(rms) + 1e-9)


def rms_series(x: np.ndarray) -> np.ndarray:
    """RMS of the magnitude over every W-sample window starting each second (index k = window [k s, k s + WIN_S))."""
    mag2 = (x ** 2).sum(1); c = np.concatenate([[0], np.cumsum(mag2)]); idx = np.arange(0, len(x) - W + 1, HZ)
    return np.sqrt((c[idx + W] - c[idx]) / W)


def local_scales(x: np.ndarray, i0: np.ndarray, before: float = 75.0, after: float = 85.0) -> np.ndarray:
    """Per-window scale = median RMS of the surrounding recording context (-75 s ... +85 s around the window start). Uses signal only (no labels) -> valid at inference."""
    r = rms_series(x); out = []
    for i in i0:
        t = i / HZ; a, b = int(max(0, t - before)), int(min(len(r), t + after)); out.append(np.median(r[a:b]) + 1e-9)
    return np.array(out)


def features(win: np.ndarray) -> np.ndarray:
    """win (3,W) scaled units -> FEATURES vector."""
    mag = np.linalg.norm(win, axis=0); f, P = welch(mag, HZ, nperseg=min(W, 256)); tot = P.sum() + 1e-12; b = (f >= .1) & (f <= .6)
    env = np.abs(mag); pk = P[b].argmax() if b.any() else 0; p = P[b] / (P[b].sum() + 1e-12)
    return np.array([np.sqrt((mag ** 2).mean()), *np.sqrt((win ** 2).mean(1)), *np.percentile(env, [10, 50, 90]), P[b].sum() / tot, P[b].max() / tot,
                     float(-(p * np.log(p + 1e-12)).sum()), f[b][pk], float((np.diff(np.sign(mag - mag.mean())) != 0).mean()), float(kurtosis(mag)), float(np.ptp(mag))], np.float32)


def labelled_windows(rec_id: str, x: np.ndarray, stride_s: float = STRIDE_S):
    """Windows fully inside ONE labelled interval (baseline_reference | voluntary_breath_hold | recovery) and not overlapping any `uncertain` interval."""
    ann = pd.read_csv(ANN); ann = ann[(ann.recording_id == rec_id) & (ann.time_base == "recording")]
    sc = rec_scale(x); out = []; r_ = rms_series(x)
    unc = ann[ann.label == "uncertain"][["start_s", "end_s"]].to_numpy()
    for _, r in ann[ann.label != "uncertain"].iterrows():
        t0 = r.start_s
        while t0 + WIN_S <= r.end_s + 1e-9:
            if not any(t0 < ue and t0 + WIN_S > us for us, ue in unc) and t0 >= 0 and int((t0 + WIN_S) * HZ) <= len(x):
                i = int(round(t0 * HZ)); a_, b_ = int(max(0, t0 - 75)), int(min(len(r_), t0 + 85)); out.append(dict(recording_id=rec_id, event_no=int(r.event_no), label=r.label, confidence=r.confidence, t0=t0, i0=i, scale=float(np.median(r_[a_:b_]) + 1e-9)))
            t0 += stride_s
    return pd.DataFrame(out), sc


def window_array(x: np.ndarray, i0: np.ndarray, scale) -> np.ndarray:
    sc = np.broadcast_to(np.asarray(scale, dtype=np.float32), (len(i0),))
    return np.stack([x[i:i + W].T / s_ for i, s_ in zip(i0, sc)]).astype(np.float32)
