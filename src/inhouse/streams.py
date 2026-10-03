"""Build CANet's two input streams from an in-house SensorTile recording (QVAR-ECG + IMU).

VALIDATION ONLY: nothing here is ever used to fit a model; healthy 19-20 y/o, no apnea, no labels.

cardiac : QVAR -> src/ecg/rpeaks.detect_rpeaks -> (RR s, R amplitude) -> the SAME grid / per-night
          normalisation as the MESA cardiac stream (src/mesa/cardiac_stream: 2 Hz, RR relative to the
          night median, amplitude / median|amp|). Per-night normalisation makes the earlier 108x
          amplitude-scale gap irrelevant *by construction* for this stream.
effort  : accelerometer -> breathing trace -> 32 Hz -> the SAME belt preprocessing as MESA
          (src/mesa/respiratory_stream: band-pass 0.03-1.5 Hz, / (1.4826*MAD), clip +/-6).
          Two sources are built:
            'mag' : src/effort/envelope.py's orientation-invariant accel-magnitude, band-passed
                    0.08-0.6 Hz (the project's Phase-4 breathing trace)  -- PRIMARY, as specified
            'pca' : first principal axis of the band-passed 3-axis accel (diagnostic alternative:
                    magnitude discards the gravity-direction change that breathing produces)
          The single accelerometer trace is fed to BOTH belt channels (Thor = Abdo).
Time alignment is by the empirical timestamp (never nominal 240 Hz).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ecg.rpeaks import RR_MAX_S, RR_MIN_S, detect_rpeaks  # noqa: E402
from src.effort import envelope as E  # noqa: E402
from src.ingest.sensortile import SensorTileRecord, load_sensortile  # noqa: E402
from src.mesa import cardiac_stream as C  # noqa: E402
from src.mesa import respiratory_stream as R  # noqa: E402

REC_DIR = Path("data/recordings/inhouse_ecg")  # gitignored; filenames contain names -> never written to results
MIN_DURATION_S = 600.0   # shorter files are aborted captures / test blips (5 of 15)
EFFORT_SOURCES = ("mag", "pca")


@dataclass
class InhouseStreams:
    code: str                  # anonymised 'S01'.. (file order); mapping is NOT written anywhere committed
    duration_s: float
    fs: float
    n_epochs: int
    n_beats: np.ndarray        # valid beats per 30 s epoch
    rr: np.ndarray             # float32 (n_epochs*60,)  cardiac grid, 2 Hz
    amp: np.ndarray
    effort: dict               # source -> float16 (2, n_epochs*960) CANet-ready belt-like traces
    effort_raw: dict           # source -> float32 (n_epochs*960,) band-passed trace BEFORE MESA normalisation
    motion_frac: np.ndarray    # per-epoch fraction of samples flagged as gross motion by envelope.py
    mag_pre: np.ndarray        # float32 (n_epochs*960,) accel magnitude (g), 32 Hz, drift-removed (for PSD/gap study)
    median_rr_s: float
    pct_rr_dropped: float


def list_recordings() -> list[Path]:
    return sorted(REC_DIR.glob("*.txt"))


def _to_grid(t_src: np.ndarray, x: np.ndarray, n_samples: int, hz: float) -> np.ndarray:
    return np.interp(np.arange(n_samples) / hz, t_src, x).astype(np.float64)


def _belt_like(x32: np.ndarray) -> np.ndarray:
    """Identical to src/mesa/respiratory_stream._belt (band-pass, robust-scale, clip) on a 32 Hz trace."""
    x = sosfiltfilt(butter(2, R.BAND, btype="band", fs=R.BELT_HZ, output="sos"), np.nan_to_num(x32))
    scale = 1.4826 * np.median(np.abs(x - np.median(x)))
    return np.clip(x / scale if scale > 1e-9 else np.zeros_like(x), -R.CLIP, R.CLIP).astype(np.float16)


def build(path: Path, code: str) -> InhouseStreams | None:
    rec: SensorTileRecord = load_sensortile(path)
    if not np.isfinite(rec.duration_s) or rec.duration_s < MIN_DURATION_S:
        return None
    n_ep = int(rec.duration_s // C.EPOCH_S)
    t = (rec.timestamp_us - rec.timestamp_us[0]) / 1e6

    # ---- cardiac
    rp = detect_rpeaks(rec)
    t_pk = rp.r_peaks / rec.fs
    rr = np.diff(t_pk); t_rr, amp = t_pk[1:], rp.r_amplitudes[1:]
    ok = (rr >= RR_MIN_S) & (rr <= RR_MAX_S)
    t_ok, rr_ok, amp_ok = t_rr[ok], rr[ok], amp[ok]
    n_beats = np.histogram(t_ok, bins=np.arange(n_ep + 1) * C.EPOCH_S)[0].astype(np.int32)
    tg = np.arange(0.0, n_ep * C.EPOCH_S, 1.0 / C.GRID_HZ)
    rr_g = (np.interp(tg, t_ok, rr_ok) / np.median(rr_ok) - 1.0).astype(np.float32)
    amp_g = np.interp(tg, t_ok, amp_ok / (np.median(np.abs(amp_ok)) or 1.0)).astype(np.float32)

    # ---- effort (accelerometer, mg -> g because envelope.py thresholds are in g)
    a_g = rec.accel_mg / 1000.0
    nb = n_ep * int(C.EPOCH_S * R.BELT_HZ)
    es = E.compute_effort_signal(a_g[:, 0], a_g[:, 1], a_g[:, 2], rec.fs)
    sos = butter(3, [E.BAND_LOW_HZ, E.BAND_HIGH_HZ], btype="band", fs=rec.fs, output="sos")
    xb = sosfiltfilt(sos, a_g - a_g.mean(0), axis=0)
    pc = xb @ np.linalg.eigh(np.cov(xb.T))[1][:, -1]
    traces = {"mag": es.bandpassed, "pca": pc}
    effort, effort_raw = {}, {}
    for k, tr in traces.items():
        x32 = _to_grid(t, tr, nb, R.BELT_HZ)
        effort_raw[k] = x32.astype(np.float32)
        b = _belt_like(x32)
        effort[k] = np.stack([b, b])
    mag32 = _to_grid(t, es.magnitude - np.median(es.magnitude), nb, R.BELT_HZ).astype(np.float32)
    motion = np.interp(np.arange(nb) / R.BELT_HZ, t, es.motion_mask.astype(float)) > 0.5
    return InhouseStreams(code, float(rec.duration_s), float(rec.fs), n_ep, n_beats, rr_g, amp_g, effort, effort_raw,
                          motion.reshape(n_ep, -1).mean(1).astype(np.float32), mag32,
                          float(np.median(rr_ok)), float(100 * (1 - ok.mean())))


def cardiac_windows(s: InhouseStreams, targets: np.ndarray, context: int = 2) -> np.ndarray:
    """(n, 2, 300) float32, identical transform to ecg_baseline.load_dataset (clip RR to +/-1, amp-1 to +/-3)."""
    w = C.epoch_windows(np.nan_to_num(s.rr), np.nan_to_num(s.amp, nan=1.0), targets, context).astype(np.float32)
    w[:, 0] = np.clip(w[:, 0], -1, 1)
    w[:, 1] = np.clip(w[:, 1] - 1, -3, 3)
    return w


def effort_windows(x: np.ndarray, targets: np.ndarray, context: int = 2) -> np.ndarray:
    """(n, 2, 4800) float32 windows from a (2, n_epochs*960) belt-like array, zero padded like the MESA GpuData."""
    spe = int(C.EPOCH_S * R.BELT_HZ); pad = context * spe
    xp = np.pad(x.astype(np.float32), ((0, 0), (pad, pad)))
    idx = targets[:, None] * spe + np.arange((2 * context + 1) * spe)[None]
    return xp[:, idx].transpose(1, 0, 2)
