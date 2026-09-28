"""
Method 6 -- QRS-area, an alternative ECG-derived-respiration (EDR) proxy to
Phase 3's R-peak amplitude.

R-peak amplitude (rpeaks.py's r_amplitudes, used in method 2 / the adopted
81.2% model) is a single sample value at the peak. QRS-area instead
integrates the absolute signal over a small window around each R-peak,
capturing the whole QRS complex's morphology rather than just its tip.
Chest-wall motion during breathing rotates the heart's electrical axis
relative to the (fixed) electrodes, which changes the QRS complex's shape as
well as its peak height -- so QRS-area is a plausible alternative EDR signal
carrying respiration-linked information the single-sample amplitude may miss
or represent differently.

Usage:
    python src/ecg/qrs_area.py       # sanity check on a01
"""
from __future__ import annotations

import sys
from pathlib import Path

import neurokit2 as nk
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingest.physionet import ApneaRecord, load_record  # noqa: E402

QRS_HALF_WINDOW_MS = 40.0  # +/- 40ms around each R-peak, a typical QRS-complex half-width


def compute_qrs_area(cleaned_signal: np.ndarray, r_peaks: np.ndarray, fs: float,
                      half_window_ms: float = QRS_HALF_WINDOW_MS) -> np.ndarray:
    """Per-beat QRS-area: integral of |cleaned ECG| over a window centered on
    each R-peak, in (signal-unit * seconds). Same cleaned-signal units as
    rpeaks.py's r_amplitudes, so both are on PhysioNet's calibrated-ECG scale
    (arbitrary but consistent across beats/records within this dataset).
    """
    half_window = int(round(half_window_ms / 1000.0 * fs))
    areas = np.empty(len(r_peaks), dtype=np.float64)
    for i, p in enumerate(r_peaks):
        lo = max(0, p - half_window)
        hi = min(len(cleaned_signal), p + half_window + 1)
        seg = np.abs(cleaned_signal[lo:hi])
        areas[i] = np.trapezoid(seg) / fs
    return areas


def _report(name: str) -> None:
    rec = load_record(name)
    cleaned = nk.ecg_clean(rec.signal, sampling_rate=int(rec.fs))
    _, info = nk.ecg_peaks(cleaned, sampling_rate=int(rec.fs))
    r_peaks = np.asarray(info["ECG_R_Peaks"])
    areas = compute_qrs_area(cleaned, r_peaks, rec.fs)
    print(f"{name}: {len(r_peaks)} beats, QRS-area mean={areas.mean():.5f} "
          f"std={areas.std():.5f} min={areas.min():.5f} max={areas.max():.5f}")


if __name__ == "__main__":
    print("=== QRS-area (EDR proxy) sanity check ===")
    _report("a01")
    _report("c01")
