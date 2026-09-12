"""
R-peak detection for Apnea-ECG records, using neurokit2.

Usage:
    python src/ecg/rpeaks.py            # sanity-check a01 and c01
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import neurokit2 as nk
import numpy as np

# Allow running as a plain script from the project root (`python src/ecg/rpeaks.py`)
# without needing to install the package or use -m.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingest.physionet import ApneaRecord, load_record  # noqa: E402

# Physiologically plausible RR-interval bounds (seconds), i.e. 30-220 bpm.
RR_MIN_S = 60.0 / 220.0
RR_MAX_S = 60.0 / 30.0


@dataclass
class RPeakResult:
    name: str
    fs: float
    n_beats: int
    r_peaks: np.ndarray  # sample indices
    rr_intervals_s: np.ndarray  # seconds, len = n_beats - 1
    mean_bpm: float
    pct_implausible_rr: float
    r_amplitudes: np.ndarray  # cleaned-signal amplitude at each r_peak, len = n_beats


def detect_rpeaks(rec: ApneaRecord) -> RPeakResult:
    """Detect R-peaks in a loaded ApneaRecord using neurokit2's ecg_peaks."""
    cleaned = nk.ecg_clean(rec.signal, sampling_rate=int(rec.fs))
    _, info = nk.ecg_peaks(cleaned, sampling_rate=int(rec.fs))
    r_peaks = np.asarray(info["ECG_R_Peaks"])
    r_amplitudes = cleaned[r_peaks]

    rr_intervals_s = np.diff(r_peaks) / rec.fs
    mean_bpm = 60.0 / np.mean(rr_intervals_s) if len(rr_intervals_s) else float("nan")

    implausible = (rr_intervals_s < RR_MIN_S) | (rr_intervals_s > RR_MAX_S)
    pct_implausible = 100.0 * np.mean(implausible) if len(rr_intervals_s) else float("nan")

    return RPeakResult(
        name=rec.name,
        fs=rec.fs,
        n_beats=len(r_peaks),
        r_peaks=r_peaks,
        rr_intervals_s=rr_intervals_s,
        mean_bpm=mean_bpm,
        pct_implausible_rr=pct_implausible,
        r_amplitudes=r_amplitudes,
    )


def _report(name: str) -> RPeakResult:
    rec = load_record(name)
    res = detect_rpeaks(rec)
    print(f"{name}: {res.n_beats} beats, mean HR = {res.mean_bpm:.1f} bpm, "
          f"implausible RR = {res.pct_implausible_rr:.2f}%")
    return res


if __name__ == "__main__":
    print("=== R-peak detection sanity check (neurokit2) ===")
    _report("a01")
    _report("c01")
