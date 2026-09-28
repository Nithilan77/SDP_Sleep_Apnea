"""
Method 7 -- CWT (continuous wavelet transform) scalogram of the per-minute
RR-interval series, for a 2D-CNN.

Each minute's RR-interval sequence is first resampled onto the same
fixed-length grid as rr_sequence.py (method 1) -- seq_len points spanning
the 60s window, so the resampled sequence has an effective sampling period
of 60/seq_len seconds (exactly 1.0s for seq_len=60). A complex Morlet CWT
is then applied to that resampled sequence, producing a (n_scales, seq_len)
time-frequency image (the scalogram) per minute. Scales are chosen so the
recovered frequency range covers the same HRV VLF/LF/HF bands the frequency-
domain bank (method 4) targets (0.01-0.49 Hz) -- same bands, different
representation (a 2D time-frequency image instead of 4 integrated-power
scalars), so this is a meaningful comparison against method 4 as well as
against the raw-sequence methods.

Library note: scipy.signal.cwt/morlet2 were removed in this scipy version
(1.18) -- PyWavelets (pywt) is used instead.

Usage:
    python src/ecg/cwt_scalogram.py     # sanity check + a01 scalogram shape
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pywt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ecg.features import MIN_BEATS_PER_WINDOW, WINDOW_S  # noqa: E402
from src.ecg.rpeaks import RPeakResult, detect_rpeaks  # noqa: E402
from src.ecg.rr_sequence import SEQ_LEN, _resample_to_grid  # noqa: E402
from src.ingest.physionet import ApneaRecord, load_record  # noqa: E402

WAVELET = "cmor1.5-1.0"  # complex Morlet, bandwidth=1.5, center freq=1.0
N_SCALES = 32
FREQ_LO_HZ = 0.01  # ~ VLF
FREQ_HI_HZ = 0.49  # ~ just under Nyquist for a 1 Hz effective sampling rate


def _scales_for_freqs(seq_len: int, n_scales: int = N_SCALES) -> tuple[np.ndarray, float]:
    sampling_period = WINDOW_S / seq_len  # seconds/sample after resampling to the grid
    freqs = np.geomspace(FREQ_LO_HZ, FREQ_HI_HZ, n_scales)
    central_freq = pywt.central_frequency(WAVELET)
    scales = central_freq / (freqs * sampling_period)
    return scales, sampling_period


def compute_minute_scalograms(rec: ApneaRecord, rpeaks: RPeakResult, seq_len: int = SEQ_LEN,
                               n_scales: int = N_SCALES):
    """Return (scalograms, labels, n_beats, minutes_kept).

    scalograms: float32 array (n_minutes_kept, n_scales, seq_len) -- CWT
                magnitude of the resampled RR-interval sequence.
    """
    if rec.labels is None:
        raise ValueError(f"{rec.name} has no .apn labels to align scalograms to")

    fs = rec.fs
    r_peaks = rpeaks.r_peaks
    peak_times_s = r_peaks / fs
    rr_s = rpeaks.rr_intervals_s
    rr_times_s = peak_times_s[1:]

    scales, sampling_period = _scales_for_freqs(seq_len, n_scales)

    scalograms, labels, n_beats_list, minutes_kept = [], [], [], []
    for minute in range(len(rec.labels)):
        t_start = minute * WINDOW_S
        t_end = t_start + WINDOW_S
        in_window = (rr_times_s >= t_start) & (rr_times_s < t_end)
        rr_win = rr_s[in_window]

        if len(rr_win) < MIN_BEATS_PER_WINDOW:
            continue

        rr_grid = _resample_to_grid(rr_win, seq_len)
        coeffs, _ = pywt.cwt(rr_grid, scales, WAVELET, sampling_period=sampling_period)
        scalograms.append(np.abs(coeffs).astype(np.float32))
        labels.append(rec.labels[minute])
        n_beats_list.append(len(rr_win))
        minutes_kept.append(minute)

    return (np.stack(scalograms), np.array(labels), np.array(n_beats_list),
            np.array(minutes_kept))


def build_scalogram_dataset(record_names: list[str], seq_len: int = SEQ_LEN,
                             n_scales: int = N_SCALES, verbose: bool = True):
    all_sc, all_label, all_record, all_minute, all_nbeats = [], [], [], [], []
    for name in record_names:
        rec = load_record(name)
        rpk = detect_rpeaks(rec)
        sc, labels, n_beats, minutes = compute_minute_scalograms(rec, rpk, seq_len, n_scales)
        all_sc.append(sc)
        all_label.append(labels)
        all_nbeats.append(n_beats)
        all_minute.append(minutes)
        all_record.extend([name] * len(labels))
        if verbose:
            print(f"  {name}: {len(labels)} minutes")

    return {
        "record": np.array(all_record),
        "minute": np.concatenate(all_minute),
        "label": np.concatenate(all_label),
        "n_beats": np.concatenate(all_nbeats),
        "scalogram": np.concatenate(all_sc, axis=0),  # (n, n_scales, seq_len)
    }


if __name__ == "__main__":
    print("=== CWT scalogram sanity check (a01) ===")
    rec = load_record("a01")
    rpk = detect_rpeaks(rec)
    sc, labels, n_beats, minutes = compute_minute_scalograms(rec, rpk)
    print(f"{len(labels)} minutes, scalogram shape per minute = {sc.shape[1:]}")
    apnea_mean = sc[labels == "A"].mean()
    normal_mean = sc[labels == "N"].mean()
    print(f"mean |CWT| -- apnea: {apnea_mean:.4f}  normal: {normal_mean:.4f}")
