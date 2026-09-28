"""
Method 8 -- STFT spectrogram of the raw per-minute ECG segment, for a
2D-CNN. Unlike every other method in this study, this one does NOT derive
from R-peaks/RR-intervals at all -- it's the cleaned ECG waveform itself,
windowed and Fourier-transformed, so QRS-complex morphology and its
respiration-linked modulation show up directly in the time-frequency image
rather than through a beat-detection intermediate step.

Same per-minute window (60s) and the same beat-count-based
minute-keep rule as every other method (reusing the RR series purely to
decide which minutes have enough signal quality to trust, for an identical
minute set across the whole study) -- the spectrogram itself is computed
from the raw cleaned ECG samples in that window, independent of R-peaks.

STFT params: nperseg=128 (1.28s), hop=100 samples (1.0s) at fs=100 Hz ->
~65 frequency bins (0-50 Hz Nyquist) x ~60 time bins per minute -- chosen to
land near method 7's (32, 60) scalogram size for a comparable 2D-CNN
training cost, not because 65 bins/1Hz hop are individually principled.
Log-magnitude (log1p) is used as the image, since raw STFT power is
dominated by a few huge QRS bins and would otherwise wash out everything
else.

Usage:
    python src/ecg/ecg_spectrogram.py    # sanity check on a01
"""
from __future__ import annotations

import sys
from pathlib import Path

import neurokit2 as nk
import numpy as np
from scipy.signal import spectrogram

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ecg.features import MIN_BEATS_PER_WINDOW, WINDOW_S  # noqa: E402
from src.ecg.rpeaks import RPeakResult, detect_rpeaks  # noqa: E402
from src.ingest.physionet import ApneaRecord, load_record  # noqa: E402

NPERSEG = 128
HOP = 100  # samples: NPERSEG - noverlap


def compute_minute_spectrograms(rec: ApneaRecord, rpeaks: RPeakResult):
    """Return (spectrograms, labels, n_beats, minutes_kept).

    spectrograms: float32 array (n_minutes_kept, n_freq_bins, n_time_bins) --
                  log1p STFT magnitude of the cleaned raw ECG in that minute.
    """
    if rec.labels is None:
        raise ValueError(f"{rec.name} has no .apn labels to align spectrograms to")

    fs = rec.fs
    r_peaks = rpeaks.r_peaks
    peak_times_s = r_peaks / fs
    rr_times_s = peak_times_s[1:]
    rr_s = rpeaks.rr_intervals_s

    cleaned = nk.ecg_clean(rec.signal, sampling_rate=int(fs))
    noverlap = NPERSEG - HOP
    window_samples = int(round(WINDOW_S * fs))

    spectrograms, labels, n_beats_list, minutes_kept = [], [], [], []
    for minute in range(len(rec.labels)):
        t_start = minute * WINDOW_S
        t_end = t_start + WINDOW_S

        in_window = (rr_times_s >= t_start) & (rr_times_s < t_end)
        n_beats = int(np.sum(in_window))
        if n_beats < MIN_BEATS_PER_WINDOW:
            continue  # same drop rule as every other method -> same minute set

        i_start = int(round(t_start * fs))
        seg = cleaned[i_start:i_start + window_samples]
        if len(seg) < window_samples:
            # Last minute of a record can run short of a full 60s of signal
            # (recording ends mid-minute); zero-pad so every minute produces
            # a fixed-shape spectrogram, matching every other method's fixed
            # per-minute output shape.
            seg = np.pad(seg, (0, window_samples - len(seg)))

        _, _, Sxx = spectrogram(seg, fs=fs, nperseg=NPERSEG, noverlap=noverlap)
        spectrograms.append(np.log1p(Sxx).astype(np.float32))
        labels.append(rec.labels[minute])
        n_beats_list.append(n_beats)
        minutes_kept.append(minute)

    return (np.stack(spectrograms), np.array(labels), np.array(n_beats_list),
            np.array(minutes_kept))


def build_spectrogram_dataset(record_names: list[str], verbose: bool = True):
    all_sp, all_label, all_record, all_minute, all_nbeats = [], [], [], [], []
    for name in record_names:
        rec = load_record(name)
        rpk = detect_rpeaks(rec)
        sp, labels, n_beats, minutes = compute_minute_spectrograms(rec, rpk)
        all_sp.append(sp)
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
        "spectrogram": np.concatenate(all_sp, axis=0),
    }


if __name__ == "__main__":
    print("=== ECG spectrogram sanity check (a01) ===")
    rec = load_record("a01")
    rpk = detect_rpeaks(rec)
    sp, labels, n_beats, minutes = compute_minute_spectrograms(rec, rpk)
    print(f"{len(labels)} minutes, spectrogram shape per minute = {sp.shape[1:]}")
    apnea_mean = sp[labels == "A"].mean()
    normal_mean = sp[labels == "N"].mean()
    print(f"mean log1p|STFT| -- apnea: {apnea_mean:.4f}  normal: {normal_mean:.4f}")
