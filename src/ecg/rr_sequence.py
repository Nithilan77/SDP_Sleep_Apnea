"""
Per-minute RR-interval sequences for the 1D-CNN (Phase 3).

Unlike features.py (which collapses each minute's RR intervals into a handful
of scalar CVHR/HRV features for the classical baseline), this module keeps
the RR-interval *sequence* itself, resampled onto a fixed-length grid so
minutes with different beat counts (bradycardia during apnea = fewer beats;
tachycardia on resumption = more) become fixed-size CNN input vectors.

Per CLAUDE.md rule 6: RR intervals are kept in TIME units (seconds) so this
transfers across sample rates (100 Hz PhysioNet vs 240 Hz QVAR) -- only the
*sequence* is resampled onto a common grid, not the time axis itself.

Uses the exact same per-minute windows and MIN_BEATS_PER_WINDOW threshold as
features.py, so the set of (record, minute) pairs evaluated here is identical
to Phase 2's -- the CNN and classical baseline numbers are directly
comparable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ecg.features import MIN_BEATS_PER_WINDOW, WINDOW_S  # noqa: E402
from src.ecg.rpeaks import RPeakResult, detect_rpeaks  # noqa: E402
from src.ingest.physionet import ApneaRecord, load_record  # noqa: E402

SEQ_LEN = 60  # fixed-length grid each minute's RR sequence is resampled onto


def _resample_to_grid(values: np.ndarray, seq_len: int) -> np.ndarray:
    """Linearly interpolate a 1-D sequence onto `seq_len` evenly-spaced points."""
    n = len(values)
    if n == 1:
        return np.full(seq_len, values[0], dtype=np.float32)
    x_src = np.linspace(0.0, 1.0, n)
    x_dst = np.linspace(0.0, 1.0, seq_len)
    return np.interp(x_dst, x_src, values).astype(np.float32)


def compute_minute_sequences(rec: ApneaRecord, rpeaks: RPeakResult, seq_len: int = SEQ_LEN):
    """Return (sequences, labels, n_beats) for one record's labelled minutes.

    sequences: float32 array (n_minutes_kept, seq_len) of RR intervals in
               seconds, resampled onto a fixed grid.
    labels:    'A'/'N' array, same length.
    n_beats:   raw beat count per kept minute (diagnostic only).
    """
    if rec.labels is None:
        raise ValueError(f"{rec.name} has no .apn labels to align sequences to")

    fs = rec.fs
    r_peaks = rpeaks.r_peaks
    peak_times_s = r_peaks / fs
    rr_s = rpeaks.rr_intervals_s  # seconds, length = n_beats - 1
    rr_times_s = peak_times_s[1:]  # timestamp of the peak ending each RR interval

    sequences, labels, n_beats_list, minutes_kept = [], [], [], []
    for minute in range(len(rec.labels)):
        t_start = minute * WINDOW_S
        t_end = t_start + WINDOW_S
        in_window = (rr_times_s >= t_start) & (rr_times_s < t_end)
        rr_win = rr_s[in_window]

        if len(rr_win) < MIN_BEATS_PER_WINDOW:
            continue  # same drop rule as features.py -> same minutes kept

        sequences.append(_resample_to_grid(rr_win, seq_len))
        labels.append(rec.labels[minute])
        n_beats_list.append(len(rr_win))
        minutes_kept.append(minute)

    return (np.stack(sequences), np.array(labels), np.array(n_beats_list),
            np.array(minutes_kept))


def build_sequence_dataset(record_names: list[str], seq_len: int = SEQ_LEN, verbose: bool = True):
    """Build the full (record, minute) -> RR-sequence dataset across many records.

    Returns a dict with parallel arrays: 'record' (str), 'minute' (int),
    'label' ('A'/'N'), 'n_beats' (int), 'sequence' (n, seq_len) float32.
    """
    all_seq, all_label, all_record, all_minute, all_nbeats = [], [], [], [], []
    for name in record_names:
        rec = load_record(name)
        rpk = detect_rpeaks(rec)
        seqs, labels, n_beats, minutes = compute_minute_sequences(rec, rpk, seq_len)
        all_seq.append(seqs)
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
        "sequence": np.concatenate(all_seq, axis=0),
    }
