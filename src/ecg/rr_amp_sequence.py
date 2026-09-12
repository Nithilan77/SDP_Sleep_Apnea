"""
Phase 3 iteration -- per-minute (RR-interval, R-peak amplitude) 2-channel
sequences for the CNN.

Adds R-peak amplitude as a second input channel alongside the RR-interval
sequence from rr_sequence.py. R-peak amplitude modulates with respiration
(EDR -- ECG-derived respiration, from the changing electrical axis as the
chest moves), so it carries information the RR-interval channel alone does
not -- this is what the literature's higher-accuracy single-lead models use
in place of a real respiration channel.

Same per-minute windows, same MIN_BEATS_PER_WINDOW threshold, same fixed-
length resampling grid as rr_sequence.py -- the set of (record, minute) pairs
evaluated is unchanged, so this stays directly comparable to Phase 3 v1 and
the Phase 2 baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ecg.features import MIN_BEATS_PER_WINDOW, WINDOW_S  # noqa: E402
from src.ecg.rpeaks import RPeakResult, detect_rpeaks  # noqa: E402
from src.ecg.rr_sequence import SEQ_LEN, _resample_to_grid  # noqa: E402
from src.ingest.physionet import ApneaRecord, load_record  # noqa: E402

N_CHANNELS = 2  # channel 0 = RR interval (s), channel 1 = R-peak amplitude


def compute_minute_sequences_2ch(rec: ApneaRecord, rpeaks: RPeakResult, seq_len: int = SEQ_LEN):
    """Return (sequences, labels, n_beats, minutes_kept) with sequences shaped
    (n_minutes_kept, 2, seq_len): channel 0 RR-interval (s), channel 1 R-peak
    amplitude (cleaned-signal units), both resampled onto the same grid.
    """
    if rec.labels is None:
        raise ValueError(f"{rec.name} has no .apn labels to align sequences to")

    fs = rec.fs
    r_peaks = rpeaks.r_peaks
    peak_times_s = r_peaks / fs
    rr_s = rpeaks.rr_intervals_s  # seconds, length = n_beats - 1
    rr_times_s = peak_times_s[1:]  # timestamp of the peak ending each RR interval
    # Align amplitude with the same peaks (index 1..n_beats-1) as the RR series.
    amp = rpeaks.r_amplitudes[1:]

    sequences, labels, n_beats_list, minutes_kept = [], [], [], []
    for minute in range(len(rec.labels)):
        t_start = minute * WINDOW_S
        t_end = t_start + WINDOW_S
        in_window = (rr_times_s >= t_start) & (rr_times_s < t_end)
        rr_win = rr_s[in_window]
        amp_win = amp[in_window]

        if len(rr_win) < MIN_BEATS_PER_WINDOW:
            continue  # same drop rule as rr_sequence.py / features.py

        rr_grid = _resample_to_grid(rr_win, seq_len)
        amp_grid = _resample_to_grid(amp_win, seq_len)
        sequences.append(np.stack([rr_grid, amp_grid], axis=0))
        labels.append(rec.labels[minute])
        n_beats_list.append(len(rr_win))
        minutes_kept.append(minute)

    return (np.stack(sequences), np.array(labels), np.array(n_beats_list),
            np.array(minutes_kept))


def build_sequence_dataset_2ch(record_names: list[str], seq_len: int = SEQ_LEN, verbose: bool = True):
    """Build the full (record, minute) -> 2-channel sequence dataset."""
    all_seq, all_label, all_record, all_minute, all_nbeats = [], [], [], [], []
    for name in record_names:
        rec = load_record(name)
        rpk = detect_rpeaks(rec)
        seqs, labels, n_beats, minutes = compute_minute_sequences_2ch(rec, rpk, seq_len)
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
        "sequence": np.concatenate(all_seq, axis=0),  # (n, 2, seq_len)
    }
