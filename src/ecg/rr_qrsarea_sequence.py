"""
Method 6 -- per-minute (RR-interval, QRS-area) 2-channel sequences for the
1D-CNN, mirroring rr_amp_sequence.py (method 2, RR + R-peak amplitude)
exactly except channel 1 is QRS-area (qrs_area.py) instead of R-peak
amplitude -- so this stays a controlled swap of one EDR proxy for another,
same model, same RR channel, same windows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import neurokit2 as nk
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ecg.features import MIN_BEATS_PER_WINDOW, WINDOW_S  # noqa: E402
from src.ecg.qrs_area import compute_qrs_area  # noqa: E402
from src.ecg.rpeaks import RPeakResult, detect_rpeaks  # noqa: E402
from src.ecg.rr_sequence import SEQ_LEN, _resample_to_grid  # noqa: E402
from src.ingest.physionet import ApneaRecord, load_record  # noqa: E402

N_CHANNELS = 2  # channel 0 = RR interval (s), channel 1 = QRS-area


def compute_minute_sequences_qrsarea(rec: ApneaRecord, rpeaks: RPeakResult, seq_len: int = SEQ_LEN):
    """Return (sequences, labels, n_beats, minutes_kept) with sequences shaped
    (n_minutes_kept, 2, seq_len): channel 0 RR-interval (s), channel 1
    QRS-area, both resampled onto the same grid.
    """
    if rec.labels is None:
        raise ValueError(f"{rec.name} has no .apn labels to align sequences to")

    fs = rec.fs
    r_peaks = rpeaks.r_peaks
    peak_times_s = r_peaks / fs
    rr_s = rpeaks.rr_intervals_s
    rr_times_s = peak_times_s[1:]

    cleaned = nk.ecg_clean(rec.signal, sampling_rate=int(fs))
    qrs_area_all = compute_qrs_area(cleaned, r_peaks, fs)
    qrs_area = qrs_area_all[1:]  # align with the same peaks (index 1..n_beats-1) as the RR series

    sequences, labels, n_beats_list, minutes_kept = [], [], [], []
    for minute in range(len(rec.labels)):
        t_start = minute * WINDOW_S
        t_end = t_start + WINDOW_S
        in_window = (rr_times_s >= t_start) & (rr_times_s < t_end)
        rr_win = rr_s[in_window]
        qrs_win = qrs_area[in_window]

        if len(rr_win) < MIN_BEATS_PER_WINDOW:
            continue

        rr_grid = _resample_to_grid(rr_win, seq_len)
        qrs_grid = _resample_to_grid(qrs_win, seq_len)
        sequences.append(np.stack([rr_grid, qrs_grid], axis=0))
        labels.append(rec.labels[minute])
        n_beats_list.append(len(rr_win))
        minutes_kept.append(minute)

    return (np.stack(sequences), np.array(labels), np.array(n_beats_list),
            np.array(minutes_kept))


def build_sequence_dataset_qrsarea(record_names: list[str], seq_len: int = SEQ_LEN, verbose: bool = True):
    all_seq, all_label, all_record, all_minute, all_nbeats = [], [], [], [], []
    for name in record_names:
        rec = load_record(name)
        rpk = detect_rpeaks(rec)
        seqs, labels, n_beats, minutes = compute_minute_sequences_qrsarea(rec, rpk, seq_len)
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
