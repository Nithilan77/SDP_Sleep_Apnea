"""
Per-minute CVHR (Cyclic Variation of Heart Rate) features for Apnea-ECG records.

Apnea-ECG labels are per-minute, so features are computed on matching 60-second
windows: minute i covers samples [i*60*fs, (i+1)*60*fs). During apnea, heart
rate typically dips (bradycardia during the pause) then surges (tachycardia on
resumption) -- so apnea minutes should show a larger within-minute HR swing
than normal minutes. That is the CVHR fingerprint this module checks for.

Usage:
    python src/ecg/features.py            # per-minute features for a01, sanity check
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ecg.rpeaks import RPeakResult, detect_rpeaks  # noqa: E402
from src.ingest.physionet import ApneaRecord, load_record  # noqa: E402

WINDOW_S = 60.0  # matches Apnea-ECG per-minute label resolution
MIN_BEATS_PER_WINDOW = 3  # need at least a few beats to trust the features


FEATURE_COLUMNS = [
    "mean_hr", "hr_swing", "sdnn_ms", "rmssd_ms", "pnn50", "rr_mean_ms", "hr_delta_prev",
]


@dataclass
class MinuteFeatures:
    """Per-minute CVHR feature table for one record."""

    name: str
    table: pd.DataFrame  # columns: minute, label, n_beats, + FEATURE_COLUMNS


def compute_minute_features(rec: ApneaRecord, rpeaks: RPeakResult) -> MinuteFeatures:
    """Compute per-minute CVHR features aligned to rec.labels (one row/minute).

    Features per minute (all derived from RR intervals falling in that 60s window):
      - mean_hr:    mean instantaneous heart rate (bpm)
      - hr_swing:   max-min instantaneous HR within the minute (bpm) -- the core
                    CVHR fingerprint: apnea causes a within-minute brady->tachy swing
      - sdnn_ms:    std of RR intervals (ms) -- overall HRV
      - rmssd_ms:   root-mean-square of successive RR differences (ms) -- short-term HRV
      - pnn50:      fraction of successive RR diffs > 50ms
      - rr_mean_ms: mean RR interval (ms)
      - hr_delta_prev: |mean_hr[minute] - mean_hr[minute-1]| -- captures the
                    cyclic (minute-to-minute) swing that within-minute swing can miss
    """
    if rec.labels is None:
        raise ValueError(f"{rec.name} has no .apn labels to align features to")

    fs = rec.fs
    r_peaks = rpeaks.r_peaks
    peak_times_s = r_peaks / fs

    # Instantaneous HR at each peak (from the RR interval ending at that peak).
    # inst_hr[0] is undefined (no preceding peak); align inst_hr[i] with peak i
    # for i >= 1, using peak_times_s[i] as the timestamp.
    inst_hr = 60.0 / rpeaks.rr_intervals_s  # length = n_beats - 1
    inst_hr_times_s = peak_times_s[1:]  # timestamp of the peak ending each RR interval
    rr_ms = rpeaks.rr_intervals_s * 1000.0
    rr_diff_ms = np.diff(rr_ms)  # successive RR differences, length = n_beats - 2

    rows = []
    n_minutes = len(rec.labels)
    for minute in range(n_minutes):
        t_start = minute * WINDOW_S
        t_end = t_start + WINDOW_S

        in_window = (inst_hr_times_s >= t_start) & (inst_hr_times_s < t_end)
        hr_win = inst_hr[in_window]
        rr_win = rr_ms[in_window]
        # rr_diff_ms[i] corresponds to the gap between rr_ms[i] and rr_ms[i+1],
        # i.e. it "belongs" to inst_hr_times_s[i+1] -- reuse in_window shifted by one.
        diff_in_window = in_window[1:] if len(in_window) > 1 else np.array([], dtype=bool)
        rr_diff_win = rr_diff_ms[diff_in_window[: len(rr_diff_ms)]] if len(rr_diff_ms) else np.array([])

        if len(hr_win) < MIN_BEATS_PER_WINDOW:
            mean_hr = hr_swing = sdnn_ms = rmssd_ms = pnn50 = rr_mean_ms = np.nan
        else:
            mean_hr = float(np.mean(hr_win))
            hr_swing = float(np.max(hr_win) - np.min(hr_win))
            sdnn_ms = float(np.std(rr_win, ddof=1))
            rr_mean_ms = float(np.mean(rr_win))
            if len(rr_diff_win) >= 2:
                rmssd_ms = float(np.sqrt(np.mean(rr_diff_win ** 2)))
                pnn50 = float(np.mean(np.abs(rr_diff_win) > 50.0))
            else:
                rmssd_ms = np.nan
                pnn50 = np.nan

        rows.append({
            "minute": minute,
            "label": rec.labels[minute],
            "n_beats": int(len(hr_win)),
            "mean_hr": mean_hr,
            "hr_swing": hr_swing,
            "sdnn_ms": sdnn_ms,
            "rmssd_ms": rmssd_ms,
            "pnn50": pnn50,
            "rr_mean_ms": rr_mean_ms,
        })

    table = pd.DataFrame(rows)
    # Cyclic minute-to-minute HR delta -- needs the neighbouring row, so add it
    # as a post-process pass over the completed per-minute table.
    table["hr_delta_prev"] = table["mean_hr"].diff().abs()

    return MinuteFeatures(name=rec.name, table=table)


def build_dataset(record_names: list[str], verbose: bool = True) -> pd.DataFrame:
    """Compute per-minute features for many records and concatenate into one table.

    Adds a 'record' column so callers can do subject-wise / leave-one-subject-out
    splits (each Apnea-ECG record is a different subject).
    """
    frames = []
    for name in record_names:
        rec = load_record(name)
        rpk = detect_rpeaks(rec)
        feats = compute_minute_features(rec, rpk)
        t = feats.table.copy()
        t.insert(0, "record", name)
        frames.append(t)
        if verbose:
            print(f"  {name}: {len(t)} minutes "
                  f"(apnea={int((t['label'] == 'A').sum())}, "
                  f"normal={int((t['label'] == 'N').sum())})")
    return pd.concat(frames, ignore_index=True)


def _report(name: str) -> MinuteFeatures:
    rec = load_record(name)
    rpk = detect_rpeaks(rec)
    feats = compute_minute_features(rec, rpk)

    t = feats.table.dropna(subset=["hr_swing"])
    n_dropped = len(feats.table) - len(t)
    apnea = t[t["label"] == "A"]
    normal = t[t["label"] == "N"]

    print(f"{name}: {len(t)} minutes with usable features "
          f"({n_dropped} dropped for too few beats)")
    print(f"  apnea minutes  (n={len(apnea):4d}): mean HR-swing = "
          f"{apnea['hr_swing'].mean():.2f} bpm, mean SDNN = {apnea['sdnn_ms'].mean():.1f} ms")
    print(f"  normal minutes (n={len(normal):4d}): mean HR-swing = "
          f"{normal['hr_swing'].mean():.2f} bpm, mean SDNN = {normal['sdnn_ms'].mean():.1f} ms")
    return feats


if __name__ == "__main__":
    print("=== Per-minute CVHR features sanity check ===")
    a01 = _report("a01")

    t = a01.table.dropna(subset=["hr_swing"])
    apnea_swing = t[t["label"] == "A"]["hr_swing"].mean()
    normal_swing = t[t["label"] == "N"]["hr_swing"].mean()

    print()
    if apnea_swing > normal_swing:
        print(f"GREEN LIGHT: apnea HR-swing ({apnea_swing:.2f} bpm) > "
              f"normal HR-swing ({normal_swing:.2f} bpm) on a01. "
              "CVHR signal looks real -> proceed to Phase 2.")
    else:
        print(f"WARNING: apnea HR-swing ({apnea_swing:.2f} bpm) is NOT > "
              f"normal HR-swing ({normal_swing:.2f} bpm) on a01. "
              "Do not proceed until this is understood.")
