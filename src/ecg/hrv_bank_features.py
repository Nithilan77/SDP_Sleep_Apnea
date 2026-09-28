"""
Feature-study Phase -- per-minute HRV feature banks 3, 4, 5 (time-domain,
frequency-domain, nonlinear/Poincare), all derived from the same per-minute
RR-interval windows as features.py (Phase 2) and rr_sequence.py (Phase 3), so
every method in the feature study is evaluated on the identical minute set.

Three banks, each computed once per minute and returned together (they share
the same RR-window extraction, so it's wasteful to walk each record three
times):

  Bank 3 -- time-domain HRV:
    sdnn_ms, rmssd_ms, pnn50 (reused verbatim from features.py's definitions),
    mean_hr, hr_range_bpm (max-min instantaneous HR in the minute -- same
    quantity Phase 2 called hr_swing, renamed here to match the literature
    term), hrv_triangular_index.

    NOTE on hrv_triangular_index: the standard Task Force (1996) definition
    uses a 7.8ms histogram bin width and is defined for 24-hour recordings.
    Applied to a single 60s window (~60-100 beats) that bin width puts nearly
    every beat in its own bin, which is degenerate. We use a coarser 125ms
    bin width instead -- a documented deviation for this per-minute context,
    not the standard long-recording metric. Treat this feature as a rough
    geometric spread measure, not a clinical triangular index.

  Bank 4 -- frequency-domain HRV: vlf_power, lf_power, hf_power, lf_hf_ratio.
    Standard bands: VLF 0.003-0.04 Hz, LF 0.04-0.15 Hz, HF 0.15-0.4 Hz.
    Computed via Lomb-Scargle periodogram directly on the (unevenly sampled)
    beat-time/RR-interval series -- no resampling-induced aliasing, and it
    works on the small, irregular sample count a single minute gives us.

    NOTE on VLF: a meaningful VLF estimate needs a window long enough to
    contain multiple cycles of a 0.003 Hz oscillation (~333s = 5.5 min). A
    60s window is far short of that. vlf_power is computed and reported for
    completeness (it was requested explicitly), but should be treated as
    unreliable/noise-dominated, not a real physiological VLF estimate -- this
    is flagged here rather than silently reported as if valid. LF is also
    borderline (60s = ~2.4 cycles of the slowest LF frequency); HF is fine
    (many cycles fit in 60s given a ~1 Hz beat rate).

  Bank 5 -- nonlinear/Poincare: sd1_ms, sd2_ms, sd1_sd2_ratio (standard
    Poincare-plot formulas from successive RR differences), sampen (sample
    entropy, m=2, r=0.2*SD(RR), vectorized).

Usage:
    python src/ecg/hrv_bank_features.py     # sanity check on a01
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lombscargle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ecg.features import MIN_BEATS_PER_WINDOW, WINDOW_S  # noqa: E402
from src.ecg.rpeaks import RPeakResult, detect_rpeaks  # noqa: E402
from src.ingest.physionet import ApneaRecord, load_record  # noqa: E402

TIME_DOMAIN_COLUMNS = [
    "sdnn_ms", "rmssd_ms", "pnn50", "mean_hr", "hr_range_bpm", "hrv_triangular_index",
]
FREQ_DOMAIN_COLUMNS = ["vlf_power", "lf_power", "hf_power", "lf_hf_ratio"]
NONLINEAR_COLUMNS = ["sd1_ms", "sd2_ms", "sd1_sd2_ratio", "sampen"]

ALL_BANK_COLUMNS = TIME_DOMAIN_COLUMNS + FREQ_DOMAIN_COLUMNS + NONLINEAR_COLUMNS

TRIANGULAR_BIN_MS = 125.0
VLF_BAND = (0.003, 0.04)
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.4)
N_FREQ_POINTS = 200
SAMPEN_M = 2
SAMPEN_R_MULT = 0.2


@dataclass
class BankFeatures:
    name: str
    table: pd.DataFrame


def _triangular_index(rr_ms: np.ndarray, bin_width_ms: float = TRIANGULAR_BIN_MS) -> float:
    if len(rr_ms) < 3:
        return float("nan")
    lo, hi = float(rr_ms.min()), float(rr_ms.max())
    if hi <= lo:
        return float("nan")
    bins = np.arange(lo, hi + bin_width_ms, bin_width_ms)
    if len(bins) < 2:
        return float("nan")
    hist, _ = np.histogram(rr_ms, bins=bins)
    peak = hist.max()
    return float(len(rr_ms) / peak) if peak > 0 else float("nan")


def _freq_bands(rr_times_s: np.ndarray, rr_s: np.ndarray) -> tuple[float, float, float, float]:
    if len(rr_s) < 6:
        return float("nan"), float("nan"), float("nan"), float("nan")

    t = rr_times_s - rr_times_s[0]
    y = rr_s - np.mean(rr_s)
    freqs = np.linspace(VLF_BAND[0], HF_BAND[1], N_FREQ_POINTS)
    angular = 2.0 * np.pi * freqs
    power = lombscargle(t, y, angular, normalize=False) * 2.0 / len(y)

    vlf_mask = (freqs >= VLF_BAND[0]) & (freqs < VLF_BAND[1])
    lf_mask = (freqs >= LF_BAND[0]) & (freqs < LF_BAND[1])
    hf_mask = (freqs >= HF_BAND[0]) & (freqs < HF_BAND[1])

    vlf = float(np.trapezoid(power[vlf_mask], freqs[vlf_mask]))
    lf = float(np.trapezoid(power[lf_mask], freqs[lf_mask]))
    hf = float(np.trapezoid(power[hf_mask], freqs[hf_mask]))
    lf_hf = lf / hf if hf > 0 else float("nan")
    return vlf, lf, hf, lf_hf


def _poincare(rr_diff_ms: np.ndarray, sdnn_ms: float) -> tuple[float, float, float]:
    if len(rr_diff_ms) < 2 or np.isnan(sdnn_ms):
        return float("nan"), float("nan"), float("nan")
    sd1 = float(np.sqrt(0.5) * np.std(rr_diff_ms, ddof=1))
    sd2_sq = 2.0 * sdnn_ms ** 2 - 0.5 * np.var(rr_diff_ms, ddof=1)
    sd2 = float(np.sqrt(sd2_sq)) if sd2_sq > 0 else float("nan")
    sd1_sd2 = sd1 / sd2 if sd2 and not np.isnan(sd2) and sd2 > 0 else float("nan")
    return sd1, sd2, sd1_sd2


def _sample_entropy(rr_ms: np.ndarray, m: int = SAMPEN_M, r_mult: float = SAMPEN_R_MULT) -> float:
    n = len(rr_ms)
    if n < m + 2:
        return float("nan")
    r = r_mult * np.std(rr_ms, ddof=1)
    if r == 0 or np.isnan(r):
        return float("nan")

    def _count_matches(mm: int) -> int:
        length = n - mm + 1
        if length < 2:
            return 0
        templates = np.array([rr_ms[i:i + mm] for i in range(length)])
        diff = np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)
        matches = diff <= r
        np.fill_diagonal(matches, False)
        return int(matches.sum())

    b_count = _count_matches(m)
    a_count = _count_matches(m + 1)
    if b_count == 0 or a_count == 0:
        return float("nan")
    return float(-np.log(a_count / b_count))


def compute_minute_bank_features(rec: ApneaRecord, rpeaks: RPeakResult) -> BankFeatures:
    """Compute per-minute time/frequency/nonlinear HRV bank features.

    Same per-minute windowing and MIN_BEATS_PER_WINDOW threshold as
    features.py, so the (record, minute) set matches every other method in
    the feature study.
    """
    if rec.labels is None:
        raise ValueError(f"{rec.name} has no .apn labels to align features to")

    fs = rec.fs
    r_peaks = rpeaks.r_peaks
    peak_times_s = r_peaks / fs

    inst_hr = 60.0 / rpeaks.rr_intervals_s
    inst_hr_times_s = peak_times_s[1:]
    rr_ms_full = rpeaks.rr_intervals_s * 1000.0
    rr_diff_ms_full = np.diff(rr_ms_full)

    rows = []
    n_minutes = len(rec.labels)
    for minute in range(n_minutes):
        t_start = minute * WINDOW_S
        t_end = t_start + WINDOW_S

        in_window = (inst_hr_times_s >= t_start) & (inst_hr_times_s < t_end)
        hr_win = inst_hr[in_window]
        rr_win_ms = rr_ms_full[in_window]
        rr_win_s = rpeaks.rr_intervals_s[in_window]
        rr_win_times_s = inst_hr_times_s[in_window]

        diff_in_window = in_window[1:] if len(in_window) > 1 else np.array([], dtype=bool)
        rr_diff_win_ms = (
            rr_diff_ms_full[diff_in_window[: len(rr_diff_ms_full)]]
            if len(rr_diff_ms_full) else np.array([])
        )

        if len(hr_win) < MIN_BEATS_PER_WINDOW:
            vals = {c: float("nan") for c in ALL_BANK_COLUMNS}
        else:
            mean_hr = float(np.mean(hr_win))
            hr_range_bpm = float(np.max(hr_win) - np.min(hr_win))
            sdnn_ms = float(np.std(rr_win_ms, ddof=1)) if len(rr_win_ms) >= 2 else float("nan")
            if len(rr_diff_win_ms) >= 2:
                rmssd_ms = float(np.sqrt(np.mean(rr_diff_win_ms ** 2)))
                pnn50 = float(np.mean(np.abs(rr_diff_win_ms) > 50.0))
            else:
                rmssd_ms = float("nan")
                pnn50 = float("nan")
            tri_idx = _triangular_index(rr_win_ms)

            vlf, lf, hf, lf_hf = _freq_bands(rr_win_times_s, rr_win_s)

            sd1, sd2, sd1_sd2 = _poincare(rr_diff_win_ms, sdnn_ms)
            sampen = _sample_entropy(rr_win_ms)

            vals = {
                "sdnn_ms": sdnn_ms, "rmssd_ms": rmssd_ms, "pnn50": pnn50,
                "mean_hr": mean_hr, "hr_range_bpm": hr_range_bpm,
                "hrv_triangular_index": tri_idx,
                "vlf_power": vlf, "lf_power": lf, "hf_power": hf, "lf_hf_ratio": lf_hf,
                "sd1_ms": sd1, "sd2_ms": sd2, "sd1_sd2_ratio": sd1_sd2, "sampen": sampen,
            }

        row = {"minute": minute, "label": rec.labels[minute], "n_beats": int(len(hr_win))}
        row.update(vals)
        rows.append(row)

    return BankFeatures(name=rec.name, table=pd.DataFrame(rows))


def build_bank_dataset(record_names: list[str], verbose: bool = True) -> pd.DataFrame:
    frames = []
    for name in record_names:
        rec = load_record(name)
        rpk = detect_rpeaks(rec)
        feats = compute_minute_bank_features(rec, rpk)
        t = feats.table.copy()
        t.insert(0, "record", name)
        frames.append(t)
        if verbose:
            print(f"  {name}: {len(t)} minutes")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    print("=== HRV feature bank sanity check (a01) ===")
    rec = load_record("a01")
    rpk = detect_rpeaks(rec)
    feats = compute_minute_bank_features(rec, rpk)
    t = feats.table.dropna(subset=ALL_BANK_COLUMNS, how="all")
    apnea = t[t["label"] == "A"]
    normal = t[t["label"] == "N"]
    print(f"{len(t)} minutes with any usable bank features")
    for col in ALL_BANK_COLUMNS:
        print(f"  {col:22s} apnea mean={apnea[col].mean():10.4f}  "
              f"normal mean={normal[col].mean():10.4f}  "
              f"NaN%={100*t[col].isna().mean():5.1f}")
