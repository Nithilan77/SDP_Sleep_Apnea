"""
Diagnostic for the three in-house recordings that stayed implausible in the
frozen-model plausibility test's RR-only variant (results/inhouse_validation/
plausibility/): user_1101hudson_ecg_000, user_1510hudson_ecg_000, and
user_2406Hari(chest_chest & ecg_000 (which inverted instead of dropping).

Since RR-only neutralizes the amplitude channel, an implausible result there
points at the RR/R-peak detection itself, not the amplitude domain gap. This
script does NOT touch the model or any detection threshold -- it only
visualizes and quantifies what the QVAR signal and R-peak detector are doing
on these three files, alongside a known-good comparison
(user_kishor0404_ecg_000, which dropped from 71% to 10.9% apnea rate in the
RR-only variant -- i.e. behaved as expected).

For each file, saves to results/inhouse_validation/hudson_debug/:
    - waveform_<file>_normal.png   -- a representative 45s window, peaks overlaid
    - waveform_<file>_biggest_gap.png -- a 45s window centered on the single
      largest RR interval in the recording, peaks overlaid (the window most
      likely to show a visibly missed beat, if beats are being under-detected)
    - rr_hist_<file>.png           -- RR-interval histogram (bimodality check:
      a second mode near ~2x the main mode would mean "real beats + missed-beat
      gaps read as one long interval")
    - summary.csv                  -- per-file quantitative summary

Usage:
    python src/ingest/hudson_debug.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ecg.rpeaks import RR_MAX_S, RR_MIN_S  # noqa: E402
from src.effort.envelope import accel_magnitude, compute_motion_mask  # noqa: E402
from src.ingest.sensortile import load_sensortile  # noqa: E402

INHOUSE_DIR = Path("data/recordings/inhouse_ecg")
RESULTS_DIR = Path("results/inhouse_validation/hudson_debug")

FILES = {
    "user_1101hudson_ecg_000.txt": "hudson_1101 (implausible, RR-only stayed high)",
    "user_1510hudson_ecg_000.txt": "hudson_1510 (implausible, RR-only stayed high)",
    "user_2406Hari(chest_chest & ecg_000.txt": "hari_2406 (inverted, RR-only went UP)",
    "user_kishor0404_ecg_000.txt": "kishor_0404 (KNOWN GOOD -- comparison baseline)",
}

WINDOW_S = 45.0  # plotted waveform window length


def _analyze_one(fname: str) -> dict:
    rec = load_sensortile(INHOUSE_DIR / fname)
    fs = rec.fs_measured
    qvar = rec.qvar

    cleaned = nk.ecg_clean(qvar, sampling_rate=int(round(fs)))
    _, info = nk.ecg_peaks(cleaned, sampling_rate=int(round(fs)))
    r_peaks = np.asarray(info["ECG_R_Peaks"])
    rr_s = np.diff(r_peaks) / fs

    too_fast_pct = 100.0 * np.mean(rr_s < RR_MIN_S) if len(rr_s) else np.nan
    too_slow_pct = 100.0 * np.mean(rr_s > RR_MAX_S) if len(rr_s) else np.nan

    mag_g = accel_magnitude(rec.accel_mg[:, 0], rec.accel_mg[:, 1], rec.accel_mg[:, 2]) / 1000.0
    motion_mask = compute_motion_mask(mag_g, fs)
    motion_pct = 100.0 * float(np.mean(motion_mask))

    max_gap_idx = int(np.argmax(rr_s)) if len(rr_s) else -1
    max_gap_s = float(rr_s[max_gap_idx]) if max_gap_idx >= 0 else np.nan

    summary = {
        "file": fname,
        "n_samples": rec.n_samples,
        "duration_s": round(rec.duration_s, 1),
        "fs_measured": round(fs, 2),
        "raw_qvar_min": float(qvar.min()),
        "raw_qvar_max": float(qvar.max()),
        "raw_qvar_std": float(qvar.std()),
        "motion_flagged_pct": round(motion_pct, 2),
        "n_beats": len(r_peaks),
        "mean_bpm": round(60.0 / np.mean(rr_s), 2) if len(rr_s) else np.nan,
        "rr_mean_s": round(float(np.mean(rr_s)), 3) if len(rr_s) else np.nan,
        "rr_std_s": round(float(np.std(rr_s)), 3) if len(rr_s) else np.nan,
        "rr_min_s": round(float(np.min(rr_s)), 3) if len(rr_s) else np.nan,
        "rr_max_s": round(float(np.max(rr_s)), 3) if len(rr_s) else np.nan,
        "implausible_too_fast_pct": round(too_fast_pct, 3),
        "implausible_too_slow_pct": round(too_slow_pct, 3),
        "implausible_total_pct": round(too_fast_pct + too_slow_pct, 3) if len(rr_s) else np.nan,
        "max_gap_s": round(max_gap_s, 2),
    }

    return {
        "summary": summary,
        "cleaned": cleaned,
        "r_peaks": r_peaks,
        "rr_s": rr_s,
        "fs": fs,
        "max_gap_beat_idx": max_gap_idx,
    }


def _plot_window(cleaned: np.ndarray, r_peaks: np.ndarray, fs: float,
                  center_s: float, window_s: float, title: str, out_path: Path) -> None:
    half = window_s / 2.0
    t0, t1 = max(0.0, center_s - half), center_s + half
    i0, i1 = int(t0 * fs), min(len(cleaned), int(t1 * fs))
    t = np.arange(i0, i1) / fs
    seg = cleaned[i0:i1]

    peaks_in_win = r_peaks[(r_peaks >= i0) & (r_peaks < i1)]

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(t, seg, linewidth=0.7, color="#4C72B0")
    ax.scatter(peaks_in_win / fs, cleaned[peaks_in_win], color="red", s=25, zorder=5, label="detected R-peak")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cleaned QVAR (a.u.)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_rr_hist(rr_s: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rr_s, bins=80, color="#55A868", alpha=0.8)
    ax.axvline(RR_MIN_S, color="red", linestyle="--", linewidth=1, label="plausible bounds")
    ax.axvline(RR_MAX_S, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("RR interval (s)")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run() -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for fname, label in FILES.items():
        print(f"\n--- {fname} ({label}) ---")
        result = _analyze_one(fname)
        s = result["summary"]
        rows.append(s)
        print(f"  n_beats={s['n_beats']} mean_bpm={s['mean_bpm']} "
              f"RR mean/std={s['rr_mean_s']}/{s['rr_std_s']}s "
              f"implausible: fast={s['implausible_too_fast_pct']}% slow={s['implausible_too_slow_pct']}% "
              f"max_gap={s['max_gap_s']}s motion_flagged={s['motion_flagged_pct']}%")

        safe_name = fname.replace("/", "_").replace(" ", "_").replace("&", "and")

        # Window 1: a "normal" window a quarter into the recording.
        normal_center_s = 0.25 * (len(result["cleaned"]) / result["fs"])
        _plot_window(result["cleaned"], result["r_peaks"], result["fs"], normal_center_s,
                     WINDOW_S, f"{label}: representative {WINDOW_S:.0f}s window",
                     RESULTS_DIR / f"waveform_{safe_name}_normal.png")

        # Window 2: centered on the single biggest RR gap (most likely to show a missed beat).
        if result["max_gap_beat_idx"] >= 0:
            gap_peak_sample = result["r_peaks"][result["max_gap_beat_idx"] + 1]
            gap_center_s = gap_peak_sample / result["fs"]
            _plot_window(result["cleaned"], result["r_peaks"], result["fs"], gap_center_s,
                         WINDOW_S, f"{label}: window around largest RR gap ({s['max_gap_s']:.1f}s)",
                         RESULTS_DIR / f"waveform_{safe_name}_biggest_gap.png")

        _plot_rr_hist(result["rr_s"], f"{label}: RR-interval histogram", RESULTS_DIR / f"rr_hist_{safe_name}.png")

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    df = run()
    print("\n=== SUMMARY ===")
    print(df.to_string(index=False))
