"""
Per-subject signal-quality report for the in-house SensorTile recordings.

Runs the existing R-peak detector (src/ecg/rpeaks.py) on the QVAR channel of
every file in data/recordings/inhouse_ecg/ and reports, per file:
    - measured vs nominal sample rate, duration, n_samples, timestamp gaps
    - whether the accelerometer (effort branch) is present and usable
    - R-peaks detected, mean HR, % physiologically-implausible RR intervals
    - a plausibility flag (sane resting HR range, implausible-RR% comparable
      to PhysioNet)

IMPORTANT: this is a signal-QUALITY / PLAUSIBILITY check only. There are no
apnea labels in this data (healthy subjects, validation-only per CLAUDE.md).
It cannot and must not be read as an accuracy or sensitivity number.

Usage:
    python src/ingest/inhouse_quality_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ecg.rpeaks import detect_rpeaks  # noqa: E402
from src.ingest.sensortile import load_sensortile  # noqa: E402

DATA_DIR = Path("data/recordings/inhouse_ecg")
RESULTS_DIR = Path("results/inhouse_validation")

# Sane resting/light-activity HR range for a plausibility flag (not a clinical bound).
HR_MIN_BPM = 40.0
HR_MAX_BPM = 120.0

# PhysioNet's clean recordings showed ~0.00-0.02% implausible RR. Give a wide
# margin above that before calling a file "too noisy to use".
IMPLAUSIBLE_RR_OK_PCT = 1.0


def _plausibility_flag(mean_bpm: float, pct_implausible_rr: float, n_samples: int) -> str:
    if n_samples < 2:
        return "TOO_SHORT"
    if np.isnan(mean_bpm) or np.isnan(pct_implausible_rr):
        return "NO_BEATS_DETECTED"
    hr_ok = HR_MIN_BPM <= mean_bpm <= HR_MAX_BPM
    rr_ok = pct_implausible_rr <= IMPLAUSIBLE_RR_OK_PCT
    if hr_ok and rr_ok:
        return "PLAUSIBLE"
    if not hr_ok and not rr_ok:
        return "IMPLAUSIBLE_HR_AND_NOISY"
    if not hr_ok:
        return "IMPLAUSIBLE_HR"
    return "NOISY_ECG"


def run_quality_report() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {DATA_DIR}")

    rows = []
    for path in files:
        print(f"\n--- {path.name} ---")
        try:
            rec = load_sensortile(path)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR loading {path.name}: {e}")
            rows.append({
                "file": path.name, "load_error": str(e),
                "fs_measured_hz": np.nan, "fs_nominal_hz": np.nan,
                "duration_s": np.nan, "n_samples": np.nan, "n_gaps": np.nan,
                "accel_usable": np.nan, "n_beats": np.nan, "mean_hr_bpm": np.nan,
                "pct_implausible_rr": np.nan, "plausibility_flag": "LOAD_ERROR",
            })
            continue

        if rec.n_samples < 2:
            rows.append({
                "file": path.name, "load_error": "",
                "fs_measured_hz": rec.fs_measured, "fs_nominal_hz": rec.fs_nominal,
                "duration_s": rec.duration_s, "n_samples": rec.n_samples,
                "n_gaps": rec.n_gaps, "accel_usable": rec.accel_is_usable,
                "n_beats": np.nan, "mean_hr_bpm": np.nan,
                "pct_implausible_rr": np.nan,
                "plausibility_flag": "TOO_SHORT",
            })
            continue

        try:
            res = detect_rpeaks(rec)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR detecting R-peaks in {path.name}: {e}")
            rows.append({
                "file": path.name, "load_error": "",
                "fs_measured_hz": rec.fs_measured, "fs_nominal_hz": rec.fs_nominal,
                "duration_s": rec.duration_s, "n_samples": rec.n_samples,
                "n_gaps": rec.n_gaps, "accel_usable": rec.accel_is_usable,
                "n_beats": np.nan, "mean_hr_bpm": np.nan,
                "pct_implausible_rr": np.nan,
                "plausibility_flag": f"RPEAK_ERROR: {e}",
            })
            continue

        flag = _plausibility_flag(res.mean_bpm, res.pct_implausible_rr, rec.n_samples)
        print(f"  fs={rec.fs_measured:.2f}Hz, {res.n_beats} beats, "
              f"mean HR={res.mean_bpm:.1f} bpm, implausible RR={res.pct_implausible_rr:.2f}%, "
              f"flag={flag}")

        rows.append({
            "file": path.name,
            "load_error": "",
            "fs_measured_hz": round(rec.fs_measured, 3),
            "fs_nominal_hz": rec.fs_nominal,
            "duration_s": round(rec.duration_s, 1),
            "n_samples": rec.n_samples,
            "n_gaps": rec.n_gaps,
            "accel_usable": rec.accel_is_usable,
            "n_beats": res.n_beats,
            "mean_hr_bpm": round(float(res.mean_bpm), 2),
            "pct_implausible_rr": round(float(res.pct_implausible_rr), 3),
            "plausibility_flag": flag,
        })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "quality_report.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    _write_readme(df, RESULTS_DIR / "README.md")
    return df


def _write_readme(df: pd.DataFrame, path: Path) -> None:
    usable = df[df["plausibility_flag"] == "PLAUSIBLE"]
    accel_usable_count = int(df["accel_usable"].fillna(False).sum())
    lines = [
        "# In-house SensorTile QVAR-ECG quality report",
        "",
        "**Scope: signal quality / plausibility only. No apnea labels exist for this data",
        "(healthy 19-20yo subjects, validation-only per CLAUDE.md Sec.4). This report",
        "cannot and does not produce a sensitivity/accuracy number.**",
        "",
        f"Files scanned: {len(df)}",
        f"Files with a plausible QVAR-derived HR (HR {HR_MIN_BPM:.0f}-{HR_MAX_BPM:.0f} bpm, "
        f"implausible-RR <= {IMPLAUSIBLE_RR_OK_PCT:.1f}%): {len(usable)}",
        f"Files with a usable (non-flat) accelerometer channel: {accel_usable_count} / {len(df)}",
        "",
        "## Per-file summary",
        "",
        df.to_markdown(index=False),
        "",
        "## Flag legend",
        "- `PLAUSIBLE`: mean HR in a sane resting/light-activity range and RR-interval",
        "  noise comparable to PhysioNet's clean recordings.",
        "- `NOISY_ECG`: HR plausible but too many physiologically-impossible RR intervals",
        "  -- suggests the QVAR ECG lead is too noisy for reliable R-peak detection here.",
        "- `IMPLAUSIBLE_HR`: RR intervals are internally consistent but the resulting mean",
        "  HR is outside a sane range -- likely a lead/contact issue, not true bradycardia",
        "  or tachycardia.",
        "- `IMPLAUSIBLE_HR_AND_NOISY`: both problems -- QVAR signal not usable as-is.",
        "- `TOO_SHORT`: fewer than 2 samples, an aborted capture.",
        "- `NO_BEATS_DETECTED` / `RPEAK_ERROR`: R-peak detector failed outright.",
        "",
        "## What this does and does not tell us",
        "- It tells us whether the QVAR channel, wired as ECG on our own hardware, yields",
        "  clean enough R-peaks for the existing PhysioNet-trained pipeline's R-peak stage",
        "  to work at all, and whether the accelerometer (effort branch) is alive in the",
        "  same files.",
        "- It does NOT tell us anything about apnea detection accuracy/sensitivity on our",
        "  hardware -- these are healthy subjects with no per-minute apnea labels, and per",
        "  CLAUDE.md this data is validation-only and is never used for training.",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    df = run_quality_report()
    print("\n=== SUMMARY ===")
    print(df[["file", "fs_measured_hz", "n_gaps", "accel_usable",
              "n_beats", "mean_hr_bpm", "pct_implausible_rr",
              "plausibility_flag"]].to_string(index=False))
