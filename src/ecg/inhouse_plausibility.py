"""
Frozen-model plausibility test (negative control) on in-house QVAR recordings.

Runs the frozen deployable ECG-branch model (results/phase3_cnn/deployable/,
the 2-channel RR-interval + R-peak-amplitude RRCNN trained ONLY on PhysioNet)
over the plausible in-house SensorTile recordings. These are healthy 19-20
year-olds with no apnea labels -- this is NOT an accuracy measurement, it is
a sanity/negative-control check: a model that works should report a LOW
apnea-minute rate on healthy subjects.

Also runs an RR-ONLY variant (amplitude channel neutralized to its training
mean, i.e. zero information after normalization) to isolate how much of any
implausible result comes from the untransferable amplitude channel
(see results/inhouse_validation/domain_gap/) vs the RR channel alone.

There are no per-minute apnea labels for this data -- per CLAUDE.md this data
is validation-only and was never used to train anything. A high predicted
apnea rate here is a signal to investigate the amplitude-channel domain gap,
NOT a result to report as the model's real-world performance.

Usage:
    python src/ecg/inhouse_plausibility.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ecg.cnn_model import RRCNN, predict_cnn  # noqa: E402
from src.ecg.features import MIN_BEATS_PER_WINDOW, WINDOW_S  # noqa: E402
from src.ecg.rpeaks import detect_rpeaks  # noqa: E402
from src.ecg.rr_sequence import SEQ_LEN, _resample_to_grid  # noqa: E402
from src.ingest.sensortile import load_sensortile  # noqa: E402

DEPLOYABLE_DIR = Path("results/phase3_cnn/deployable")
INHOUSE_DIR = Path("data/recordings/inhouse_ecg")
QUALITY_CSV = Path("results/inhouse_validation/quality_report.csv")
RESULTS_DIR = Path("results/inhouse_validation/plausibility")

APNEA_PROB_THRESHOLD = 0.5


def _build_unlabelled_minute_sequences(rec, rpk, seq_len: int = SEQ_LEN):
    """Same per-minute windowing/resampling as rr_amp_sequence.py, but chunks
    the recording by elapsed time instead of aligning to .apn labels (there
    are none). Returns (sequences (n,2,seq_len), minute_index, n_beats)."""
    fs = rec.fs_measured
    r_peaks = rpk.r_peaks
    peak_times_s = r_peaks / fs
    rr_s = rpk.rr_intervals_s
    rr_times_s = peak_times_s[1:]
    amp = rpk.r_amplitudes[1:]

    n_minutes = int(rec.duration_s // WINDOW_S)
    sequences, minutes_kept, n_beats_list = [], [], []
    for minute in range(n_minutes):
        t_start, t_end = minute * WINDOW_S, (minute + 1) * WINDOW_S
        mask = (rr_times_s >= t_start) & (rr_times_s < t_end)
        rr_win, amp_win = rr_s[mask], amp[mask]
        if len(rr_win) < MIN_BEATS_PER_WINDOW:
            continue
        rr_grid = _resample_to_grid(rr_win, seq_len)
        amp_grid = _resample_to_grid(amp_win, seq_len)
        sequences.append(np.stack([rr_grid, amp_grid], axis=0))
        minutes_kept.append(minute)
        n_beats_list.append(len(rr_win))

    if not sequences:
        return (np.zeros((0, 2, seq_len), dtype=np.float32),
                np.array([], dtype=int), np.array([], dtype=int))
    return np.stack(sequences), np.array(minutes_kept), np.array(n_beats_list)


def _load_model_and_norm():
    with open(DEPLOYABLE_DIR / "norm_stats.json") as f:
        norm = json.load(f)
    mu = np.array(norm["mean"], dtype=np.float32).reshape(1, 2, 1)
    sd = np.array(norm["std"], dtype=np.float32).reshape(1, 2, 1)

    model = RRCNN(seq_len=SEQ_LEN, n_channels=2)
    state = torch.load(DEPLOYABLE_DIR / "model_state_dict.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, mu, sd


def run_plausibility() -> pd.DataFrame:
    qdf = pd.read_csv(QUALITY_CSV)
    plausible_files = qdf.loc[qdf["plausibility_flag"] == "PLAUSIBLE", "file"].tolist()
    print(f"Plausible in-house files ({len(plausible_files)}): {plausible_files}")

    model, mu, sd = _load_model_and_norm()

    rows = []
    for fname in plausible_files:
        print(f"\n--- {fname} ---")
        rec = load_sensortile(INHOUSE_DIR / fname)
        rpk = detect_rpeaks(rec)
        seqs, minutes, n_beats = _build_unlabelled_minute_sequences(rec, rpk)

        if len(seqs) == 0:
            print("  no minutes with enough beats -- skipping")
            rows.append({"file": fname, "n_minutes": 0,
                         "apnea_pct_2ch": np.nan, "mean_prob_2ch": np.nan,
                         "apnea_pct_rronly": np.nan, "mean_prob_rronly": np.nan})
            continue

        # 2-channel: RR + R-amplitude, normalized with the model's own training stats.
        X_2ch = (seqs - mu) / sd
        probs_2ch = predict_cnn(model, X_2ch)

        # RR-only variant: neutralize the amplitude channel to its training
        # mean BEFORE normalizing, so it becomes exactly 0 post-normalization
        # (zero information), isolating the RR channel's own contribution.
        seqs_rronly = seqs.copy()
        seqs_rronly[:, 1, :] = mu[0, 1, 0]  # raw amplitude <- training mean
        X_rronly = (seqs_rronly - mu) / sd
        probs_rronly = predict_cnn(model, X_rronly)

        apnea_pct_2ch = 100.0 * float(np.mean(probs_2ch >= APNEA_PROB_THRESHOLD))
        apnea_pct_rronly = 100.0 * float(np.mean(probs_rronly >= APNEA_PROB_THRESHOLD))
        print(f"  n_minutes={len(seqs)}  2ch: apnea%={apnea_pct_2ch:.1f} mean_prob={probs_2ch.mean():.3f}"
              f"  |  RR-only: apnea%={apnea_pct_rronly:.1f} mean_prob={probs_rronly.mean():.3f}")

        rows.append({
            "file": fname,
            "n_minutes": len(seqs),
            "apnea_pct_2ch": round(apnea_pct_2ch, 2),
            "mean_prob_2ch": round(float(probs_2ch.mean()), 4),
            "apnea_pct_rronly": round(apnea_pct_rronly, 2),
            "mean_prob_rronly": round(float(probs_rronly.mean()), 4),
        })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "per_subject_predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    _write_readme(df, RESULTS_DIR / "README.md")
    return df


def _write_readme(df: pd.DataFrame, path: Path) -> None:
    valid = df[df["n_minutes"] > 0]
    total_minutes = int(valid["n_minutes"].sum())
    overall_2ch = float((valid["apnea_pct_2ch"] * valid["n_minutes"]).sum() / total_minutes) if total_minutes else float("nan")
    overall_rronly = float((valid["apnea_pct_rronly"] * valid["n_minutes"]).sum() / total_minutes) if total_minutes else float("nan")

    lines = [
        "# Frozen-model plausibility test (negative control) on in-house QVAR data",
        "",
        "**Not an accuracy measurement.** These are healthy 19-20yo subjects with no",
        "apnea labels (validation-only per CLAUDE.md Sec.4). The frozen deployable model",
        "(`results/phase3_cnn/deployable/`, trained ONLY on PhysioNet) is run here purely",
        "as a negative control: a model that transfers should report a LOW apnea-minute",
        "rate on healthy subjects. A high rate is a red flag to investigate -- most likely",
        "the R-peak-amplitude channel's domain gap documented in",
        "`results/inhouse_validation/domain_gap/` -- not a finding about real apnea.",
        "",
        f"Overall predicted apnea-minute rate across {total_minutes} minutes "
        f"({len(valid)} subjects): **2-channel (RR+amplitude) = {overall_2ch:.1f}%**, "
        f"**RR-only (amplitude neutralized) = {overall_rronly:.1f}%**.",
        "",
        "## Per-subject predictions",
        "",
        df.to_markdown(index=False),
        "",
        "## How to read this",
        "- `apnea_pct_2ch` / `mean_prob_2ch`: the full frozen model (RR + R-amplitude), "
        "exactly as trained.",
        "- `apnea_pct_rronly` / `mean_prob_rronly`: same model and same RR channel, but the "
        "amplitude channel's raw value is replaced with the model's own training-mean "
        "amplitude before normalization -- so that channel carries exactly zero "
        "information (it normalizes to 0). This isolates what the RR channel alone is "
        "driving vs what the (likely out-of-distribution) amplitude channel is driving.",
        "- If `apnea_pct_2ch` is much higher than `apnea_pct_rronly`, the amplitude channel "
        "is the dominant cause of any implausible (high) apnea rate -- consistent with the "
        "domain-gap analysis if that showed a large PhysioNet-vs-QVAR amplitude scale ratio.",
        "- If both are similarly high, the RR channel itself (or the model's threshold/"
        "calibration in general) is also part of the story, not just amplitude scale.",
        "",
        "## Cross-reference",
        "See `results/inhouse_validation/domain_gap/README.md` for the measured "
        "PhysioNet-vs-QVAR amplitude and QRS-area scale ratios that this test's RR-only "
        "vs 2-channel comparison is meant to explain.",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    df = run_plausibility()
    print("\n=== SUMMARY ===")
    print(df.to_string(index=False))
