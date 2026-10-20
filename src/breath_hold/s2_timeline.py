"""Mithin session-2 timestamp reconstruction + evidence-based synchronisation audit.

Inputs (read-only): MithinBreathHold-TimeStamps.mp4 (phone stopwatch screen-recording, lap list
transcribed by reading its frames) and the three SensorTile logs (+ the separate Desktop 'Mithin10' pair).

What is DIRECTLY OBSERVED  : lap durations, stopwatch total, phone status-bar clock (laps_extracted.csv)
What is CALCULATED         : cumulative lap boundaries (sums of the observed durations)
What is INFERRED           : which laps are holds (30 s laps; lap 3,5,...,15) -- NOT stated in the video
What is TESTED here        : whether any tile's accelerometer effort envelope is quiet at the lap-derived
                             hold windows, at some offset, more than at the other offsets (shift null).
Nothing here edits a signal or creates an event label from the signal.

    python3 -m src.breath_hold.s2_timeline
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

NPERM = 300
OUT = Path("results/breath_hold_s2/mithin")
VIDEO = Path("data/recordings/drive_2026-10-19/MithinBreathHold-TimeStamps.mp4")
D = Path.home() / "Desktop"
TILES = {
    "ECG": D / "Mithin-BreathHold/user_MithinBreathHold_ECG_000.csv",
    "EMG": D / "Mithin-BreathHold/user_MithinBreathHold_EMG_000.csv",
    "EOG": D / "Mithin-BreathHold/user_MithinBreathHold_EOG_000.csv",   # complete file (Drive copy is truncated)
}
# Transcribed from the video frames (read at full resolution, 3 scroll positions each; checked twice).
LAPS = [1080.84, 89.18, 30.13, 89.86, 30.06, 89.87, 30.11, 89.86, 30.14, 89.99, 29.98, 89.91,
        30.06, 90.01, 29.94, 90.06, 3.96]
STOPWATCH_TOTAL_S = 33 * 60 + 34.04          # displayed, stopped
HOLD_LAPS = list(range(3, 16, 2))            # INFERRED: the seven ~30 s laps


def lap_table() -> pd.DataFrame:
    end = np.cumsum(LAPS)
    start = end - np.array(LAPS)
    t = pd.DataFrame({"lap": np.arange(1, len(LAPS) + 1), "lap_time_s_observed": LAPS,
                      "start_s_calculated": start.round(2), "end_s_calculated": end.round(2)})
    t["inferred_role"] = ["lead-in (18 min; role not stated)"] + [
        ("HOLD %d (inferred)" % (HOLD_LAPS.index(i) + 1)) if i in HOLD_LAPS else "rest (inferred)"
        for i in range(2, 17)] + ["trailing partial lap"]
    t["source"] = "video frames; stopwatch lap list"
    return t


def hold_windows() -> np.ndarray:
    t = lap_table()
    return t[t.lap.isin(HOLD_LAPS)][["start_s_calculated", "end_s_calculated"]].to_numpy()


def load_effort_env(path: Path, env_fs=1.0):
    df = pd.read_csv(path, usecols=["time[us]", "acc_x[mg]", "acc_y[mg]", "acc_z[mg]"]).dropna()
    t = df["time[us]"].to_numpy(float) / 1e6
    fs = (len(t) - 1) / (t[-1] - t[0])
    a = df[["acc_x[mg]", "acc_y[mg]", "acc_z[mg]"]].to_numpy(float)
    mag = np.linalg.norm(a, axis=1)
    sos = butter(3, [0.08, 0.6], btype="band", fs=fs, output="sos")
    x = sosfiltfilt(sos, mag - np.median(mag))
    env = np.abs(hilbert(x))
    n = int(env_fs * (t[-1] - t[0]))
    e = np.array([env[int(i * fs / env_fs):int((i + 1) * fs / env_fs)].mean() for i in range(n)])
    return e, fs, t[-1] - t[0], len(t)


def scan(env: np.ndarray, holds: np.ndarray, flank=60.0, margin=3.0):
    """For each offset tau (recording time of stopwatch zero), score = median log env(hold) - median log env(flanks).
    Hold windows are inset by `margin` s (stopwatch lag/reaction uncertainty). Needs >= 5 of 7 holds fully covered."""
    L = len(env); le = np.log(env + 1e-9)
    taus = np.arange(-int(holds[-1, 1]) + 60, L - int(holds[0, 0]) - 60)
    sc = np.full(len(taus), np.nan)
    for k, tau in enumerate(taus):
        d = []
        for a, b in holds:
            ha, hb = int(round(a + tau + margin)), int(round(b + tau - margin))
            fa, fb = max(0, int(round(a + tau - flank))), min(L, int(round(b + tau + flank)))
            if ha < 0 or hb > L or fa > ha - 5 or fb < hb + 5:
                continue
            fl = np.r_[le[fa:ha - 5], le[hb + 5:fb]]
            if len(fl) < 20:
                continue
            d.append(np.median(le[ha:hb]) - np.median(fl))
        if len(d) >= 5:
            sc[k] = np.mean(d)
    return taus, sc


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lt = lap_table()
    lt.to_csv(OUT / "laps_extracted.csv", index=False)
    meta = {
        "source_file": str(VIDEO), "video": {"duration_s": 13.6, "fps": 53.98, "frames": 735, "size": "384x832"},
        "what_the_video_shows": "iPhone Stopwatch screen recording, stopwatch STOPPED at 33:34.04; the lap list is scrolled "
                                "from Lap 17 to Lap 1. Phone status-bar clock 12:55 (the time the video was filmed, not a session "
                                "clock). Battery 98%->97%. No sensor data, taps or hold labels are visible.",
        "directly_observed": {"laps_s": LAPS, "stopwatch_total_s": STOPWATCH_TOTAL_S, "slowest_lap": 1, "fastest_lap": 15},
        "calculated": {"sum_of_laps_s": round(sum(LAPS), 2), "total_minus_sum_s": round(STOPWATCH_TOTAL_S - sum(LAPS), 2),
                       "note": "0.08 s: total display vs lap sum; consistent with rounding / running lap 17 captured before the stop."},
        "inferred_NOT_in_video": {
            "hold_laps": HOLD_LAPS, "basis": "seven laps of ~30 s alternate with ~90 s laps; the same 30 s/90 s structure as session 1 "
            "(v1 protocol), but session 1 labelled laps 2,4,..14 as holds because its lap 1 was a short lead-in. Here lap 1 is "
            "18:00.84, so lap parity is shifted. Which laps are holds is an assumption until a signal check supports it.",
            "protocol_v2_followed": False,
            "protocol_v2_evidence": "v2 specifies 9 holds (5-60 s) and 20 laps with Lap 1 = sync taps; the video has 17 laps, 7 equal holds -> v1-style schedule",
        },
        "unknowns": ["stopwatch start time vs sensor start (no sync taps found in ECG tile)", "clock/timezone of the stopwatch",
                     "what lap 1 (18 min) was spent on", "whether the 3.96 s lap 17 is real data or a stop-lag"],
    }
    (OUT / "timestamps_extracted.json").write_text(json.dumps(meta, indent=1))

    holds = hold_windows()
    res = {}
    for name, p in TILES.items():
        env, fs, dur, n = load_effort_env(p)
        taus, sc = scan(env, holds)
        valid = np.isfinite(sc)
        i = int(np.nanargmin(sc))
        z = (np.nanmean(sc) - sc[i]) / np.nanstd(sc)
        # best score within the physically-plausible start-lag window (tile started within 0-120 s of stopwatch)
        res[name] = {"duration_s": round(dur, 1), "fs_hz": round(fs, 2), "rows": n,
                     "n_offsets": int(valid.sum()), "best_tau_s": int(taus[i]), "best_score": round(float(sc[i]), 3),
                     "null_mean": round(float(np.nanmean(sc)), 3), "null_sd": round(float(np.nanstd(sc)), 3),
                     "z_best_vs_shift_null": round(float(z), 2),
                     "frac_offsets_as_good": round(float(np.nanmean(sc <= sc[i])), 4)}
        pd.DataFrame({"tau_s": taus, "score_log_env_hold_minus_flank": sc}).to_csv(OUT / f"scan_{name}.csv", index=False)
        # max-statistic null: permute 30 s blocks of the envelope (keeps its marginal distribution and short-range
        # autocorrelation, destroys the 30 s/120 s schedule), repeat the WHOLE offset scan, keep the best score.
        rng = np.random.default_rng(20261020)
        B = 30
        nb = len(env) // B
        mins = []
        for _ in range(NPERM):
            perm = rng.permutation(nb)
            e2 = np.concatenate([env[j * B:(j + 1) * B] for j in perm])
            mins.append(np.nanmin(scan(e2, holds)[1]))
        mins = np.array(mins)
        res[name]["perm_n"] = NPERM
        res[name]["perm_p_best_score"] = round(float((1 + np.sum(mins <= sc[i])) / (1 + NPERM)), 4)
        res[name]["perm_null_best_median"] = round(float(np.median(mins)), 3)
        print(name, res[name])
    (OUT / "sync_scan_summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
