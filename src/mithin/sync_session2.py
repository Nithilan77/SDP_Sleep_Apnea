"""Session-2 (Mithin) timeline -> SensorTile alignment feasibility test.

Inputs: results/mithin_s2/timestamps_extracted.csv (laps read from the screen recording) and the three tile logs.
The stopwatch has no shared event with any tile, so the offset t0 (stopwatch zero in tile time) is UNKNOWN. This script
(a) writes the event table in stopwatch time, (b) searches every candidate offset for the one at which a per-second tile
feature differs most between the inferred hold windows and rest windows, and (c) compares the best score with a null built
from the same feature (block-shuffled), so a peak is only reported if it exceeds the null.
    venv/bin/python -m src.mithin.sync_session2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ingest.sensortile import load_sensortile  # noqa: E402

OUT = Path("results/mithin_s2")
D = Path("data/recordings/drive_2026-10-19")
TILES = {"ECG": D / "user_MithinBreathHold_ECG_000.csv",
         "EMG": D / "user_MithinBreathHold_EMG_000.csv",
         "EOG": Path.home() / "Desktop/Mithin-BreathHold/user_MithinBreathHold_EOG_000.csv"}
SEED = 0
MIN_HOLD_S = int(__import__("os").environ.get("MIN_HOLD_S", 60))   # minimum hold seconds that must fall inside the tile for an offset to be scored
TAG = "" if MIN_HOLD_S == 60 else f"_min{MIN_HOLD_S}"


def event_table() -> pd.DataFrame:
    laps = pd.read_csv(OUT / "timestamps_extracted.csv")
    end = laps.lap_time_s.cumsum()
    start = end - laps.lap_time_s
    role = ["setup_baseline"] + ["rest" if l % 2 == 0 else "hold" for l in laps.lap[1:]]
    role[-1] = "unknown_partial"
    ev = pd.DataFrame({"event_id": [f"L{l:02d}" for l in laps.lap], "role_inferred": role,
                       "stopwatch_start_s": start.round(2), "stopwatch_end_s": end.round(2),
                       "duration_s": laps.lap_time_s, "provenance": "durations observed; start/end calculated by cumulative sum; role inferred"})
    ev["hold_no"] = np.where(ev.role_inferred == "hold", (ev.index // 2), np.nan)
    return ev


def breathing_feature(rec, bin_s=1.0):
    """per-second breathing-band (0.1-0.6 Hz) accel envelope (g, gravity removed by band-pass) and its time axis."""
    fs = rec.fs_measured
    t = (rec.timestamp_us - rec.timestamp_us[0]) / 1e6
    acc = rec.accel_mg / 1000.0
    sos = butter(2, [0.1, 0.6], btype="band", fs=fs, output="sos")
    bp = np.sqrt(sum(sosfiltfilt(sos, acc[:, i]) ** 2 for i in range(3)))
    n = int(t[-1] // bin_s)
    idx = np.minimum((t / bin_s).astype(int), n - 1)
    f = np.bincount(idx, weights=bp, minlength=n) / np.maximum(np.bincount(idx, minlength=n), 1)
    return f


def score(f, ev, t0):
    hold = ev[ev.role_inferred == "hold"]; rest = ev[ev.role_inferred == "rest"]
    n = len(f)
    def sel(rows):
        m = np.zeros(n, bool)
        for a, b in zip(rows.stopwatch_start_s, rows.stopwatch_end_s):
            i, j = int(round(a + t0)), int(round(b + t0))
            m[max(i, 0):max(min(j, n), 0)] = True
        return m
    mh, mr = sel(hold), sel(rest)
    if mh.sum() < MIN_HOLD_S or mr.sum() < 60:
        return np.nan, int(mh.sum())
    lf = np.log(f + 1e-9)
    return float((lf[mr].mean() - lf[mh].mean()) / lf.std()), int(mh.sum())   # >0: lower breathing amplitude in holds


def main():
    OUT.mkdir(exist_ok=True)
    ev = event_table(); ev.to_csv(OUT / "event_table.csv", index=False)
    rng = np.random.default_rng(SEED); res = {}
    for name, p in TILES.items():
        rec = load_sensortile(p)
        f = breathing_feature(rec)
        offs = np.arange(-2014 + 120, len(f) - 120, 1)
        sc = np.array([score(f, ev, o)[0] for o in offs])
        best = int(np.nanargmax(sc))
        # null: same offset scan on block-shuffled (30 s blocks) versions of the feature; compare max statistics
        null_max = []
        for _ in range(200):
            blocks = [f[i:i + 30] for i in range(0, len(f), 30)]
            g = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
            s = np.array([score(g, ev, o)[0] for o in offs[::3]])
            null_max.append(np.nanmax(s))
        null_max = np.array(null_max)
        res[name] = {"duration_s": float(len(f)), "fs": rec.fs_measured, "best_offset_s": float(offs[best]),
                     "best_score": float(sc[best]), "null_max_p95": float(np.percentile(null_max, 95)),
                     "null_max_mean": float(null_max.mean()), "p_value_max_stat": float((null_max >= sc[best]).mean()),
                     "hold_seconds_overlapping_at_best": score(f, ev, offs[best])[1],
                     "n_candidate_offsets": int(len(offs))}
        print(name, res[name], flush=True)
        pd.DataFrame({"offset_s": offs, "score": sc}).to_csv(OUT / f"offset_scan_{name}{TAG}.csv", index=False)
    json.dump(res, open(OUT / f"sync_scan{TAG}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
