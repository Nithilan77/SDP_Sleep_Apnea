"""Automated validation of a generated SensorTile-format recording against the measured schema and physiology.
    python3 -m src.synth.validate <recording.csv> [truth.json]
Returns a dict {check: (passed, value, rule)}; a recording is `valid` iff every REQUIRED check passes. Independent of the generator (uses only the file + truth)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from src.synth.render import COLUMNS, MG_PER_LSB, DPS_PER_LSB


def _bp(x, fs, lo, hi):
    return sosfiltfilt(butter(3, [lo, hi], btype="band", fs=fs, output="sos"), x)


def validate(df: pd.DataFrame, truth: dict | None = None, tile: str = "ECG") -> dict:
    C = {}; add = lambda k, ok, v, rule, req=True: C.__setitem__(k, dict(passed=bool(ok), value=v, rule=rule, required=req))
    add("columns", list(df.columns) == COLUMNS, list(df.columns)[:3], "16 columns in SensorTile order")
    if list(df.columns) != COLUMNS: return dict(valid=False, checks=C)
    n = len(df); t = df[COLUMNS[0]].to_numpy(np.int64); dt = np.diff(t)
    add("no_nan", not df.isna().any().any(), int(df.isna().sum().sum()), "no missing values")
    add("timestamps_strictly_increasing", (dt > 0).all(), int((dt <= 0).sum()), "monotonic, no duplicates")
    fs = (n - 1) / ((t[-1] - t[0]) / 1e6); add("fs_range", 230 <= fs <= 250, round(fs, 2), "measured fs 230-250 Hz (measured in-house 238-243.2)")
    add("timestamp_jitter", np.abs(dt - np.median(dt)).max() <= 5, int(np.abs(dt - np.median(dt)).max()), "max |dt - median| <= 5 us (in-house SD 0.3-0.5 us)")
    add("start_offset", 0 < t[0] < 10000, int(t[0]), "first timestamp 2-4 ms in real files")
    a_lsb = df[COLUMNS[1:4]].to_numpy(); a_mg = df[COLUMNS[4:7]].to_numpy()
    add("accel_unit_consistency", np.abs(a_mg - a_lsb * MG_PER_LSB).max() < 0.002, float(np.abs(a_mg - a_lsb * MG_PER_LSB).max()), "mg = LSB x 0.061")
    g = np.linalg.norm(a_mg, axis=1); add("gravity_magnitude", 940 <= np.median(g) <= 1050, round(float(np.median(g)), 1), "median |a| 940-1050 mg (in-house 984-1005)")
    add("accel_saturation", (np.abs(a_mg) >= 1999).mean() < 0.001, float((np.abs(a_mg) >= 1999).mean()), "<0.1% samples at +-2 g")
    gy = df[COLUMNS[10:13]].to_numpy(); add("gyro_unit_consistency", np.abs(gy - df[COLUMNS[7:10]].to_numpy() * DPS_PER_LSB).max() < 0.002, 0, "dps = LSB x 0.004375")
    tc = df[COLUMNS[14]].to_numpy(); add("temperature_range", 15 <= tc.min() and tc.max() <= 45, [round(float(tc.min()), 1), round(float(tc.max()), 1)], "15-45 degC (in-house 22.7-40.1)")
    q = df[COLUMNS[15]].to_numpy(); add("qvar_int16", q.min() >= -32768 and q.max() <= 32767, [int(q.min()), int(q.max())], "fits int16")
    add("qvar_quantisation", bool((q % 4 == 0).all()) or tile != "ECG", float((q % 4 == 0).mean()), "ECG-tile QVAR values are multiples of 4 LSB (measured on real tiles)", req=tile == "ECG")
    run = 0; best = 0
    for v in (np.diff(q) == 0): run = run + 1 if v else 0; best = max(best, run)
    add("no_flatline", best / fs < 2.0 or tile != "ECG", round(best / fs, 2), "no constant QVAR run > 2 s on an ECG tile", req=tile == "ECG")
    add("no_clipping", tile != "ECG" or (q == q.max()).mean() < 0.01, float((q == q.max()).mean()), "<1% samples at max on an ECG tile", req=tile == "ECG")
    if tile == "ECG":
        from src.ingest.sensortile import SensorTileRecord
        from src.ecg.rpeaks import detect_rpeaks
        rec = SensorTileRecord(name="synthetic", n_samples=n, timestamp_us=t.astype(float), accel_mg=a_mg, gyro_dps=gy, temp_degC=tc, qvar=q.astype(float), fs_measured=fs, duration_s=(t[-1] - t[0]) / 1e6)
        k = detect_rpeaks(rec); rr = k.rr_intervals_s; hr = 60 / np.median(rr)
        add("heart_rate_plausible", 40 <= hr <= 120, round(float(hr), 1), "median HR 40-120 bpm")
        add("rr_plausible_frac", (np.mean((rr < .3) | (rr > 2))) < 0.05, float(np.mean((rr < .3) | (rr > 2))), "<5% implausible RR (real in-house: 0-0.02%; noisy profiles may degrade the detector)")
        if truth:
            nb = len(truth["beat_times_s"]); add("beat_count_vs_truth", abs(len(k.r_peaks) - nb) <= 0.08 * nb, [len(k.r_peaks), nb], "detected beats within 8% of simulated beats (detector mis-detections of T-waves/ectopics exist)")
    m = _bp(np.linalg.norm(a_mg, axis=1) - np.median(g), fs, 0.08, 0.6)
    if truth:
        P = truth["profile"]; W = int(60 * fs); hold = np.zeros(n, bool)
        for e in truth["events"]: hold[e["start_sample"]:e["end_sample"]] = True
        bl = [s for s in range(0, n - W, W) if not hold[max(s - 5 * W // 60, 0):s + W].any()]
        if bl and P["breath_amp_mg"] > 0.5:
            f, Pw = welch(m[bl[0]:bl[0] + W], fs, nperseg=W // 2); b = (f >= .1) & (f <= .6); fpk = f[b][Pw[b].argmax()]
            add("breathing_freq_matches", abs(fpk - P["breath_hz"]) < 0.06, [round(float(fpk), 3), round(P["breath_hz"], 3)], "accel breathing peak within 0.06 Hz of simulated rate (only if breath amp > 0.5 mg)", req=False)
        env = np.abs(m); rat = []
        for e in truth["events"]:
            a, b = e["start_sample"], e["end_sample"]
            if e["kind"] in (1, 2) and b - a > int(8 * fs) and a > int(20 * fs):
                rat.append(np.sqrt(np.mean(m[a + int(2 * fs):b - int(1 * fs)] ** 2)) / (np.sqrt(np.mean(m[a - int(15 * fs):a - int(1 * fs)] ** 2)) + 1e-9))
        if rat and P["breath_amp_mg"] > 0.5 and P["quality"] != "poor_breathing":
            add("hold_effort_drop", np.median(rat) < 0.7, round(float(np.median(rat)), 3), "median effort ratio (hold/pre) < 0.7 where breathing is visible (> 0.5 mg)", req=False)
    valid = all(v["passed"] for v in C.values() if v["required"])
    return dict(valid=valid, checks=C, fs=round(fs, 3), n=n)


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1]); tr = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
    r = validate(df, tr); print(json.dumps({k: v for k, v in r.items()}, indent=1, default=str)[:3000]); sys.exit(0 if r["valid"] else 1)
