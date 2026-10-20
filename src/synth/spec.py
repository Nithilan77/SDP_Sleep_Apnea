"""Derive a measured specification of the in-house SensorTile data (in-house data; used here to set the simulator's
parameter RANGES and to check generated files; any model trained on data generated from this spec must record that the generator was fitted to these in-house recordings, see CLAUDE.md §4).  Distinguishes MEASURED values from simulator ASSUMPTIONS.

    python3 -m src.synth.spec          ->  results/synth/spec/real_data_spec.json, per_recording_table.csv
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt, welch, find_peaks
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ingest.sensortile import load_sensortile
from src.ecg.rpeaks import detect_rpeaks

import os
OUT = Path(os.environ.get("SYNTH_SPEC_DIR", "results/synth/spec")); H = Path.home() / "Desktop"
EXCLUDE = [x for x in os.environ.get("SYNTH_SPEC_EXCLUDE", "").split(",") if x]   # e.g. MukuBreathHold: fit the spec without the held-out session
FILES = sorted(Path("data/recordings/inhouse_ecg").glob("*.txt")) + [
    Path("data/recordings/breath_hold/user_MukuBreathHold_ECG_07_10_2026.csv"),
    H / "Mithin-BreathHold/user_MithinBreathHold_ECG_000.csv", H / "Mithin-BreathHold/user_MithinBreathHold_EMG_000.csv",
    H / "Mithin-BreathHold/user_MithinBreathHold_EOG_000.csv", H / "Mithin/user_Mithin10_ECG_000.csv"]
MIN_S = 600
# LSB->physical conversions read straight from the file columns (see check below), not assumed
COLS_RAW = ["time[us]", "acc_x[LSB]", "acc_y[LSB]", "acc_z[LSB]", "acc_x[mg]", "acc_y[mg]", "acc_z[mg]", "gyro_x[LSB]", "gyro_y[LSB]", "gyro_z[LSB]",
            "gyro_x[dsp]", "gyro_y[dsp]", "gyro_z[dsp]", "temp[LSB]", "temp[C]", "ah_qvar[LSB]"]


def bp(x, fs, lo, hi, o=3):
    return sosfiltfilt(butter(o, [lo, hi], btype="band", fs=fs, output="sos"), x)


def beat_template(q, fs, r_idx, half=0.35):
    h = int(half * fs); segs = [q[i - h:i + h + 1] - np.median(q[i - h:i + h + 1]) for i in r_idx[5:-5:3] if i - h >= 0 and i + h + 1 <= len(q)]
    segs = np.array(segs); t = np.median(segs, 0); return t, np.abs(t).max()


def analyse(p: Path) -> dict | None:
    r = load_sensortile(p)
    if r.duration_s < MIN_S: return None
    fs = r.fs_measured; t = (r.timestamp_us - r.timestamp_us[0]) / 1e6; dt = np.diff(r.timestamp_us)
    q = r.qvar.astype(float); row = dict(file_role=p.name if "Breath" in p.name or "Mithin10" in p.name else "inhouse_ecg_txt", duration_s=r.duration_s, fs_hz=fs,
        dt_us_median=float(np.median(dt)), dt_us_sd=float(dt.std()), n_gaps=int(r.n_gaps), qvar_sd=float(q.std()), qvar_p1=float(np.percentile(q, 1)), qvar_p99=float(np.percentile(q, 99)),
        qvar_unique=int(len(np.unique(q))), qvar_clip_frac=float(np.mean(q == q.max())) if np.mean(q == q.max()) > .01 else 0.0)
    # accelerometer
    a = r.accel_mg; mag = np.linalg.norm(a, axis=1)
    row.update(grav_mag_mg=float(np.median(mag)), acc_axis_sd_mg=[float(x) for x in a.std(0)], gyro_sd_dps=[float(x) for x in r.gyro_dps.std(0)],
               temp_c_min=float(r.temp_degC.min()), temp_c_max=float(r.temp_degC.max()))
    m = bp(mag - np.median(mag), fs, 0.08, 0.6); row["breath_band_rms_mg"] = float(m.std())
    hf = bp(a[:, 2], fs, 20, min(60, fs / 2 - 5)); row["acc_noise_rms_mg_20_60hz"] = float(hf.std())
    # breathing rate per 60 s windows with a clear peak
    rates, amps = [], []; W = int(60 * fs)
    for s in range(0, len(m) - W, W):
        f, P = welch(m[s:s + W], fs, nperseg=W // 2); b = (f >= .15) & (f <= .6)
        if b.any() and P[b].max() > 8 * np.median(P[(f >= .08) & (f <= .6)]) and f[b][P[b].argmax()] > 0.12:
            rates.append(float(f[b][P[b].argmax()])); amps.append(float(np.sqrt(P[b].max() * (f[1] - f[0]))))
    row.update(breath_windows_frac=len(rates) / max(len(m) // W, 1), breath_hz_median=float(np.median(rates)) if rates else None,
               breath_hz_p10=float(np.percentile(rates, 10)) if rates else None, breath_hz_p90=float(np.percentile(rates, 90)) if rates else None)
    # cardiac (only if the tile carries ECG)
    f, P = welch(q - q.mean(), fs, nperseg=int(8 * fs)); card = P[(f >= 0.8) & (f <= 3)].max() / np.median(P[(f >= 5) & (f <= 15)]); row["cardiac_prominence"] = float(card)
    if card > 20:
        k = detect_rpeaks(r); rr = k.rr_intervals_s; rr = rr[(rr > .3) & (rr < 2)]
        row.update(n_beats=int(len(k.r_peaks)), hr_bpm=float(60 / np.median(rr)), sdnn_ms=float(1e3 * rr.std()), rmssd_ms=float(1e3 * np.sqrt(np.mean(np.diff(rr) ** 2))),
                   rr_p1_s=float(np.percentile(rr, 1)), rr_p99_s=float(np.percentile(rr, 99)), r_amp_lsb_median=float(np.median(k.r_amplitudes)), r_amp_cv=float(np.std(k.r_amplitudes) / np.mean(k.r_amplitudes)))
        row["wander_sd_lsb"] = float(sosfiltfilt(butter(2, 0.5, "low", fs=fs, output="sos"), q).std())
        row["hf_noise_sd_lsb"] = float(bp(q, fs, 40, min(100, fs / 2 - 5)).std())
        tpl, amp = beat_template(q, fs, k.r_peaks); row["template_peak_lsb"] = float(amp)
        # QRS width: samples above half max / fs
        row["qrs_fwhm_ms"] = float(1e3 * (np.abs(tpl) > 0.5 * np.abs(tpl).max()).sum() / fs)
        np.save(OUT / f"template_{len(list(OUT.glob('template_*.npy')))}.npy", tpl / np.abs(tpl).max())
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("template_*.npy"): f.unlink()
    rows = []
    for p in FILES:
        if p.exists() and p.stat().st_size > 0 and not any(x in p.name for x in EXCLUDE):
            try:
                x = analyse(p)
            except Exception as e:
                x = dict(file_role=p.name, error=str(e))
            if x: x["source_index"] = len(rows); rows.append(x); print(len(rows), {k: (round(v, 2) if isinstance(v, float) else v) for k, v in list(x.items())[:6]}, flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "per_recording_table.csv", index=False)
    ecg = df[df.cardiac_prominence > 20]
    rng = lambda s, d=df: {"p5": float(np.nanpercentile(d[s].astype(float), 5)), "median": float(np.nanmedian(d[s].astype(float))), "p95": float(np.nanpercentile(d[s].astype(float), 95)), "n": int(d[s].notna().sum())}
    spec = {"n_recordings_analysed": len(df), "n_with_clean_ecg": len(ecg),
            "MEASURED": {"fs_hz": rng("fs_hz"), "dt_us_median": rng("dt_us_median"), "dt_us_sd": rng("dt_us_sd"), "hr_bpm": rng("hr_bpm", ecg), "sdnn_ms": rng("sdnn_ms", ecg), "rmssd_ms": rng("rmssd_ms", ecg),
                         "r_amp_lsb": rng("r_amp_lsb_median", ecg), "r_amp_cv": rng("r_amp_cv", ecg), "wander_sd_lsb": rng("wander_sd_lsb", ecg), "hf_noise_sd_lsb": rng("hf_noise_sd_lsb", ecg),
                         "qrs_fwhm_ms": rng("qrs_fwhm_ms", ecg), "qvar_sd_lsb": rng("qvar_sd", ecg), "grav_mag_mg": rng("grav_mag_mg"), "breath_band_rms_mg": rng("breath_band_rms_mg"),
                         "acc_noise_rms_mg": rng("acc_noise_rms_mg_20_60hz"), "breath_hz_median": rng("breath_hz_median"), "breath_windows_frac": rng("breath_windows_frac"),
                         "temp_c": {"min": float(df.temp_c_min.min()), "max": float(df.temp_c_max.max())}},
            "FILE_FORMAT": {"delimiter": ",", "header": COLS_RAW, "dtypes": "time int µs; LSB columns int; physical columns float (3 decimals); temp[C] 3 decimals",
                            "conversions_from_files": {"acc_mg_per_lsb": 0.061, "gyro_dps_per_lsb": 0.004375, "temp_c": "25 + temp[LSB]/256"},
                            "truncated_last_row_in_s2_logs": True, "qvar_quantisation_lsb": 4, "qvar_quantisation_note": "all values of both ECG tiles are multiples of 4 (gcd of differences = 4); ~410-495 distinct values per 60 s"},
            "ASSUMED_BY_SIMULATOR (not measured)": ["respiratory modulation of ECG amplitude (EDR) depth", "hold pause kinematics (ramp-down/up time constants)", "motion-artefact burst statistics",
                "EMG and EOG tile QVAR content (real EMG/EOG-labelled tiles show no recognisable biosignal; simulated as flat/clipped per the audit)", "gyroscope breathing coupling",
                "inter-recording variation of electrode placement/orientation (only 1 session per subject available)"]}
    json.dump(spec, open(OUT / "real_data_spec.json", "w"), indent=1); print(json.dumps(spec["MEASURED"], indent=0)[:3000])


if __name__ == "__main__":
    main()
