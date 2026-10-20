"""Render a SensorTile.box PRO-format recording (16 columns) from the beat-level simulator.

Parameter RANGES come from the measured in-house specification (results/synth/spec/real_data_spec.json, per-recording table, beat templates);
see SPEC_ASSUMED for what is not measured.  Ground truth (events, sync taps, beat times) is taken from the schedule, never recovered from the signal.
Deterministic given (profile, seed).
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import butter, sosfilt
from src.synth.physio import Priors, simulate_batch, HOLD, GEN_VERSION

import os
SPEC_DIR = Path(os.environ.get("SYNTH_SPEC_DIR", "results/synth/spec"))
COLUMNS = ["time[us]", "acc_x[LSB]", "acc_y[LSB]", "acc_z[LSB]", "acc_x[mg]", "acc_y[mg]", "acc_z[mg]", "gyro_x[LSB]", "gyro_y[LSB]", "gyro_z[LSB]",
           "gyro_x[dsp]", "gyro_y[dsp]", "gyro_z[dsp]", "temp[LSB]", "temp[C]", "ah_qvar[LSB]"]
MG_PER_LSB, DPS_PER_LSB = 0.061, 0.004375
QVAR_STEP = 4              # measured on both ECG tiles: gcd of sample differences = 4, 100% of values % 4 == 0
ACC_FS_MG = 2000.0          # +-2 g full scale (protocol note: accelerometer saturates above 2 g)


@dataclass
class Profile:
    name: str; seed: int; duration_s: float = 1800.0; tile: str = "ECG"       # ECG | EMG | EOG tile role
    fs_hz: float = 243.1; hr_rr0: float = 0.9; hrv_sd_s: float = 0.045
    r_amp_lsb: float = 250.0; template_idx: int = 0; template_stretch: float = 1.0
    wander_sd_lsb: float = 130.0; hf_noise_lsb: float = 20.0; qvar_offset_lsb: float = -100.0
    breath_hz: float = 0.3; breath_amp_mg: float = 0.8; acc_noise_mg: float = 2.4; gravity_mg: float = 995.0; tilt_deg: float = 10.0
    motion_rate_per_s: float = 1 / 400; temp_c0: float = 24.0; temp_c1: float = 33.0
    events: list = field(default_factory=list)       # [(start_s, end_s, kind)]  kind: 1 hold / 2 central / 3 obstructive
    sync_taps_s: list = field(default_factory=list)  # double-tap times (first tap of each pair)
    tap_peak_mg: float = 1200.0; quality: str = "good"


SPEC_ASSUMED = ["EDR depth", "hold ramp-down/up kinematics and recovery gasp", "motion-artefact statistics", "gyro breathing coupling", "50 Hz mains tone",
                "EMG/EOG tile QVAR morphology (real tiles uninformative)", "temperature time-course"]


def _spec():
    t = pd.read_csv(SPEC_DIR / "per_recording_table.csv"); return t, [np.load(f) for f in sorted(SPEC_DIR.glob("template_*.npy"))]


def sample_profile(name: str, seed: int, duration_s: float = 1800.0, tile: str = "ECG", holds: list | None = None, taps: bool = False, quality: str | None = None) -> Profile:
    """Draw a recording profile from ranges measured on the in-house data (p5-p95, slightly widened) - NOT a copy of one session."""
    rng = np.random.default_rng(seed); tab, tpl = _spec(); ecg = tab[tab.cardiac_prominence > 20]; lg = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    ql = quality or rng.choice(["good", "good", "good", "noisy", "poor_breathing"])
    p = Profile(name=name, seed=seed, duration_s=duration_s, tile=tile, quality=str(ql))
    p.fs_hz = float(rng.choice([rng.uniform(242.6, 243.2)] * 9 + [rng.uniform(238.0, 238.2)]))
    p.hr_rr0 = float(60 / rng.uniform(46, 82)); p.hrv_sd_s = lg(0.025, 0.09)
    p.template_idx = int(rng.integers(len(tpl))); p.template_stretch = float(rng.uniform(0.85, 1.2))
    ecg2 = ecg[ecg.template_peak_lsb / ecg.hf_noise_sd_lsb >= 3]            # drop the two tiles whose own R-peak SNR < 3 (detector fails on them in reality)
    vals = ecg2.template_peak_lsb.dropna().to_numpy(); p.r_amp_lsb = float(rng.choice(vals) * np.exp(rng.uniform(np.log(0.8), np.log(1.25))))   # empirical bootstrap of measured R amplitudes (+-20% jitter): preserves the measured, heavy-tailed distribution
    snr = lg(3.5, 8) if ql == "noisy" else lg(8, 80); p.hf_noise_lsb = float(np.clip(p.r_amp_lsb / snr, 13, 60))   # joint amp/noise: real SNR 3-85
    p.wander_sd_lsb = lg(40, 380); p.qvar_offset_lsb = float(rng.normal(-100, 150))
    p.breath_hz = float(rng.uniform(0.17, 0.45)); p.breath_amp_mg = lg(0.13, 2.4) * (0.25 if ql == "poor_breathing" else 1.0)     # measured band-RMS range 0.13-2.4 mg (28x between sessions)
    p.acc_noise_mg = lg(1.5, 5.4) / 0.57   # measured value is the 20-60 Hz band SD (~0.57 x white sigma)
    p.gravity_mg = float(rng.uniform(985, 1005)); p.tilt_deg = float(rng.uniform(0, 25))
    p.motion_rate_per_s = 1 / lg(150, 900); p.temp_c0 = float(rng.uniform(22.7, 28)); p.temp_c1 = float(rng.uniform(31, 37))
    p.events = [tuple(h) for h in (holds or [])]; p.sync_taps_s = [5.0, duration_s - 25.0] if taps else []
    p.tap_peak_mg = float(rng.uniform(500, 1500))
    return p


def _lpnoise(rng, n, fs, fc, order=2):
    sos = butter(order, fc, "low", fs=fs, output="sos"); x = sosfilt(sos, rng.normal(0, 1, n)); return x / (x.std() + 1e-12)


def render(p: Profile) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(p.seed); T = p.duration_s; fs = p.fs_hz
    n = int(T * fs); dt_us = 1e6 / fs; t_us = np.round(rng.integers(2000, 4500) + np.arange(n) * dt_us + rng.normal(0, 0.4, n).cumsum() * 0).astype(np.int64)
    t = (t_us - t_us[0]) / 1e6
    pr = Priors(); pr.log_rr0_mean = np.log(p.hr_rr0); pr.log_rr0_sd = 1e-6; pr.log_hrv_mean = np.log(p.hrv_sd_s); pr.log_hrv_sd = 1e-6
    pr.log_amp_mean = 0.0; pr.log_amp_sd = 1e-6; pr.breath_hz = (p.breath_hz, p.breath_hz); pr.p_gap_minute = 0.0
    forced = [[(a, b, c) for a, b, c in p.events]] if p.events else None
    br = simulate_batch(rng, 1, T, pr, severity=0 if forced is None else None, forced=forced if forced else [[]], voluntary=all(c == HOLD for _, _, c in p.events) if p.events else False)
    bt, amp_rel, ph = br.beat_t[0], br.amp[0], br.resp_phase_fn[0]
    # --- QVAR / ECG ----------------------------------------------------------------------------------------------------------------
    tpl_all = _spec()[1]; tpl = tpl_all[p.template_idx % len(tpl_all)]
    half = (len(tpl) - 1) / 2 / 243.0 * p.template_stretch                                  # template seconds half-width (templates were taken at ~243 Hz)
    q = np.zeros(n)
    if p.tile == "ECG":
        hw = int(half * fs) + 1; xs = (np.arange(-hw, hw + 1) / fs) / p.template_stretch * 243.0 + (len(tpl) - 1) / 2
        seg = np.interp(xs, np.arange(len(tpl)), tpl, left=0, right=0)
        for tb, ar in zip(bt, amp_rel):
            i = int(round(tb * fs))
            if i - hw < 0 or i + hw + 1 > n: continue
            q[i - hw:i + hw + 1] += ar * p.r_amp_lsb * seg
        resp_w = np.sin(np.interp(t, bt, ph, left=ph[0], right=ph[-1])) * 0.5
        q += p.wander_sd_lsb * (0.8 * _lpnoise(rng, n, fs, 0.35) + 0.4 * resp_w)
        q += rng.normal(0, p.hf_noise_lsb / 0.7, n) + rng.uniform(0, 8) * np.sin(2 * np.pi * 50 * t + rng.uniform(0, 6.28))
        q += p.qvar_offset_lsb
    elif p.tile == "EMG":                       # real EMG-labelled tiles show a clipped flat QVAR (audit); simulated accordingly
        q = np.clip(p.qvar_offset_lsb + 200 * _lpnoise(rng, n, fs, 3.0) + rng.normal(0, 25, n) + 600, -2000, 960)
    else:                                       # EOG tile: near-flat, ~180 distinct values
        q = p.qvar_offset_lsb + 25 * _lpnoise(rng, n, fs, 0.8) + rng.normal(0, 8, n)
    q = (np.clip(np.round(q / QVAR_STEP) * QVAR_STEP, -32768, 32767)).astype(np.int32)   # measured: every real QVAR value is a multiple of 4 LSB
    # --- breathing waveform and holds -----------------------------------------------------------------------------------------------
    phase = np.interp(t, bt, ph, left=ph[0] - 2 * np.pi * p.breath_hz * max(bt[0], 0), right=ph[-1]) if len(bt) else 2 * np.pi * p.breath_hz * t
    env = np.ones(n); gasp = np.zeros(n)
    for a, b, c in p.events:
        sc = 0.05 if c in (HOLD, 2) else 1.3
        ramp_d = 1 / (1 + np.exp(np.clip(-(t - a) / 0.8, -50, 50))); ramp_u = 1 / (1 + np.exp(np.clip(-(t - b) / 0.8, -50, 50)))
        m = (ramp_d * (1 - ramp_u)); env = env * (1 - m * (1 - sc)); gasp += np.exp(-((t - b - 1.8) / 1.5) ** 2) * (t > b) * (2.5 if c in (HOLD, 2) else 0.0)
    breath = np.sin(phase) * env + 0.3 * np.sin(2 * phase + 0.7) * env
    breath = breath * (1 + gasp)
    # --- accelerometer -------------------------------------------------------------------------------------------------------------
    tilt = np.deg2rad(p.tilt_deg); az = rng.uniform(0, 2 * np.pi); g0 = p.gravity_mg * np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), -np.cos(tilt)])
    static = np.tile(g0, (n, 1)); mot_steps = []
    acc = static; acc = acc + np.outer(breath, np.array([0.3, 0.2, 1.0]) * p.breath_amp_mg * np.array([rng.choice([-1, 1]), rng.choice([-1, 1]), 1]))
    acc += np.outer(_lpnoise(rng, n, fs, 0.05), rng.uniform(0.3, 1.5, 3))                    # slow sensor drift (mg)
    static_steps = []
    n_mot = rng.poisson(p.motion_rate_per_s * T); motion = []
    for _ in range(n_mot):
        tm = rng.uniform(5, T - 5); k = int(tm * fs); d = np.deg2rad(rng.uniform(1, 8)) * rng.choice([-1, 1], 3) ; step = p.gravity_mg * d * np.array([1, 1, 0.1])
        static_steps.append((k, step)); tau = np.exp(-np.clip(t - tm, 0, None) / 1.5) * (t >= tm); acc += np.outer(tau * np.sin(2 * np.pi * 3 * (t - tm)), rng.uniform(20, 80, 3) * rng.choice([-1, 1], 3)); motion.append(float(tm))
    stat = np.tile(g0, (n, 1))
    for k, step in static_steps: stat[k:] += step
    stat = stat / np.linalg.norm(stat, axis=1, keepdims=True) * p.gravity_mg; acc = acc - np.tile(g0, (n, 1)) + stat
    taps = []
    for ts in p.sync_taps_s:
        for off in (0.0, 0.5):
            tt = ts + off; tau = np.exp(-np.clip(t - tt, 0, None) / 0.03) * (t >= tt); acc[:, 2] += p.tap_peak_mg * tau * np.cos(2 * np.pi * 25 * (t - tt)); taps.append(float(tt))
    acc += rng.normal(0, p.acc_noise_mg, (n, 3))
    acc = np.clip(acc, -ACC_FS_MG, ACC_FS_MG); acc_lsb = np.round(acc / MG_PER_LSB).astype(np.int32); acc_mg = np.round(acc_lsb * MG_PER_LSB, 3)
    gyro = rng.normal(0, 0.45, (n, 3)) + np.outer(breath, [0.02, 0.03, 0.01]) + np.outer(_lpnoise(rng, n, fs, 0.1), [0.3, 0.3, 0.3]) - 0.5
    gy_lsb = np.round(gyro / DPS_PER_LSB).astype(np.int32); gy = np.round(gy_lsb * DPS_PER_LSB, 3)
    temp = p.temp_c1 - (p.temp_c1 - p.temp_c0) * np.exp(-t / rng.uniform(300, 900)) + 0.05 * _lpnoise(rng, n, fs, 0.01)
    tl = np.round((temp - 25) * 256).astype(np.int32); tc = np.round(25 + tl / 256, 3)
    df = pd.DataFrame({COLUMNS[0]: t_us, COLUMNS[1]: acc_lsb[:, 0], COLUMNS[2]: acc_lsb[:, 1], COLUMNS[3]: acc_lsb[:, 2], COLUMNS[4]: acc_mg[:, 0], COLUMNS[5]: acc_mg[:, 1], COLUMNS[6]: acc_mg[:, 2],
                       COLUMNS[7]: gy_lsb[:, 0], COLUMNS[8]: gy_lsb[:, 1], COLUMNS[9]: gy_lsb[:, 2], COLUMNS[10]: gy[:, 0], COLUMNS[11]: gy[:, 1], COLUMNS[12]: gy[:, 2], COLUMNS[13]: tl, COLUMNS[14]: tc, COLUMNS[15]: q})
    truth = dict(profile=asdict(p), generator=GEN_VERSION, n_samples=n, events=[dict(id=f"E{i+1}", kind=int(c), start_s=float(a), end_s=float(b), duration_s=float(b - a),
                 start_sample=int(round(a * fs)), end_sample=int(round(b * fs))) for i, (a, b, c) in enumerate(p.events)], sync_taps_s=taps, motion_artifact_times_s=motion,
                 beat_times_s=[float(x) for x in bt] if p.tile == "ECG" else [], spec_assumed=SPEC_ASSUMED, synthetic=True)
    return df, truth


def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.3f")
