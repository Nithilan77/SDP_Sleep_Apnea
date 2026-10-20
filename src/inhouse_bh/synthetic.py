"""Simulator-generated labelled breath-hold windows for Track B (labels from the simulator schedule; always `voluntary_breath_hold`, never apnea).
The generator spec is fitted WITHOUT the held-out session (SYNTH_SPEC_DIR=results/synth/spec_noS1) when evaluating on S1."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt
from src.synth.render import sample_profile, render
from src.inhouse_bh.common import HZ, W, WIN_S, rec_scale, window_array, local_scales

HOLD_D = [15, 20, 25, 30, 45, 60]


def synth_trace(df: pd.DataFrame, fs: float):
    t = (df["time[us]"].to_numpy() - df["time[us]"].iloc[0]) / 1e6; a = df[["acc_x[mg]", "acc_y[mg]", "acc_z[mg]"]].to_numpy() / 1000.0
    xb = sosfiltfilt(butter(3, [0.08, 0.6], btype="band", fs=fs, output="sos"), a - np.median(a, 0), axis=0)
    tg = np.arange(0, t[-1], 1 / HZ); return np.stack([np.interp(tg, t, xb[:, i]) for i in range(3)], 1).astype(np.float32)


def synthetic_windows(n_recordings: int, seed: int, minutes: float = 14.0, stride_s: float = 2.0, margin: float = 1.5, max_ratio: float = 0.5, return_stats: bool = False):
    """-> arrays X (n,3,W), y (1 hold / 0 non-hold), meta DataFrame. Hold core = [start+margin, end-margin]; baseline = [prev_end+25, start-6]; recovery = [end+margin, end+20]."""
    rng = np.random.default_rng(seed); Xs, ys, meta = [], [], []; stats = dict(recordings=0, rejected_recordings=0, rejected_windows=0)
    for k in range(n_recordings):
        dur = minutes * 60; t = float(rng.uniform(60, 100)); ev = []
        while True:
            d = float(rng.choice(HOLD_D) + rng.uniform(-1, 1))
            if t + d > dur - 60: break
            ev.append((t, t + d, 1)); t += d + max(70.0, 2.5 * d) * rng.uniform(1.0, 1.3)
        sd = int(seed * 1000 + k); p = sample_profile(f"SYN-BH-{sd}", sd, dur, "ECG", holds=ev, taps=False)
        df, truth = render(p); x = synth_trace(df, df.attrs.get("fs", p.fs_hz)); r_ = None
        stats['recordings'] += 1; r_X, r_y, r_m = [], [], []
        segs = []; prev = 20.0
        for i, (a, b, _) in enumerate(ev):
            segs += [("baseline_reference", prev + 5 if i else 60, a - 6, 0), ("voluntary_breath_hold", a + margin, b - margin, 1), ("recovery", b + margin, b + 20, 0)]; prev = b + 25
        for lab, s0, s1, yy in segs:
            t0 = s0
            while t0 + WIN_S <= s1 + 1e-9:
                i0 = int(round(t0 * HZ))
                if i0 + W <= len(x): r_X.append(window_array(x, np.array([i0]), local_scales(x, np.array([i0])))[0]); r_y.append(yy); r_m.append(dict(synthetic_id=p.name, label=lab, t0=t0, seed=sd))
                t0 += stride_s
        # QC gate: reject a synthetic recording whose simulated holds are not separable from its own baseline (breathing too weak vs sensor noise: the label would be noise).
        # median RMS(hold windows) / median RMS(non-hold windows) must be < max_ratio. Uses only the synthetic recording itself.
        ry = np.array(r_y); rr = np.sqrt((np.stack(r_X) ** 2).mean((1, 2))) if len(r_X) else np.array([])
        if len(ry) and (ry == 1).any() and (ry == 0).any() and np.median(rr[ry == 1]) / (np.median(rr[ry == 0]) + 1e-9) < max_ratio:
            Xs += r_X; ys += r_y; meta += r_m
        else:
            stats['rejected_recordings'] += 1; stats['rejected_windows'] += len(r_X)
    out = (np.stack(Xs).astype(np.float32), np.array(ys), pd.DataFrame(meta))
    return (*out, stats) if return_stats else out
