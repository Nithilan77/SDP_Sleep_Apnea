"""Figures for the in-house page: per session, per tile accelerometer-PC1 spectrogram (0.05-1 Hz) and a QVAR 6 s excerpt.
Session-1 ECG tile only: stopwatch-nominal hold windows from results/breath_hold/event_summary.csv are shaded and labelled nominal (NOT verified boundaries).
Session 2: no event overlay (no validated alignment). Outputs results/inhouse_audit/figs/*.png"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

B = "data/recordings/breath_hold/user_MukuBreathHold_%s_07_10_2026.csv"
M = os.path.expanduser("~/Desktop/Mithin-BreathHold/user_MithinBreathHold_%s_000.csv")
OUT = "results/inhouse_audit/figs"
ev = pd.read_csv("results/breath_hold/event_summary.csv")

def load(p):
    d = pd.read_csv(p).dropna().reset_index(drop=True); t = d["time[us]"].values / 1e6
    return d, t, (len(t) - 1) / (t[-1] - t[0])

for sess, tpl in (("S1", B), ("S2", M)):
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), gridspec_kw=dict(width_ratios=[3, 1.2]))
    for row, tile in enumerate(("ECG", "EMG", "EOG")):
        d, t, fs = load(tpl % tile)
        a = d[["acc_x[mg]", "acc_y[mg]", "acc_z[mg]"]].values; a = a - a.mean(0); pc = a @ np.linalg.svd(a[::10], full_matrices=False)[2][0]
        dec = int(fs // 20); x = pc[::dec]; f, tt, S = spectrogram(x, fs=fs / dec, nperseg=int(60 * fs / dec), noverlap=int(50 * fs / dec))
        m = (f >= .05) & (f <= 1.0); ax = axes[row, 0]
        ax.pcolormesh(tt, f[m], 10 * np.log10(S[m] + 1e-12), shading="auto", cmap="viridis")
        ax.axhspan(.15, .6, fill=False, ec="white", lw=.6, ls="--")
        ax.set_ylabel(f"{tile} tile\nacc PC1 Hz"); ax.set_xlim(0, t[-1] - t[0])
        if sess == "S1" and tile == "ECG":
            for _, e in ev.iterrows(): ax.axvspan(e.nominal_start, e.nominal_end, color="red", alpha=.25, lw=0)
            ax.set_title("Session 1 ECG tile: red = stopwatch-nominal hold windows (protocol, not verified boundaries)", fontsize=8, loc="left")
        elif row == 0: ax.set_title(f"{sess}: accelerometer-PC1 spectrogram; dashed box = 0.15-0.6 Hz breathing band" + ("; no event overlay (alignment unvalidated)" if sess == "S2" else ""), fontsize=8, loc="left")
        s0 = int(120 * fs); axq = axes[row, 1]; axq.plot(t[s0:s0 + int(6 * fs)] - t[s0], d["ah_qvar[LSB]"].values[s0:s0 + int(6 * fs)], lw=.6, color="#C62828")
        axq.set_title(f"{tile} tile QVAR, 6 s excerpt (120 s in)", fontsize=8, loc="left")
    axes[2, 0].set_xlabel("time (s)"); axes[2, 1].set_xlabel("s"); fig.tight_layout(); fig.savefig(f"{OUT}/{sess}_tiles.png", dpi=110); plt.close(fig)
print("ok")
