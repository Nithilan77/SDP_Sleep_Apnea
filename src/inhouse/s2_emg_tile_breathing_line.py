"""Exploratory: track the breathing spectral line in the Session-2 EMG-tile accelerometer and look at its amplitude over time.
Method: acc PC1 -> 60 s spectrogram -> per-window peak freq in 0.15-0.7 Hz (median-filtered track) -> bandpass +/-0.08 Hz around the track -> Hilbert envelope (5 s smooth).
No labels are used or inferred. Output: results/inhouse_audit/s2_emg_breathing_line.{png,json}."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import spectrogram, butter, sosfiltfilt, hilbert, medfilt

p = os.path.expanduser("~/Desktop/Mithin-BreathHold/user_MithinBreathHold_EMG_000.csv")
d = pd.read_csv(p).dropna().reset_index(drop=True); t = d["time[us]"].values / 1e6; fs = (len(t) - 1) / (t[-1] - t[0])
a = d[["acc_x[mg]", "acc_y[mg]", "acc_z[mg]"]].values; a = a - a.mean(0); pc = a @ np.linalg.svd(a[::10], full_matrices=False)[2][0]
dec = int(fs // 20); fs2 = fs / dec; x = pc[::dec]; tt_ = t[::dec][:len(x)] - t[0]
f, tt, S = spectrogram(x, fs=fs2, nperseg=int(30 * fs2), noverlap=int(27 * fs2)); m = (f >= .15) & (f <= .7)
track = medfilt(f[m][np.argmax(S[m], 0)], 9); tr_t = tt
ftr = np.interp(tt_, tr_t, track)
# time-varying narrow band: filter in blocks around the local track frequency
env = np.zeros_like(x)
for s0 in range(0, len(x), int(20 * fs2)):
    fc = ftr[min(s0 + int(10 * fs2), len(x) - 1)]; sos = butter(3, [max(fc - .08, .05), fc + .08], "bp", fs=fs2, output="sos")
    seg = slice(max(0, s0 - int(30 * fs2)), min(len(x), s0 + int(50 * fs2))); y = sosfiltfilt(sos, x[seg]); e = np.abs(hilbert(y))
    k0 = s0 - seg.start; env[s0:s0 + int(20 * fs2)] = e[k0:k0 + len(env[s0:s0 + int(20 * fs2)])]
k = int(5 * fs2); env_s = np.convolve(env, np.ones(k) / k, "same")
ref = np.median(env_s[int(40 * fs2):-int(40 * fs2)]); rel = env_s / ref
# dips: contiguous stretches where rel < 0.4 for >= 15 s
low = rel < 0.4; dips = []; i = 0
while i < len(low):
    if low[i]:
        j = i
        while j < len(low) and low[j]: j += 1
        if (j - i) / fs2 >= 15: dips.append((round(tt_[i], 1), round(tt_[j - 1], 1)))
        i = j
    else: i += 1
fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
ax[0].plot(tr_t, track, color="#1565C0"); ax[0].set_ylabel("tracked breathing\nline (Hz)")
ax[1].plot(tt_, rel, color="#2E7D32", lw=.6); ax[1].axhline(.4, color="grey", ls=":"); ax[1].set_ylabel("line amplitude\n/ median")
for a0, b0 in dips: ax[1].axvspan(a0, b0, color="red", alpha=.2)
ax[2].plot(tt_[::10], x[::10], lw=.3, color="#555"); ax[2].set_ylabel("acc PC1 (mg)"); ax[2].set_xlabel("time since start of this tile's log (s)")
ax[0].set_title("Session-2 EMG tile: breathing-line amplitude (exploratory; shaded = amplitude <40% of median for >=15 s; NOT labelled holds)", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig("results/inhouse_audit/s2_emg_breathing_line.png", dpi=110)
res = dict(duration_s=float(tt_[-1]), track_hz_median=float(np.median(track)), track_hz_range=[float(track.min()), float(track.max())], dips_ge15s=dips, n_dips=len(dips),
           dip_durations_s=[round(b - a_, 1) for a_, b in dips], dip_start_gaps_s=[round(dips[i + 1][0] - dips[i][0], 1) for i in range(len(dips) - 1)])
json.dump(res, open("results/inhouse_audit/s2_emg_breathing_line.json", "w"), indent=1); print(res)
