"""Does each tile's IMU carry a breathing peak (0.15-0.6 Hz, away from the 0.1 Hz drift edge), and does each QVAR carry cardiac content?
60 s windows; writes results/inhouse_audit/breathing_cardiac.json."""
import json, os
import numpy as np, pandas as pd
from scipy.signal import welch

B = "data/recordings/breath_hold/user_MukuBreathHold_%s_07_10_2026.csv"
M = os.path.expanduser("~/Desktop/Mithin-BreathHold/user_MithinBreathHold_%s_000.csv")
FILES = {f"S1_{k}": B % k for k in ("ECG", "EMG", "EOG")} | {f"S2_{k}": M % k for k in ("ECG", "EMG", "EOG")}

def win_peaks(x, fs, lo, hi, win=60):
    n = int(win*fs); out = []
    for s in range(0, len(x)-n, n):
        f, P = welch(x[s:s+n]-x[s:s+n].mean(), fs=fs, nperseg=n//2)
        m = (f >= lo) & (f <= hi)
        out.append((f[m][np.argmax(P[m])], P[m].max()/np.median(P[m])))
    return np.array(out)

res = {}
for k, p in FILES.items():
    d = pd.read_csv(p).dropna().reset_index(drop=True)
    t = d["time[us]"].values/1e6; fs = (len(t)-1)/(t[-1]-t[0])
    a = d[["acc_x[mg]", "acc_y[mg]", "acc_z[mg]"]].values
    # PCA first component of accel (axis-agnostic)
    ac = a - a.mean(0); u, s, vt = np.linalg.svd(ac[::10], full_matrices=False); pc = ac @ vt[0]
    w = win_peaks(pc, fs, 0.15, 0.6)
    q = d["ah_qvar[LSB]"].values.astype(float)
    wc = win_peaks(q, fs, 0.7, 3.0, win=30)
    res[k] = dict(
        breath_peak_median_hz=float(np.median(w[:, 0])), breath_peak_median_prominence=float(np.median(w[:, 1])),
        frac_windows_prom_gt_8=float((w[:, 1] > 8).mean()), n_windows=int(len(w)),
        breath_hz_iqr=[float(np.percentile(w[:, 0], 25)), float(np.percentile(w[:, 0], 75))],
        qvar_cardiac_peak_median_hz=float(np.median(wc[:, 0])), qvar_cardiac_prominence=float(np.median(wc[:, 1])),
        qvar_cardiac_frac_gt_8=float((wc[:, 1] > 8).mean()))
    print(k, {x: round(v, 2) if not isinstance(v, list) else [round(i, 2) for i in v] for x, v in res[k].items()})
json.dump(res, open("results/inhouse_audit/breathing_cardiac.json", "w"), indent=1)
