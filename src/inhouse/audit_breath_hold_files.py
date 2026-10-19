"""Per-file audit of the six breath-hold SensorTile logs (2 sessions x ECG/EMG/EOG tiles).
Measured rate, gaps, QVAR stats, accel/gyro breathing-band power. Writes results/inhouse_audit/audit.json."""
import json, sys, hashlib, os
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

FILES = {
 "S1_ECG": "data/recordings/breath_hold/user_MukuBreathHold_ECG_07_10_2026.csv",
 "S1_EMG": "data/recordings/breath_hold/user_MukuBreathHold_EMG_07_10_2026.csv",
 "S1_EOG": "data/recordings/breath_hold/user_MukuBreathHold_EOG_07_10_2026.csv",
 "S2_ECG": os.path.expanduser("~/Desktop/Mithin-BreathHold/user_MithinBreathHold_ECG_000.csv"),
 "S2_EMG": os.path.expanduser("~/Desktop/Mithin-BreathHold/user_MithinBreathHold_EMG_000.csv"),
 "S2_EOG": os.path.expanduser("~/Desktop/Mithin-BreathHold/user_MithinBreathHold_EOG_000.csv"),
}
out = {}
for k, p in FILES.items():
    d = pd.read_csv(p); n_trunc = int(d.isna().any(axis=1).sum()); d = d.dropna().reset_index(drop=True)
    t = d["time[us]"].values / 1e6
    dt = np.diff(t)
    fs = 1 / np.median(dt)
    r = dict(truncated_rows_dropped=n_trunc, path=p, size=os.path.getsize(p), rows=len(d), duration_s=float(t[-1]-t[0]),
             fs_median=float(fs), fs_mean=float((len(t)-1)/(t[-1]-t[0])),
             n_gaps_gt_3dt=int((dt > 3/fs).sum()), max_gap_s=float(dt.max()),
             nonmonotonic=int((dt <= 0).sum()))
    q = d["ah_qvar[LSB]"].values.astype(float)
    r["qvar"] = dict(mean=float(q.mean()), std=float(q.std()), min=float(q.min()), max=float(q.max()),
                     frac_at_max=float((q == q.max()).mean()), n_unique=int(len(np.unique(q))))
    sos = butter(3, [0.08, 0.6], "bp", fs=fs, output="sos")
    a = d[["acc_x[mg]", "acc_y[mg]", "acc_z[mg]"]].values
    r["acc_axis_std_mg"] = a.std(0).tolist()
    r["acc_breath_band_std_mg"] = [float(sosfiltfilt(sos, a[:, i]).std()) for i in range(3)]
    g = d[["gyro_x[dsp]", "gyro_y[dsp]", "gyro_z[dsp]"]].values
    r["gyro_breath_band_std_dps"] = [float(sosfiltfilt(sos, g[:, i]).std()) for i in range(3)]
    # breathing spectral peak/median ratio on best axis
    best = int(np.argmax(r["acc_breath_band_std_mg"]))
    f, P = welch(a[:, best] - a[:, best].mean(), fs=fs, nperseg=int(fs*60))
    m = (f >= 0.1) & (f <= 0.5)
    r["acc_best_axis"] = best
    r["acc_peak_hz"] = float(f[m][np.argmax(P[m])]); r["acc_peak_over_median"] = float(P[m].max()/np.median(P[m]))
    # QVAR spectral content: cardiac (0.8-3Hz) vs mains/other
    f, P = welch(q - q.mean(), fs=fs, nperseg=int(fs*10))
    r["qvar_psd_peak_hz"] = float(f[1:][np.argmax(P[1:])])
    out[k] = r
    print(k, f"dur={r['duration_s']:.1f}s fs={fs:.1f} gaps={r['n_gaps_gt_3dt']} qvar_std={q.std():.0f}",
          "accBB(mg)=", np.round(r["acc_breath_band_std_mg"], 2), "pk/med=%.1f@%.2fHz" % (r["acc_peak_over_median"], r["acc_peak_hz"]))
json.dump(out, open("results/inhouse_audit/audit.json", "w"), indent=1)
