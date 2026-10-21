"""Overnight-recording analysis for the presentation dashboard: plain outputs from the same real code as the detailed dashboard (no hard-coded results)."""
from __future__ import annotations
import datetime, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.webapp import pipeline as P, present_data as D
from src.ecg.rpeaks import detect_rpeaks

STRIDE = 10.0


def analyze(path: Path) -> dict:
    from src.deploy import checkpoints as CK
    rec = P.validate_and_load(path)                       # InputError with the reason for bad files
    rpk = detect_rpeaks(rec); t_b = rpk.r_peaks[1:] / rec.fs; hr = 60.0 / rpk.rr_intervals_s; ok = (hr > 30) & (hr < 200)
    mins = np.arange(0, int(rec.duration_s // 60)); hr_min = np.array([np.median(hr[ok & (t_b >= m * 60) & (t_b < m * 60 + 60)]) if (ok & (t_b >= m * 60) & (t_b < m * 60 + 60)).sum() >= 10 else np.nan for m in mins])
    try:
        CK.require("headline.pt"); CK.require("deployed_heads.joblib")
        import joblib
        from src.inhouse import streams as S
        from src.deploy.predict_session import predict_windows, segments
        from src.transfer.train_heads import HEADS
        s = S.build(path, "UP")
        if s is None: raise P.InputError("Breathing-motion streams could not be built (recording too short or unreadable).")
        bundle = joblib.load(HEADS); df = predict_windows(s, bundle["heads"], STRIDE)
    except CK.CheckpointError as e:
        raise P.InputError(f"Breathing analysis unavailable: {e}")
    hours = rec.duration_s / 3600; heads = {}
    for key, col, name in (("A", "p_frozen", "Head A — deep features from the MESA-trained network"), ("B", "p_handcrafted", "Head B — 8 simple breathing-motion measures")):
        fl = df[col].values >= 0.5; seg = segments(df.t_centre_s.values, fl, STRIDE)
        heads[key] = dict(name=name, flag_pct=100 * float(fl.mean()), n_seg=len(seg), seg_per_hour=len(seg) / hours, longest=max((b - a for a, b in seg), default=0.0), seg=seg, p=df[col].values)
    tc = df.t_centre_s.values / 3600; fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    ax[0].plot((mins + .5) / 60, hr_min, color="#C62828", lw=.9); ax[0].set_ylabel("beats / min"); ax[0].set_title("Heart rate (1-minute median)", loc="left", fontsize=10)
    n = len(s.effort_raw["mag"]); tt = np.arange(n) / 32.0 / 3600; w = 32 * 10; rms = np.sqrt(np.convolve(s.effort_raw["mag"] ** 2, np.ones(w) / w, "same")); d = max(1, n // 20000)
    ax[1].plot(tt[::d], rms[::d], color="#2E7D32", lw=.6); ax[1].set_ylabel("size of chest motion"); ax[1].set_title("Breathing / chest motion (10-second average)", loc="left", fontsize=10)
    for key, c in (("A", "#1565C0"), ("B", "#EF6C00")):
        ax[2].plot(tc, heads[key]["p"], color=c, lw=.7, alpha=.85, label=f"Head {key}")
        for a_, b_ in heads[key]["seg"]: ax[2].axvspan(a_ / 3600, b_ / 3600, color=c, alpha=.12)
    ax[2].axhline(.5, color="grey", ls=":"); ax[2].set_ylim(-.02, 1.02); ax[2].set_ylabel("pause-like score"); ax[2].set_xlabel("hours into the recording"); ax[2].legend(fontsize=8, ncol=2)
    ax[2].set_title("How much each stretch looks like a breathing pause (shaded = flagged)", loc="left", fontsize=10); fig.tight_layout()
    for h in heads.values(): h.pop("p"); h["seg"] = h["seg"][:12]
    ref = D.transfer()
    return dict(file=rec.name, analysed=datetime.datetime.now().isoformat(timespec="seconds"), duration_h=hours, fs=float(rec.fs_measured), n_beats=int(len(rpk.r_peaks)), hr_mean=float(np.nanmean(hr_min)),
                hr_min=float(np.nanpercentile(hr_min, 5)), hr_max=float(np.nanpercentile(hr_min, 95)), motion_pct=100 * float(s.motion_frac.mean()), heads=heads, ref=dict(frozen=100 * ref["pooled_frozen"], hand=100 * ref["pooled_hand"], n=ref["n_nights"]),
                in_sample=(path.name == bundle["meta"]["trained_on"]), plot=P._png(fig), heads_trained_on=bundle["meta"]["trained_on"])
