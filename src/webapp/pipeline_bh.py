"""Breath-hold demonstration pipeline for the dashboard (Track B). Real model outputs from src/inhouse_bh; nothing hard-coded."""
from __future__ import annotations
import sys, datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.inhouse_bh import predict as BH, common as Cm
from src.deploy import checkpoints as CK
from src.webapp.pipeline import InputError, _png


def analyze(path: Path) -> dict:
    if path.suffix.lower() not in (".csv", ".txt"):
        raise InputError(f"Unsupported file type '{path.suffix}'. Upload a SensorTile .csv or .txt log (the accelerometer columns are used).")
    try:
        models = BH.load_models()
    except CK.CheckpointError as e:
        raise InputError(f"Breath-hold models unavailable: {e}")
    try:
        r = BH.predict_recording(path, models)
    except BH.BHError as e:
        raise InputError(str(e))
    t = np.arange(len(r["trace"])) / Cm.HZ; p = r["probs"]["ENS(mean)"]; tw = r["t0"] + Cm.WIN_S / 2
    d = max(1, len(t) // 20000)
    fig, ax = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    ax[0].plot(t[::d], np.linalg.norm(r["trace"], axis=1)[::d], lw=.4, color="#2E7D32"); ax[0].set_ylabel("|band-passed accel| (g)"); ax[0].set_title("Chest-accelerometer breathing band (0.08–0.6 Hz)", fontsize=9, loc="left")
    rs = Cm.rms_series(r["trace"]); ax[1].plot(np.arange(len(rs)) + Cm.WIN_S / 2, rs, color="#1565C0"); ax[1].set_ylabel("10 s RMS (g)"); ax[1].set_title("Effort amplitude", fontsize=9, loc="left")
    for k, c in (("LR", "#8D6E63"), ("HGB", "#6A1B9A"), ("CNN", "#EF6C00")): ax[2].plot(tw, r["probs"][k], lw=.7, color=c, alpha=.7, label=k)
    ax[2].plot(tw, p, color="k", lw=1.2, label="ensemble (mean)"); ax[2].axhline(BH.THR, color="grey", ls=":"); ax[2].set_ylim(-.02, 1.02); ax[2].legend(ncol=4, fontsize=7); ax[2].set_xlabel("recording time (s)")
    ax[2].set_title("P(breath-hold-like low-effort window)", fontsize=9, loc="left")
    for a, b in r["segments"]:
        for x_ in ax: x_.axvspan(a, b, color="#C62828", alpha=.15)
    fig.tight_layout(); r["plot_main"] = _png(fig)
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True); lim = (0, min(r["duration_s"], 600))
    ax[0].plot(t[::d], np.linalg.norm(r["trace"], axis=1)[::d], lw=.5, color="k", label="original")
    for k, a_ in enumerate(r["augmented"]): ax[0].plot(t[::d], np.linalg.norm(a_["trace"], axis=1)[::d], lw=.4, alpha=.6, label=f"augmented #{k+1}")
    ax[0].legend(fontsize=7, ncol=4); ax[0].set_ylabel("|accel| (g)"); ax[0].set_title("Original vs augmented signal (recording-level gain, rotation, noise, drift)", fontsize=9, loc="left")
    ax[1].plot(tw, p, color="k", label="original")
    for k, a_ in enumerate(r["augmented"]): ax[1].plot(tw[:len(a_["p"])], a_["p"], lw=.7, alpha=.7, label=f"augmented #{k+1}")
    ax[1].axhline(BH.THR, color="grey", ls=":"); ax[1].set_ylim(-.02, 1.02); ax[1].set_xlabel("recording time (s)"); ax[1].set_title("Ensemble output: original vs augmented", fontsize=9, loc="left"); ax[1].legend(fontsize=7, ncol=4)
    fig.tight_layout(); r["plot_aug"] = _png(fig)
    dur_flag = sum(b - a for a, b in r["segments"])
    r["summary"] = (f"{len(r['segments'])} segment(s) flagged as breath-hold-like low-effort ({dur_flag:.0f} s in total) by the ensemble in this recording." if r["segments"]
                    else "No breath-hold-like low-effort segments were flagged by this model in this recording. This is not evidence that breathing was normal throughout, and it says nothing about clinical apnea.")
    r["analysed"] = datetime.datetime.now().isoformat(timespec="seconds")
    for k in ("trace",): r.pop(k)
    for a_ in r["augmented"]: a_.pop("trace")
    r["probs"] = {k: v.tolist() for k, v in r["probs"].items()}; r["t0"] = r["t0"].tolist()
    return r
