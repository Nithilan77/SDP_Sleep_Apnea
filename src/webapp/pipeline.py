"""Web-app pipeline: one uploaded SensorTile recording -> metadata, validation, plots and REAL model outputs.

Everything shown is computed from the uploaded file by existing project modules (no hard-coded predictions):
  - src.ingest.sensortile.load_sensortile            (parse + empirical fs)
  - src.ecg.rpeaks.detect_rpeaks                      (R-peaks, RR, amplitude)
  - deployable CNN / RNN / LSTM (PhysioNet-trained)   (per-minute apnea probability; RR+amp and RR-only variants)
  - src.deploy / src.transfer effort heads            (per-window hold-likelihood from the chest accelerometer)
"""
from __future__ import annotations

import base64, io, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ingest.sensortile import load_sensortile  # noqa: E402
from src.ecg.rpeaks import detect_rpeaks  # noqa: E402
from src.ecg.cnn_model import RRCNN, predict_cnn  # noqa: E402
from src.ecg.rr_sequence import SEQ_LEN  # noqa: E402
from src.ecg.inhouse_plausibility import _build_unlabelled_minute_sequences  # noqa: E402
from src.ecg.rnn_lstm import RecurrentApnea  # noqa: E402
from src.effort import envelope as E  # noqa: E402

MIN_DURATION_S = 600.0
THR = 0.5
CNN_DIR = ROOT / "results/phase3_cnn/deployable"
REC_DIRS = {"RNN (primary)": ROOT / "results/phase3_rnn_lstm/deployable_rnn", "LSTM (primary)": ROOT / "results/phase3_rnn_lstm/deployable_lstm",
            "RNN (v2, val-selected)": ROOT / "results/phase3_rnn_lstm_v2/deployable_rnn", "LSTM (v2, val-selected)": ROOT / "results/phase3_rnn_lstm_v2/deployable_lstm"}


class InputError(ValueError):
    """A user-facing problem with the uploaded file."""


def _png(fig) -> str:
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=100, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def validate_and_load(path: Path):
    if path.suffix.lower() not in (".csv", ".txt"):
        raise InputError(f"Unsupported file type '{path.suffix}'. Upload a SensorTile .csv (new firmware) or .txt (old tab-delimited) recording.")
    try:
        rec = load_sensortile(path)
    except Exception as e:  # parser raises ValueError/KeyError/pandas errors on wrong layouts
        raise InputError(f"Could not parse as a SensorTile recording: {str(e)[:300]}")
    if rec.n_samples < 1000 or not np.isfinite(rec.duration_s):
        raise InputError(f"Recording too short or unreadable ({rec.n_samples} samples).")
    if rec.duration_s < MIN_DURATION_S:
        raise InputError(f"Recording is {rec.duration_s:.0f} s; at least {MIN_DURATION_S:.0f} s (10 min) is required "
                         "(the effort models use 150 s context windows).")
    if not 180 <= rec.fs_measured <= 300:
        raise InputError(f"Measured sampling rate {rec.fs_measured:.1f} Hz is implausible for this device (expected ~240 Hz).")
    if np.nanstd(rec.qvar) < 1e-6 or not np.isfinite(rec.accel_mg).all():
        raise InputError("QVAR channel is flat or accelerometer contains non-finite values; check the sensor and the file.")
    prom = cardiac_prominence(rec)
    if prom < CARDIAC_PROM_MIN:
        raise InputError(f"The QVAR channel shows no clear cardiac rhythm (spectral prominence {prom:.1f} < {CARDIAC_PROM_MIN:.0f}). "
                         "This ECG apnea classifier needs a recording whose QVAR is wired as chest ECG; EMG/EOG-placement tiles and IMU-only logs are rejected.")
    rec.cardiac_prominence = prom
    return rec


CARDIAC_PROM_MIN = 20.0   # empirical (6 tiles, results/inhouse_audit/breathing_cardiac.json): ECG tiles 54 and 321; EMG/EOG tiles 6.8-9.8. n is small -> conservative gate, not a validated classifier


def cardiac_prominence(rec, n_win: int = 40, win_s: float = 30.0) -> float:
    """Median (over up to n_win 30 s windows) of peak/median Welch power in 0.7-3 Hz of QVAR: high when a cardiac rhythm dominates."""
    from scipy.signal import welch
    fs, q = rec.fs_measured, np.asarray(rec.qvar, float); n = int(win_s * fs)
    starts = np.linspace(0, len(q) - n, min(n_win, max(1, len(q) // n))).astype(int); out = []
    for s0 in starts:
        x = q[s0:s0 + n]; f, P = welch(x - x.mean(), fs=fs, nperseg=n // 2); m = (f >= .7) & (f <= 3.0)
        out.append(P[m].max() / max(np.median(P[m]), 1e-12))
    return float(np.median(out))


def metadata(rec) -> dict:
    return {"File": rec.name, "Samples": f"{rec.n_samples:,}", "Duration": f"{rec.duration_s/60:.1f} min ({rec.duration_s:.0f} s)",
            "Measured sampling rate": f"{rec.fs_measured:.2f} Hz (nominal 240 Hz)",
            "QVAR (ECG) std": f"{np.std(rec.qvar):.1f} LSB",
            "Accelerometer std (x,y,z)": ", ".join(f"{v:.1f}" for v in rec.accel_mg.std(0)) + " mg",
            "Mean temperature": f"{np.nanmean(rec.temp_degC):.1f} °C"}


def signal_plot(rec) -> str:
    fs = rec.fs_measured; t = (rec.timestamp_us - rec.timestamp_us[0]) / 1e6
    a = rec.accel_mg / 1000; es = E.compute_effort_signal(a[:, 0], a[:, 1], a[:, 2], fs)
    fig, ax = plt.subplots(3, 1, figsize=(12, 7))
    s0 = int(min(30, rec.duration_s / 4) * fs); s1 = s0 + int(10 * fs)
    ax[0].plot(t[s0:s1], rec.qvar[s0:s1], lw=.6, color="#C62828"); ax[0].set_title("QVAR / ECG - 10 s excerpt"); ax[0].set_xlabel("s")
    d = max(1, len(t) // 20000)
    ax[1].plot(t[::d], es.bandpassed[::d], lw=.4, color="#2E7D32"); ax[1].set_title("Chest accelerometer, breathing band (0.08-0.6 Hz), whole recording")
    ax[1].set_ylabel("g")
    w = int(10 * fs); rms = np.sqrt(np.convolve(es.bandpassed ** 2, np.ones(w) / w, "same"))
    ax[2].plot(t[::d], rms[::d], color="#1565C0"); ax[2].set_title("Effort amplitude (10 s RMS of breathing band)"); ax[2].set_xlabel("recording time (s)")
    fig.tight_layout(); return _png(fig)


def _load_models():
    models = {}
    n = json.load(open(CNN_DIR / "norm_stats.json")); m = RRCNN(seq_len=SEQ_LEN, n_channels=2)
    m.load_state_dict(torch.load(CNN_DIR / "model_state_dict.pt", map_location="cpu")); m.eval()
    models["CNN"] = (m, n, lambda mm, X: predict_cnn(mm, X))
    for name, d in REC_DIRS.items():
        if (d / "model_state_dict.pt").exists():
            n = json.load(open(d / "norm_stats.json")); m = RecurrentApnea(n["arch"], 2, n["hidden"], n.get("pool", "last"), n.get("bidir", False))
            m.load_state_dict(torch.load(d / "model_state_dict.pt", map_location="cpu")); m.eval()
            def pred(mm, X):
                with torch.no_grad(): return torch.sigmoid(mm(torch.tensor(X, dtype=torch.float32))).numpy()
            models[name] = (m, n, pred)
    return models


def ecg_branch(rec) -> dict:
    rpk = detect_rpeaks(rec)
    seqs, minutes, nb = _build_unlabelled_minute_sequences(rec, rpk)
    out = dict(n_beats=int(len(rpk.r_peaks)), mean_hr=float(60 / np.median(rpk.rr_intervals_s)), n_minutes=int(len(seqs)), rows=[], missing=[])
    if len(seqs) == 0:
        raise InputError("Too few detectable heartbeats (R-peaks) to form any 1-minute ECG window; check electrode contact.")
    series = {}
    for name, (m, n, pred) in _load_models().items():
        mu = np.array(n["mean"], np.float32).reshape(1, 2, 1); sd = np.array(n["std"], np.float32).reshape(1, 2, 1)
        p2 = pred(m, (seqs - mu) / sd)
        s_rr = seqs.copy(); s_rr[:, 1, :] = mu[0, 1, 0]; p1 = pred(m, (s_rr - mu) / sd)   # amplitude neutralised
        out["rows"].append(dict(model=name, pct_2ch=100 * float((p2 >= THR).mean()), pct_rr=100 * float((p1 >= THR).mean()),
                                mean_2ch=float(p2.mean()), mean_rr=float(p1.mean())))
        series[name] = (p2, p1)
    out["missing"] = [k for k in REC_DIRS if k not in series]
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    col = dict(zip(["CNN", *REC_DIRS], ["#1565C0", "#EF6C00", "#2E7D32", "#8D6E63", "#6A1B9A"]))
    for name, (p2, p1) in series.items():
        ax[0].plot(minutes, p2, lw=1, label=name, color=col[name]); ax[1].plot(minutes, p1, lw=1, label=name, color=col[name])
    for a_, t_ in zip(ax, ("P(apnea) per minute - RR + R-amplitude (amplitude channel is NOT calibrated for QVAR)", "P(apnea) per minute - RR only (amplitude neutralised)")):
        a_.axhline(THR, color="grey", ls=":"); a_.set_ylim(-.02, 1.02); a_.set_title(t_, fontsize=9, loc="left"); a_.legend(fontsize=7, ncol=5)
    ax[1].set_xlabel("minute of recording"); fig.tight_layout(); out["plot"] = _png(fig)
    return out


def effort_branch(path: Path, stride: float = 5.0) -> dict:
    import joblib
    from src.inhouse import streams as S
    from src.deploy.predict_session import predict_windows, segments
    from src.transfer.train_heads import HEADS
    s = S.build(path, "UP")
    if s is None:
        raise InputError("Effort branch could not build streams (recording too short / unreadable).")
    bundle = joblib.load(HEADS); df = predict_windows(s, bundle["heads"], stride)
    res = dict(n_windows=len(df), heads={}, in_sample=(path.name == bundle["meta"]["trained_on"]))
    fig, ax = plt.subplots(3, 1, figsize=(12, 6.5), sharex=True)
    tt = np.arange(len(s.effort_raw["mag"])) / 32.0
    ax[0].plot(tt, s.effort_raw["mag"], lw=.4, color="#43A047"); ax[0].set_ylabel("breathing band (g)")
    for k, (col, nm, c) in enumerate((("p_handcrafted", "B hand-crafted effort head", "#1976D2"), ("p_frozen", "A frozen MESA-CANet head", "#EF6C00")), 1):
        fl = df[col].values >= THR; seg = segments(df.t_centre_s.values, fl, stride)
        res["heads"][nm] = dict(pct=100 * float(fl.mean()), segments=[(round(a), round(b)) for a, b in seg[:30]], n_segments=len(seg))
        ax[k].plot(df.t_centre_s, df[col], color=c); ax[k].axhline(THR, color="grey", ls=":"); ax[k].set_ylim(-.02, 1.02); ax[k].set_title(nm, fontsize=9, loc="left")
        for a_, b_ in seg: ax[k].axvspan(a_, b_, color=c, alpha=.15)
    ax[2].set_xlabel("recording time (s)"); fig.tight_layout(); res["plot"] = _png(fig)
    return res


def analyze(path: Path) -> dict:
    rec = validate_and_load(path)
    r = dict(meta=metadata(rec), signals=signal_plot(rec), ecg=None, effort=None, warnings=[],
             validation=dict(channels="QVAR (ECG), accelerometer x/y/z, gyroscope x/y/z, temperature", fs_measured=float(rec.fs_measured), duration_s=float(rec.duration_s),
                             cardiac_prominence=float(rec.cardiac_prominence), gate=CARDIAC_PROM_MIN, min_duration_s=MIN_DURATION_S))
    try: r["ecg"] = ecg_branch(rec)
    except InputError as e: r["warnings"].append(str(e))
    try: r["effort"] = effort_branch(path)
    except InputError as e: r["warnings"].append(str(e))
    if r["ecg"] and r["ecg"]["missing"]:
        r["warnings"].append("Recurrent checkpoints not found for: " + ", ".join(r["ecg"]["missing"]) + " (run python3 -m src.ecg.rnn_lstm --final).")
    return r
