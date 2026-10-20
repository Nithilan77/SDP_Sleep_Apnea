"""Breath-hold detector inference: SensorTile recording -> per-window P(voluntary breath-hold-like low-effort window) from the Track-B models.

TASK: detects *low chest-motion windows resembling voluntary breath-holds*. It is NOT a sleep-apnea detector: nothing here was trained on clinical apnea labels.
Deployed models are trained on session 1 (+ augmentation/synthetic), so session 1 outputs are IN-SAMPLE (see meta.json).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import joblib, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.inhouse_bh import common as Cm
from src.inhouse_bh.augment import augment_window
from src.deploy import checkpoints as CK

DEPLOY = Path(__file__).resolve().parents[2] / "results/inhouse_bh/deployed"
THR = 0.5; MIN_RUN = 5          # a segment = >= 5 consecutive flagged windows (stride 2 s, window 10 s -> >= 18 s, matching the ~20 s labelled hold core)
MIN_DURATION_S = 120.0


class BHError(ValueError):
    pass


def load_models():
    for n in ("bh_sklearn", "bh_cnn"): CK.require(n)
    from src.inhouse_bh.run_experiments import Net
    sk = joblib.load(DEPLOY / "sklearn_models.joblib"); cnn = Net(); cnn.load_state_dict(torch.load(DEPLOY / "cnn.pt", map_location="cpu")); cnn.eval()
    return sk, cnn, json.loads((DEPLOY / "meta.json").read_text())


def trace_from_path(path: Path):
    try: x, dur, fs = Cm.trace32(path)
    except Exception as e: raise BHError(f"Could not parse as a SensorTile recording: {str(e)[:250]}")
    if dur < MIN_DURATION_S: raise BHError(f"Recording is {dur:.0f} s; at least {MIN_DURATION_S:.0f} s is needed for a breath-hold scan.")
    if not 180 <= fs <= 300: raise BHError(f"Measured sampling rate {fs:.1f} Hz is implausible for this device (expected ~240 Hz).")
    if not np.isfinite(x).all() or float(np.std(x)) < 1e-7: raise BHError("Accelerometer channels are flat or non-finite; the breathing trace cannot be built.")
    return x, dur, fs


def windows_for(x: np.ndarray, stride_s: float = Cm.STRIDE_S):
    i0 = np.arange(0, len(x) - Cm.W, int(stride_s * Cm.HZ)); sc = Cm.local_scales(x, i0)
    X = Cm.window_array(x, i0, sc); rms = np.sqrt((X ** 2).mean((1, 2)))
    reason = np.where(~np.isfinite(X).all((1, 2)), "non-finite", np.where(sc < 1e-5, "flat sensor", np.where(rms > 10, "motion artefact (>10x local RMS)", "")))
    return i0, X, reason


def score(X: np.ndarray, models) -> dict:
    sk, cnn, _ = models; Fm = np.stack([Cm.features(w) for w in X]); out = {n: m.predict_proba(Fm)[:, 1] for n, m in sk.items()}
    with torch.no_grad(): out["CNN"] = torch.sigmoid(cnn(torch.tensor(X))).numpy()
    out["ENS(mean)"] = np.mean([out[k] for k in ("LR", "HGB", "CNN")], 0); return out


def segments(t: np.ndarray, flag: np.ndarray, stride_s: float, min_run: int = MIN_RUN):
    out, i = [], 0
    while i < len(flag):
        if flag[i]:
            j = i
            while j + 1 < len(flag) and flag[j + 1] and abs(t[j + 1] - t[j] - stride_s) < 1e-6: j += 1
            if j - i + 1 >= min_run: out.append((float(t[i]), float(t[j] + Cm.WIN_S)))
            i = j + 1
        else: i += 1
    return out


def predict_recording(path: Path, models=None, n_aug: int = 3, seed: int = 0) -> dict:
    models = models or load_models(); x, dur, fs = trace_from_path(path)
    i0, X, reason = windows_for(x); ok = reason == ""; t0 = i0 / Cm.HZ
    P = {k: np.full(len(X), np.nan) for k in ("LR", "HGB", "CNN", "ENS(mean)")}
    if ok.any():
        for k, v in score(X[ok], models).items(): P[k][ok] = v
    res = dict(duration_s=dur, fs=fs, t0=t0, n_windows=int(len(X)), n_scored=int(ok.sum()), n_rejected=int((~ok).sum()), rejected_reasons=pd.Series(reason[~ok]).value_counts().to_dict(), probs=P,
               meta=models[2], trace=x)
    p = P["ENS(mean)"]; flag = np.nan_to_num(p) >= THR
    res["segments"] = segments(t0, flag, Cm.STRIDE_S); res["flag_frac"] = float(np.nanmean(p >= THR)) if ok.any() else float("nan")
    # augmented variants of the SAME recording (recording-level transform preserving axis relations); model outputs compared with the original
    rng = np.random.default_rng(seed); aug = []
    for k in range(n_aug):
        xa = augment_trace(x, rng); ia, Xa, ra = windows_for(xa); oka = ra == ""; pa = np.full(len(Xa), np.nan)
        if oka.any(): pa[oka] = score(Xa[oka], models)["ENS(mean)"]
        aug.append(dict(trace=xa, p=pa, n_windows=int(len(Xa)), n_rejected=int((~oka).sum()), flag_frac=float(np.nanmean(pa >= THR)), segments=segments(ia / Cm.HZ, np.nan_to_num(pa) >= THR, Cm.STRIDE_S),
                        agreement=float(np.mean((np.nan_to_num(pa) >= THR) == (np.nan_to_num(p) >= THR))) if len(pa) == len(p) else float("nan")))
    res["augmented"] = aug; return res


def augment_trace(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Recording-level label-preserving perturbation: gain, small joint rotation, noise, slow drift (same transform over the whole recording, so windows stay consistent)."""
    from src.inhouse_bh.augment import _rot
    sc = Cm.rec_scale(x); y = (_rot(rng, 8.0) @ x.T).T * np.exp(rng.uniform(np.log(0.8), np.log(1.25)))
    t = np.arange(len(x)) / Cm.HZ; y = y + rng.normal(0, 0.05 * sc, x.shape) + 0.1 * sc * np.sin(2 * np.pi * 0.02 * t + rng.uniform(0, 6.28))[:, None] * rng.uniform(-1, 1, 3)
    return y.astype(np.float32)
