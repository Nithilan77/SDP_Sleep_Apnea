"""Augmentations of 3-axis breathing-band windows (3,W) in per-recording-scaled units.  Axis relationships are preserved:
 gain (sensor/coupling), rotation (small orientation change applied jointly to all axes), white noise (accel noise floor), slow drift (<0.05 Hz, posture creep),
 time-warp (+-6% breathing-rate change), and band-edge variation (re-filtering with 0.06-0.10 / 0.5-0.7 Hz edges).  Label-preserving; windows are not made
 to cross the (uncertain) boundaries because augmentation acts inside the window only.  Each returns a new array; deterministic given rng."""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt
from src.inhouse_bh.common import HZ, W

ALL = ["gain", "rotate", "noise", "drift", "warp", "bandedge"]


def _rot(rng, deg=10.0):
    a = np.deg2rad(rng.uniform(-deg, deg, 3)); cx, cy, cz = np.cos(a); sx, sy, sz = np.sin(a)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]); Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def augment_window(x: np.ndarray, rng: np.random.Generator, comps=ALL, p=0.6, noise_sd=0.04) -> np.ndarray:
    y = x.copy()
    for c in comps:
        if rng.random() > p: continue
        if c == "gain": y = y * np.exp(rng.uniform(np.log(0.7), np.log(1.4)))
        elif c == "rotate": y = _rot(rng) @ y
        elif c == "noise": y = y + rng.normal(0, noise_sd * rng.uniform(0.5, 2.0), y.shape)
        elif c == "drift":
            t = np.arange(W) / HZ; y = y + (rng.uniform(-0.15, 0.15, (3, 1)) * np.sin(2 * np.pi * rng.uniform(0.01, 0.05) * t + rng.uniform(0, 6.28)))
        elif c == "warp":
            s = rng.uniform(0.94, 1.06); src = np.clip(np.arange(W) * s + (1 - s) * (W - 1) / 2, 0, W - 1); y = np.stack([np.interp(src, np.arange(W), ch) for ch in y])
        elif c == "bandedge":
            lo, hi = rng.uniform(0.06, 0.10), rng.uniform(0.5, 0.7); y = sosfiltfilt(butter(2, [lo, hi], "band", fs=HZ, output="sos"), y, axis=1)
    return y.astype(np.float32)
