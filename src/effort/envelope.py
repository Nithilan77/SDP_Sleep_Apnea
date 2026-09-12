"""
Effort-branch signal processing (Phase 4): IMU accelerometer -> breathing
envelope, rate, and amplitude.

Pure signal processing, no training -- per CLAUDE.md §6, there is no public
dataset pairing IMU with expert apnea labels, so unlike the ECG branch this
cannot be trained or cross-validated the same way. It can only be built
against known respiratory physics and checked against synthetic data (this
module) and, later, our own breath-hold recordings (Phase 5).

Pipeline (CLAUDE.md's architecture diagram):
    accel magnitude -> motion mask
                     -> bandpass 0.08-0.6 Hz -> Hilbert envelope -> breathing rate + effort amplitude

0.08-0.6 Hz = 4.8-36 breaths/min: comfortably covers normal adult respiratory
rate (~12-20/min) with margin for sleep bradypnea and stress-driven tachypnea.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, hilbert

BAND_LOW_HZ = 0.08
BAND_HIGH_HZ = 0.6
FILTER_ORDER = 3

MOTION_WINDOW_S = 2.0
MOTION_STD_THRESHOLD_G = 0.05  # rolling std of raw accel magnitude above this = gross
                                # body movement, not quiet chest-wall breathing motion

EFFORT_AMPLITUDE_WINDOW_S = 10.0  # smoothing window for the reported effort-amplitude trace


def accel_magnitude(ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> np.ndarray:
    """Combine 3-axis acceleration into one orientation-invariant magnitude (g)."""
    return np.sqrt(ax ** 2 + ay ** 2 + az ** 2)


def compute_motion_mask(mag: np.ndarray, fs: float, window_s: float = MOTION_WINDOW_S,
                         std_threshold: float = MOTION_STD_THRESHOLD_G) -> np.ndarray:
    """Flag samples during gross body movement (rolling std of raw magnitude too high
    to be quiet chest-wall breathing motion). True = motion -> exclude from breathing
    rate/cessation confidence (a data-quality gate, not a cessation signal itself:
    a breath-hold is quiet, not noisy -- it shows up as an envelope collapse, not
    as motion).
    """
    win = max(1, int(round(window_s * fs)))
    mean = uniform_filter1d(mag, size=win, mode="nearest")
    mean_sq = uniform_filter1d(mag ** 2, size=win, mode="nearest")
    var = np.clip(mean_sq - mean ** 2, 0.0, None)
    return np.sqrt(var) > std_threshold


def bandpass_filter(signal: np.ndarray, fs: float, low: float = BAND_LOW_HZ,
                     high: float = BAND_HIGH_HZ, order: int = FILTER_ORDER) -> np.ndarray:
    """Zero-phase Butterworth bandpass isolating the respiratory frequency band."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def hilbert_envelope(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Analytic-signal envelope (instantaneous amplitude) and unwrapped phase."""
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    return envelope, phase


def instantaneous_breathing_rate_bpm(phase: np.ndarray, fs: float) -> np.ndarray:
    """Breathing rate (breaths/min) from the analytic signal's instantaneous frequency."""
    inst_freq_hz = np.diff(phase) / (2 * np.pi) * fs
    inst_freq_hz = np.concatenate([[inst_freq_hz[0]], inst_freq_hz])  # pad to len(phase)
    return inst_freq_hz * 60.0


def effort_amplitude_trace(envelope: np.ndarray, fs: float,
                            window_s: float = EFFORT_AMPLITUDE_WINDOW_S) -> np.ndarray:
    """Smoothed envelope -- the reported 'effort amplitude' signal."""
    win = max(1, int(round(window_s * fs)))
    return uniform_filter1d(envelope, size=win, mode="nearest")


@dataclass
class EffortSignal:
    fs: float
    magnitude: np.ndarray              # raw accel magnitude (g)
    motion_mask: np.ndarray            # True = gross motion, exclude from rate/cessation confidence
    bandpassed: np.ndarray             # magnitude, 0.08-0.6 Hz filtered
    envelope: np.ndarray               # Hilbert envelope of the bandpassed signal -- full time
                                        # resolution; use THIS for cessation.detect_cessations()
    inst_breathing_rate_bpm: np.ndarray
    effort_amplitude: np.ndarray       # envelope smoothed over ~10s -- for reporting/plotting a
                                        # coarse effort-level trend, NOT for detecting individual
                                        # 10-20s cessation events (the smoothing blurs their edges)


def compute_effort_signal(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, fs: float,
                           amplitude_window_s: float = EFFORT_AMPLITUDE_WINDOW_S) -> EffortSignal:
    """Full pipeline: 3-axis accel -> EffortSignal (envelope, rate, amplitude, motion mask)."""
    mag = accel_magnitude(ax, ay, az)
    motion_mask = compute_motion_mask(mag, fs)
    bp = bandpass_filter(mag, fs)
    envelope, phase = hilbert_envelope(bp)
    rate = instantaneous_breathing_rate_bpm(phase, fs)
    amp = effort_amplitude_trace(envelope, fs, window_s=amplitude_window_s)
    return EffortSignal(fs=fs, magnitude=mag, motion_mask=motion_mask, bandpassed=bp,
                         envelope=envelope, inst_breathing_rate_bpm=rate, effort_amplitude=amp)


def summarize_breathing_rate(effort: EffortSignal, low_bpm: float = 6.0,
                              high_bpm: float = 40.0) -> float:
    """Single representative breathing rate: median instantaneous rate over quiet
    (non-motion), physiologically plausible samples."""
    valid = ~effort.motion_mask
    rates = effort.inst_breathing_rate_bpm[valid]
    plausible = rates[(rates >= low_bpm) & (rates <= high_bpm)]
    return float(np.median(plausible)) if len(plausible) else float("nan")
