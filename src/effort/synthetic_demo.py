"""
Synthetic sanity check for the effort branch (Phase 4).

There is no real IMU data yet (SensorTile hardware is down) and no public
apnea dataset with IMU either, so the only thing to validate the envelope +
cessation pipeline against right now is a synthetic accelerometer signal
with known, inserted "breath-hold" gaps. This is NOT training -- the pipeline
has no learned parameters -- it's a check that the signal-processing chain
actually detects a collapse in effort when one is there, and does not fire
on gaps shorter than the 10s minimum.

Usage:
    python src/effort/synthetic_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.effort.cessation import detect_cessations  # noqa: E402
from src.effort.envelope import compute_effort_signal, summarize_breathing_rate  # noqa: E402

FS = 50.0                # synthetic IMU accelerometer rate (Hz)
DURATION_S = 360.0       # 6 minutes
BREATHING_HZ = 0.25      # 15 breaths/min, normal adult resting rate
BREATHING_AMP_G = 0.02   # typical subtle chest-wall acceleration during breathing
GRAVITY_G = 1.0
NOISE_STD_G = 0.005

# (start_s, end_s) breath-hold gaps -- amplitude collapses to ~0 for this stretch.
# Includes one 7s gap, deliberately BELOW the 10s cessation threshold, to check
# the detector correctly does NOT flag it.
GAPS = [
    (60.0, 80.0),    # 20s -- should detect
    (150.0, 165.0),  # 15s -- should detect
    (220.0, 227.0),  #  7s -- should NOT detect (below MIN_DURATION_S)
    (300.0, 312.0),  # 12s -- should detect
]


def generate_synthetic_accel(fs: float = FS, duration_s: float = DURATION_S,
                              breathing_hz: float = BREATHING_HZ,
                              breathing_amp_g: float = BREATHING_AMP_G,
                              gravity_g: float = GRAVITY_G, noise_std_g: float = NOISE_STD_G,
                              gaps: list[tuple[float, float]] | None = None, seed: int = 0):
    """Synthetic 3-axis accelerometer signal: mostly-still chest lying on one axis,
    with a sinusoidal breathing oscillation that collapses to ~0 during `gaps`.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs

    amp = np.full(n, breathing_amp_g)
    for g0, g1 in (gaps or []):
        amp[(t >= g0) & (t < g1)] = 0.0

    breathing = amp * np.sin(2 * np.pi * breathing_hz * t)
    az = gravity_g + breathing + rng.normal(0, noise_std_g, n)
    ax = rng.normal(0, noise_std_g / 2, n)
    ay = rng.normal(0, noise_std_g / 2, n)
    return t, ax, ay, az


def _overlap_s(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def main() -> None:
    print(f"Generating synthetic accel signal: {DURATION_S:.0f}s @ {FS:.0f} Hz, "
          f"true breathing rate = {BREATHING_HZ * 60:.1f} bpm")
    print(f"Inserted gaps: {GAPS}")

    t, ax, ay, az = generate_synthetic_accel(gaps=GAPS)

    effort = compute_effort_signal(ax, ay, az, FS)
    rate_bpm = summarize_breathing_rate(effort)
    print(f"\nEstimated breathing rate (quiet, plausible samples only): {rate_bpm:.1f} bpm "
          f"(true = {BREATHING_HZ * 60:.1f} bpm)")

    # Cessation detection uses the raw Hilbert envelope, not the 10s-smoothed
    # effort_amplitude trace: that smoothing window is comparable to (or larger
    # than) the 10-20s events we're trying to detect and blurs away their edges
    # (a first pass using effort_amplitude here missed 2 of 3 real gaps for
    # exactly this reason -- see cessation.py's module docstring).
    detected = detect_cessations(effort.envelope, FS)
    print(f"\nDetected cessation events ({len(detected)}):")
    for e in detected:
        print(f"  {e.start_s:6.1f}s - {e.end_s:6.1f}s  (duration {e.duration_s:.1f}s)")

    print("\n=== Checking against ground truth ===")
    all_ok = True
    for g0, g1 in GAPS:
        gap_dur = g1 - g0
        should_detect = gap_dur >= 10.0
        best_overlap = max((_overlap_s(g0, g1, e.start_s, e.end_s) for e in detected), default=0.0)
        was_detected = best_overlap >= 0.5 * gap_dur  # majority of the true gap was flagged

        status = "OK" if (was_detected == should_detect) else "** MISMATCH **"
        if was_detected != should_detect:
            all_ok = False
        print(f"  gap {g0:.0f}-{g1:.0f}s ({gap_dur:.0f}s, "
              f"{'>=10s, expect detection' if should_detect else '<10s, expect NO detection'}): "
              f"{'detected' if was_detected else 'not detected'}  [{status}]")

    print()
    if all_ok:
        print("GREEN LIGHT: envelope + cessation pipeline behaves as expected on synthetic data "
              "-- detects real gaps >=10s, does not fire on the 7s sub-threshold gap.")
    else:
        print("WARNING: at least one gap did not match expectations -- investigate before "
              "trusting this pipeline on real data.")


if __name__ == "__main__":
    main()
