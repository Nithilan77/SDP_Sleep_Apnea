"""
Cessation (breathing-pause) detection from the effort-amplitude trace (Phase 4).

Flags stretches where breathing effort collapses relative to its recent local
baseline for at least MIN_DURATION_S -- a candidate breathing pause on the
effort channel. Only in fusion with the ECG branch can obstructive apnea
(effort continues against a blocked airway) be told apart from central apnea
(effort itself stops) -- see CLAUDE.md §7; this module just flags "effort
went quiet for a while," nothing more.

IMPORTANT input choice: feed detect_cessations() the RAW Hilbert envelope
(EffortSignal.envelope from envelope.py), not the heavily-smoothed
EffortSignal.effort_amplitude trace. That trace is smoothed with a ~10s
window for *reporting* a coarse effort-level trend; using it here blurs away
the edges of exactly the 10-20s events this module targets, and can shrink a
real 15s cessation's detected duration below the 10s minimum -- caught during
development (see src/effort/synthetic_demo.py) when 2 of 3 real synthetic
gaps went undetected until the input was switched from effort_amplitude to
the raw envelope.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

MIN_DURATION_S = 10.0
COLLAPSE_THRESHOLD_FRAC = 0.3   # effort must drop below this fraction of local baseline
BASELINE_WINDOW_S = 60.0        # local baseline = smoothed effort amplitude over this window


@dataclass
class CessationEvent:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def rolling_baseline(effort_amplitude: np.ndarray, fs: float,
                      window_s: float = BASELINE_WINDOW_S) -> np.ndarray:
    """Local 'normal breathing' effort level -- a wide smoothing of the effort trace,
    so the collapse threshold adapts to this person/night's own effort amplitude
    rather than a fixed absolute value."""
    win = max(1, int(round(window_s * fs)))
    return uniform_filter1d(effort_amplitude, size=win, mode="nearest")


def detect_cessations(effort_amplitude: np.ndarray, fs: float,
                       min_duration_s: float = MIN_DURATION_S,
                       threshold_frac: float = COLLAPSE_THRESHOLD_FRAC,
                       baseline_window_s: float = BASELINE_WINDOW_S) -> list[CessationEvent]:
    """Return contiguous stretches where effort_amplitude < threshold_frac * local
    baseline for at least min_duration_s.
    """
    baseline = rolling_baseline(effort_amplitude, fs, baseline_window_s)
    collapsed = effort_amplitude < (threshold_frac * baseline)
    min_samples = int(round(min_duration_s * fs))

    events = []
    n = len(collapsed)
    i = 0
    while i < n:
        if collapsed[i]:
            j = i
            while j < n and collapsed[j]:
                j += 1
            if (j - i) >= min_samples:
                events.append(CessationEvent(start_s=i / fs, end_s=j / fs))
            i = j
        else:
            i += 1
    return events
