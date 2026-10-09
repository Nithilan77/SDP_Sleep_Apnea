# Breath-hold session — first analysis of real labelled in-house events

**Recording:** 2026-10-07, chest SensorTile box (QVAR-ECG + accelerometer), one healthy volunteer, 7 voluntary ~30 s holds
(stopwatch laps 2,4,…,14). Validation only; raw recording is not committed. Full write-up in `docs/PROGRESS.md` §7e.
This file supersedes the earlier v1 write-up, whose numbers (mean ratio 0.648, "6/7 < 0.8") came from boundaries that
were essentially the nominal stopwatch window and are reproduced below as the *wide* window.

## Headline

**Chest-accelerometer effort visibly collapses during voluntary breath-holds on real hardware.** Under the conservative
inner window (nominal inset by the 4.8 s sync lag) the effort ratio (hold RMS / pre-hold RMS) is below 0.8 in 7/7 events
and below 0.5 in 5/7. Magnitude is **not** pinned down: treat each event's ratio as a range (inner → wide), because
automatic boundary detection did not generalise (below).

| Event | Inner (b) | Wide / nominal (c) | Detector (a) | Detector visual verdict |
|---|---|---|---|---|
| 1 | 0.25 | 0.62 | fell back | FAIL — real quiet ≈997–1015 s not found |
| 2 | 0.27 | 0.31 | 0.07 | OK |
| 3 | 0.64 | 0.49 | 0.05 | FAIL — truncated ≈1240.6 s by a ~3.5 s mid-hold burst |
| 4 | 0.14 | 0.52 | 0.09 | OK |
| 5 | 0.46 | 0.78 | 0.16 | FAIL — hold not cleanly quiet; 6 s fragment |
| 6 | 0.10 | 0.33 | 0.04 | OK |
| 7 | 0.60 | 1.49 | 0.47 | FAIL — start ≈13 s late (quiet from ≈1710.5 s) |
| mean | 0.35 | 0.65 | — | — |

Event 7's wide value of 1.49 is boundary bleed, not increased effort: with a 4.8 s lag the quiet stretch ends ≈5.5 s before
the stopwatch end, so the nominal window contains the recovery-gasp burst. The pre-hold baseline is anchored at the nominal
start, so events whose quiet begins earlier than nominal (e.g. Event 4) have a baseline partly contaminated by quiet, which
biases ratios upward (conservative). The detector column is shown for transparency only; it is not a headline.

## Boundary detection: what was tried, what failed

1. **The bug (previous version).** Refinement ran on the signed bandpassed trace and called any 5 consecutive samples
   (≈20 ms at 243 Hz) with `|x|` under a percentile threshold "quiet". `|bandpassed|` crosses zero twice per breath, so every
   zero-crossing qualified and the first match was always the left edge of the ±12 s search window — all 7 start offsets landed at
   −11.0…−11.9 s. Not seven detections; one search bound.
2. **Hilbert-envelope detector** with threshold = 30% of the event's pre-hold median envelope, a minimum quiet run of 2 measured
   breath periods (per-event, 2.1–3.9 s), chaining across ≤1 period of activity, ≥80% quiet, and an explicit refusal (flagged
   fallback to nominal) when a boundary lands on the search-window edge. Correct on Events 2, 4, 6 only.
3. **Floor-relative threshold** (noise floor from the event's own window + margin). No material change. Stopped here by agreement:
   further tuning on one subject / one session would overfit the detector to this session.

`event_summary.csv` carries `refinement_fell_back_to_nominal`, `fallback_reason`, and a manual `detector_visual_verdict`
column so each row says whether it is a refined detection, a fallback, or a visually-failed detection.

## HR direction: unconfirmed

Hold-minus-pre mean HR changes sign with the choice of boundaries (earlier versions: +0.84 bpm, 5/7 up; −3.08 bpm, 1/7 up; this
version: −1.35 bpm, SD 4.8, 3/7 up). No bradycardia/tachycardia claim is made. Voluntary awake holds plausibly show an
anticipatory sympathetic rise, but that is a hypothesis here, not a finding. Event 1 also contains a ~153 bpm two-beat R-peak
artifact near 1000 s that the single-beat despiker misses, inflating that event's hold-mean HR.

## Sync

Stopwatch total 1800.46 s vs recording 1795.6 s: 4.8 s difference, consistent with the reported 5–10 s manual start lag.

## Limitations / next

- One subject, one session, 7 events: plausibility, not accuracy; no false-positive rate (the 5–7 s SHORT hold from
  `docs/breath_hold_protocol.md` was not recorded).
- Next session: include 3 sharp sync taps (or an event marker) to remove the lag uncertainty, the SHORT hold, and a rest-period
  comparison to rule out "general stillness" vs breath cessation. A second session is also the first honest chance to validate
  any boundary detector against held-out events.
- If effort cessation holds up, revisit §7d adaptation with labelled (breath-hold) supervision, the ingredient all four earlier
  attempts lacked.

## Files

`analyze_session.py` (in `src/breath_hold/`), `event_summary.csv`, `event_XX.png` (HR + bandpassed effort + Hilbert envelope,
threshold, detector boundaries, fixed-pad window), `session_overview.png`, `summary.json`, `run.log`.
