# Breath-hold session — first analysis of real labelled in-house events

**Date:** 2026-10-08
**Session:** chest SensorTile box, 7 voluntary ~30s breath holds, new .csv firmware format (first real use)

## Headline result

**Accelerometer effort DOES drop during breath-hold on real data:** 6/7 events
show effort RMS ratio (hold vs pre-hold) < 0.8, several markedly so (0.31,
0.33, 0.49, 0.52, 0.62, 0.78). Mean ratio 0.648 (SD 0.407). This is the
first direct, labelled evidence that the chest accelerometer's signal
carries breath-cessation information -- a meaningful counterpoint to §7c,
which showed the MESA-belt-trained model's *learned representation* doesn't
transfer, not that the raw accelerometer signal itself is uninformative.

Event 7 is an outlier (ratio 1.494, effort increased) -- not yet explained,
flagged for visual inspection rather than dismissed.

## HR result does not match the naive prediction

5/7 events show HR INCREASING during the hold (mean delta +0.84 bpm, SD
3.07 -- essentially no consistent direction across events). This is not
the bradycardia signature used in clinical OSA scoring.

**Not treated as a data quality problem.** These are voluntary, anticipated,
awake breath-holds -- physiologically different from involuntary sleep
apnea. The expected vagal bradycardia is strongest in diving-reflex
conditions and in sleep; a short (~30s) voluntary awake hold commonly shows
an anticipatory sympathetic HR rise that can dominate over any bradycardia
at this duration. This is a genuine limitation of voluntary-breath-hold
protocols as a stand-in for sleep apnea physiology, stated plainly rather
than smoothed over.

**Methodological caveat on event boundary refinement:** the refinement
method (steepest HR gradient near nominal boundaries) assumes a
bradycardia/tachycardia signature to lock onto. Given that signature is
absent/reversed in most events, HR-based refined boundaries should be
trusted less than the nominal stopwatch boundaries; effort-based analysis
is on firmer ground.

## Sync check

Stopwatch total (1800.46s) vs recording duration (1795.6s): 4.8s difference
-- consistent with the reported 5-10s manual multi-device start lag.
Confirms the devices were reasonably synchronized at the session level.

## Files

- `analyze_session.py` -- analysis script (R-peak detection, effort
  envelope, per-event windowed comparison, boundary refinement).
- `event_summary.csv` -- per-event HR and effort metrics.
- `event_XX.png` -- per-event HR + effort trace plots (7 files).
- `session_overview.png` -- full-session HR and effort trace with nominal
  hold windows shaded.
- `summary.json` -- numeric summary.

## Next steps

- Visual inspection of session_overview.png and event_07.png (the outlier).
- Compare hold-period effort against REST-period effort (not just
  immediate pre-hold baseline) to rule out "general stillness" as a
  confound distinct from "breath cessation specifically."
- If the effort-cessation signal holds up visually, this motivates
  revisiting the adaptation question (§7d) with labelled rather than
  purely unsupervised objectives -- the missing ingredient in all four
  prior adaptation attempts.
