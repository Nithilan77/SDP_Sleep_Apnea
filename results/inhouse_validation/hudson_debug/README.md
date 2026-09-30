# Hudson/Hari debug: why RR-only stayed implausible

**Starting hypothesis (to test, not assume):** both hudson recordings have low
mean HR (44.7/45.8 bpm), so R-peak *under-detection* (missed beats -> doubled
RR intervals read as false bradycardia) seemed like the likely cause of the
implausibly high apnea rate surviving amplitude-channel neutralization.

**That hypothesis is NOT supported by the evidence below and is rejected.**
The real causes are different for the two hudson files vs Hari, and neither
is a code bug in the loader/peak detector.

## What the diagnostics actually show

| file | n_beats | mean HR | RR mean/std (s) | implausible RR% (fast/slow) | max single gap |
|---|---|---|---|---|---|
| hudson_1101 | 15484 | 44.7 | 1.343 / 0.245 | 0.0% / 0.019% | 2.9s |
| hudson_1510 | 2150 | 45.8 | 1.310 / 0.199 | 0.0% / 0.0% | 1.71s |
| hari_2406 | 3796 | 57.5 | 1.043 / 0.249 | 0.0% / 0.896% | 4.36s |
| kishor_0404 (known good) | 23522 | 57.0 | 1.053 / 0.168 | 0.0% / 0.225% | 5.03s |

Both hudson files have **implausible-RR% at or below kishor's** (the file that
behaved correctly). If under-detection were happening at any meaningful rate,
implausible-RR% would be markedly elevated, not equal-or-lower. The RR
histograms for both hudson files (`rr_hist_*.png`) are **unimodal**, centered
tightly around 1.3-1.4s -- no second mode near 2x that value, which is the
signature a real missed-beat problem would leave (a "doubled" second peak).
The waveform plots (`waveform_*_normal.png`) show every visible beat correctly
flagged with a red marker, beat after beat, at a consistent ~1.3s spacing.

**Conclusion for the hudson files: R-peak detection is working correctly.**
The low, stable ~45 bpm is a genuine, sustained measurement, not a detection
artifact. PhysioNet's training distribution has mean HR 68 bpm (std 11.8) --
45 bpm sits close to 2 standard deviations below that mean, near the edge of
what the model saw during training. The most likely explanation is that the
model reads a **sustained, unusually low resting heart rate** (plausible for
a fit/athletic young subject, especially if this segment is near sleep onset)
as bradycardia consistent with its learned apnea/CVHR pattern, even though
there is no real cyclic surge-and-drop -- this is a **generalization failure
on an atypical-but-real physiological range**, not a bug, and not a bad
capture. It should be logged as a known model limitation (the training data's
age/clinical-population skew), not something to fix by adjusting a threshold.

The `biggest_gap` plots for both hudson files land on a single isolated
motion/artifact spike (visible as a burst up to +/-10,000-40,000, vs a normal
beat's ~100-150 amplitude) -- that is what produced the (still tiny, 1-3 beat)
"implausible" tail, not a systematic issue.

## Hari (`user_2406Hari...`): a different, real cause

Hari's implausible-RR% (0.896%) is close to but under the NOISY_ECG threshold
(1.0%), and its RR histogram shows a **small secondary mode around 0.5-0.7s**
alongside the main ~1.0-1.2s mode -- consistent with occasional split/double
beat detection during a noisier stretch. The `biggest_gap` waveform plot
(`waveform_user_2406Hari..._biggest_gap.png`) shows exactly this: a clean,
regular ECG (~1s spacing, correctly detected) on either side of a **genuine
~10-15s noise burst** (amplitude spikes to 400-500 vs a normal ~50-100),
during which beat detection becomes locally unreliable.

Critically, this file's IMU-based motion gate (`compute_motion_mask`, from
`src/effort/envelope.py`) flags only **0.09%** of the recording -- essentially
nothing. **The chest was not moving during this burst**, so the existing
accelerometer-based data-quality gate does not and cannot catch this kind of
noise; it must be an electrode-contact or EMG/muscle-tension artifact that is
purely electrical, invisible to the IMU. This is a genuine, localized
data-quality problem in a small fraction of the recording (real cause (b):
a bad *segment*, not the whole capture, and not a code bug), and it also
surfaces a real gap in the project's current tooling: **there is no
ECG-signal-based noise/quality gate, only an IMU-based motion gate**, and
they are not interchangeable.

## Verdict against the three candidates

- **(a) R-peak under-detection on lower-amplitude/noisier signal** -- ruled
  out for both hudson files (implausible-RR% at or below the known-good
  file's; unimodal RR histograms; every visible beat correctly marked).
- **(b) Genuinely bad capture / electrode contact issue** -- confirmed for
  Hari, but only for a short (~10-15s) segment, not the whole file; not
  present in either hudson file.
- **(c) Loader/channel-handling bug specific to these files** -- ruled out;
  the loader parses all four files identically (see `src/ingest/sensortile.py`,
  unchanged) and the raw QVAR signal for all four is well-formed (no
  saturation, no flatlines -- see `summary.csv`).
- **New finding, not in the original three candidates**: the hudson files'
  implausible model output is best explained by a **physiological HR range
  outside PhysioNet's training distribution** (sustained bradycardia in a
  healthy young subject), which the frozen model was never trained to
  distinguish from pathological bradycardia. This is a model-generalization
  limitation, to be documented, not a data or code problem to fix here.

## Files in this directory
- `summary.csv` -- the quantitative table above, all four files.
- `waveform_<file>_normal.png` -- a representative 45s window with detected
  R-peaks overlaid.
- `waveform_<file>_biggest_gap.png` -- a 45s window centered on the single
  largest RR interval in that recording.
- `rr_hist_<file>.png` -- full-recording RR-interval histogram with the
  plausibility bounds (`RR_MIN_S`/`RR_MAX_S` from `src/ecg/rpeaks.py`) marked.

No thresholds, the deployable model, or any detection code were changed to
produce these diagnostics or this write-up.
