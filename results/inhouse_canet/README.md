# In-house hardware track for CANet -- characterization, plausibility, deployability

**This is NOT an apnea-detection accuracy result.** The cohort is healthy 19-20 y/o with no apnea and no labels
(validation-only, never trained on, never committed). It tells us (1) whether our wearable can drive the pipeline
end to end, (2) how far accelerometer effort is from the belt signal CANet was trained on, (3) whether a healthy
negative control looks plausible. Nights are anonymised S01.. (no names/ids in any output).

Setup: 10 of 15 recordings are usable (>= 600 s; 5 are aborted captures of 1-32 s), 0.8-7.5 h each. Sleep state is NOT known
(no staging), so rates are per *scored recorded* hour, in event-epochs/h (a flagged 30 s epoch, not an AHI event).
Models: MESA-trained final models (`src/mesa/train_final.py`, MESA only; weights gitignored): ECG-only CardiacCNN, effort-only
CANet, headline CANet (ECG+effort). Cardiac stream = QVAR -> R-peaks -> MESA-identical grid/normalisation (per-night, so the earlier
108x amplitude-scale gap is neutralised by construction). Effort stream = accelerometer breathing trace (`src/effort/envelope.py`
magnitude, plus a principal-axis diagnostic) -> 32 Hz -> MESA-identical belt preprocessing, fed to both belt channels.

## 1. Deployability (`deployability/`)
The wearable pipeline runs end to end: QVAR+IMU -> both streams -> trained CANet -> per-epoch probability -> per-night rate
(10/10 nights, per-night CSV + per-epoch probabilities). Technically deployable. The numbers are not meaningful (see 3).

## 2. Belt -> accelerometer domain gap (`domain_gap/`)
- A classifier tells MESA belt epochs from in-house accelerometer epochs with AUROC **0.99** (handcrafted features) and **0.997-0.998**
  (effort-encoder embedding), subject/night-grouped CV; control (MESA half vs half) = 0.51 / 0.47.
- Breathing IS present in the accelerometer (spectral peak near 0.2-0.25 Hz, median ~16-18 bpm vs 14 for belts), but the signal is
  broadband and irregular: spectral entropy 0.64-0.66 vs 0.46 (night medians at the 87-98th percentile of MESA), zero-crossings 40-42 vs
  30 /min, breathing-rate spread IQR 10 vs 4 bpm, weaker low-frequency peak. Amplitude after normalisation is comparable (RMS ~0.9-1.0).
- Caveat: sleep/posture/activity of these recordings is unverified; part of the gap may be activity, not the sensor.

## 3. Healthy negative control (`negative_control/`)
Fraction of epochs flagged at the MESA operating points (median over 10 nights) vs MESA out-of-fold low-AHI subjects (a0h4 < 5, n=67):
| model | in-house | MESA low-AHI median [p90] | nights > MESA p90 |
|---|---|---|---|
| ECG-only | 0.58 | 0.43 [0.67] | 4/10 |
| effort-only (accel mag / pca) | 0.84 / 0.80 | 0.35 [0.55] | 10/10 |
| **headline CANet (accel mag / pca)** | **0.94 / 0.92** | 0.28 [0.48] | 10/10 |
| headline, effort = 0 / white noise | 0.69 / 0.63 | | 7/10, 9/10 |
| effort-only, white noise | 0.33 | 0.35 [0.55] | 0/10 |

**Verdict: implausibly HIGH -- a transfer failure, not a result.** Healthy 19-20 y/o should sit at or below the MESA low-AHI rate;
the headline model flags >90%. Isolation: the effort side dominates (effort-only 84% while pure noise gives 33%, i.e. the model reads the
structured-but-wrong-morphology accelerometer trace as disrupted breathing); the cardiac side is moderately elevated (58%, 4/10 nights above
MESA p90; young healthy hearts have large respiratory sinus arrhythmia, not verified as the cause); fusion compounds both. The model should not
be used on accelerometer effort without adaptation (e.g. belt-like calibration or a MESA-independent effort model). MESA belts remain the
only validated effort input.
