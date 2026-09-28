# Project Progress — Cardiorespiratory Sleep Apnea Detection

**Last updated:** 2026-09-28
**Status at a glance:** Phase 0–3 done (ECG branch, classical baseline + CNN), plus a frozen deployable ECG model. An exploratory 8-method ECG feature-extraction comparison study (§4b) confirmed Phase 3's choice was sound and found one viable alternative (QRS-area EDR). Phase 4 (effort/IMU branch) started against synthetic data only. Real-hardware validation and fusion (Phase 5+) blocked on SensorTile hardware.

This document is the state of the project: what exists, what the numbers actually are, how each result was produced, and what isn't done yet. For the full project rationale (motivation, team, hardware, constraints), see `CLAUDE.md` in the project root — this file assumes that context and focuses on **what has actually been built and measured**.

---

## 1. Project overview

We're building a wearable sleep apnea *screening* tool (not a diagnostic device) that fuses two signals from a single chest-worn sensor:

- **ECG** — heart electrical activity, sampled via an ST SensorTile.box PRO's QVAR channel wired to chest electrodes.
- **Respiratory effort** — chest-wall motion from the same device's onboard IMU (accelerometer/gyro).

The scientific claim to prove is that **fusing heart + breathing signals beats using either alone** for detecting apnea (breathing pauses during sleep).

**The core constraint that shapes everything:** our own test subjects (2 students) are healthy 19–20 year-olds who essentially don't have apnea. So:
- Training data must be **public, expert-labelled clinical data** — the only place real apnea exists for us.
- Our own SensorTile recordings are for **validation only** — voluntary breath-holds (awake, timestamped) as safe labelled apnea-like events, plus healthy overnight recordings as negative controls. **We never train on our own data.**

**Architecture — two branches, decision-level fusion** (not one end-to-end model, because the two branches train on completely different data and decision-level fusion keeps each branch's output interpretable):

```
ECG branch (trained on PhysioNet):  QVAR/ECG → R-peaks → RR-intervals → CVHR features → [classical | CNN] → P(apnea)/min
Effort branch (signal processing,   IMU → accel magnitude → bandpass → Hilbert envelope → breathing rate / effort amplitude
  no training — no public IMU data exists for apnea):
                                                        ↓                    ↓
                                                  DECISION-LEVEL FUSION → per-night report (events/hour, severity)
```

**Most of this document is the ECG branch** (the left half of that diagram). The effort branch (right half) is signal-processing, not a trained model — it has now been built and checked against a synthetic signal (§3–4, Phase 4), but not against real IMU data yet, since that needs SensorTile hardware (see §7).

---

## 2. Data — PhysioNet Apnea-ECG

Downloaded to `data/physionet/` (328 files). This is a single-lead ECG database, **not** the Sleep-EDF dataset (that's the sister team's sleep-staging track — different files, different problem).

- **Sampling rate: 100 Hz** (confirmed empirically, see Phase 1 below).
- **35 labelled records** used for training/evaluation, split by the database's own naming convention:
  - `a01`–`a20` — apnea group (heavy apnea burden)
  - `b01`–`b05` — borderline group (low-to-moderate apnea burden)
  - `c01`–`c10` — control group (healthy, ~0 apnea minutes)
- **35 withheld test records** (`x01`–`x35`) — no `.apn` labels, not used anywhere in this project so far.
- Each labelled record has: `.dat` (ECG signal), `.hea` (header/metadata), `.apn` (**per-minute** apnea/normal label — the label granularity every "per-minute" result below matches), `.qrs` (precomputed R-peaks, not used — we detect our own, see Phase 1).

**Known open issue, not yet reconciled:** our own SensorTile hardware samples QVAR (wired as ECG) at **240 Hz**, not PhysioNet's 100 Hz. Per CLAUDE.md's methodology rules, RR-interval features are kept in **time units (seconds)** specifically so this transfers across sample rates — but this hasn't been exercised yet since we have no 240 Hz data flowing through the pipeline. This becomes live work in Phase 5 (own-hardware ingestion).

---

## 3–4. What's been built, phase by phase

### Phase 1 — Ingestion (loader + R-peak detection)

**Files:**
- `src/ingest/physionet.py` — loads one record's ECG signal + per-minute `.apn` labels via `wfdb`. `ApneaRecord` dataclass; `load_record(name)`.
- `src/ecg/rpeaks.py` — R-peak detection via `neurokit2` (`ecg_clean` + `ecg_peaks`). `RPeakResult` dataclass holds R-peak sample indices, RR-intervals (seconds), R-peak amplitudes (added in Phase 3 for the CNN's second channel — see below), mean HR, and % of physiologically-implausible RR intervals (outside 30–220 bpm).
- `src/ecg/features.py` — per-minute CVHR/HRV feature extraction (see Phase 2).

**Reproduce:**
```
python src/ingest/physionet.py     # loader sanity check
python src/ecg/rpeaks.py           # R-peak detection sanity check
python src/ecg/features.py         # CVHR feature sanity check
```

**Verified results:**

| Check | a01 | c01 |
|---|---|---|
| Sampling rate | 100.0 Hz | 100.0 Hz |
| Labelled minutes (apnea / normal) | 470 / 19 | 0 / 484 |
| R-peaks detected | 29,992 beats | 27,719 beats |
| Mean HR | 60.9 bpm | 58.0 bpm |
| Implausible RR intervals | 0.00% | 0.02% |

Per-minute CVHR sanity check on a01 (before building the full pipeline): apnea minutes showed a mean HR-swing of **36.37 bpm** vs **26.63 bpm** for normal minutes (SDNN 138.5ms vs 80.9ms) — confirmed the expected CVHR fingerprint (bradycardia during the pause, tachycardia on resumption) was present in the data before investing in Phase 2.

---

### Phase 2 — Classical CVHR baseline

**Goal:** establish an honest, simple, interpretable "floor" that any later model must beat.

**Files:**
- `src/ecg/features.py` — `compute_minute_features()` extracts 7 features per labelled minute: `mean_hr`, `hr_swing` (max−min instantaneous HR within the minute — the core CVHR fingerprint), `sdnn_ms`, `rmssd_ms`, `pnn50`, `rr_mean_ms`, `hr_delta_prev` (|mean_hr[minute] − mean_hr[minute−1]|, catching cyclic minute-to-minute swings that a single window can miss). `build_dataset()` runs this across many records.
- `src/eval/splits.py` — `list_labelled_records()` (the 35 a/b/c records), `leave_one_subject_out()` (generator yielding train/test record splits), `record_group()` (official apnea-group-a/b vs control-group-c label).
- `src/eval/metrics.py` — `binary_metrics()` computing accuracy, sensitivity, specificity, precision, F1 (accuracy alone is misleading given class imbalance — see §5).
- `src/ecg/cvhr_baseline.py` — main script: logistic regression (`sklearn`, `class_weight="balanced"`, features standardized) evaluated with 35-fold leave-one-subject-out cross-validation.

**Reproduce:** `python src/ecg/cvhr_baseline.py` (builds/caches features to `results/phase2_cvhr_baseline/features.csv` on first run, ~30s; LOSO training itself is near-instant for logistic regression).

**Result — per-minute, subject-independent (LOSO), 16,949 usable minutes across all 35 records** (96 minutes dropped for having fewer than 3 beats or being a record's first minute, where the `hr_delta_prev` feature is undefined):

```
accuracy = 0.711   sensitivity = 0.637   specificity = 0.758   precision = 0.621   F1 = 0.629
```

**Per-recording** (crude heuristic: flag a recording "apnea" if >5% of its minutes are predicted apnea, vs the official apnea-group-a/b vs control-group-c label): 26/35 = 0.743 — all 9 errors were control recordings misclassified as apnea. Investigated and found **not fixable by choosing a better threshold**: sorting all 35 records by predicted apnea-fraction shows apnea-group and control-group recordings heavily overlap (e.g. control `c08`=45% vs apnea `b05`=12%) — a real limitation of this crude per-minute-fraction approach, not a tuning miss. Treated as a known-weak diagnostic, not a claim.

Outputs: `results/phase2_cvhr_baseline/features.csv`, `per_minute_predictions.csv`, `per_recording_predictions.csv`.

---

### Phase 3 — 1D-CNN (done, closed)

**Goal:** beat the classical floor using RR-interval sequences directly (not hand-crafted features), per CLAUDE.md's "start simple" rule — one small architecture, escalated only as needed, never straight to the literature-ceiling model.

Went through three iterations; documented in full (with the ablation) in **`results/phase3_cnn/README.md`**, summarized here.

**Files:**
- `src/ecg/rr_sequence.py` — resamples each minute's RR-interval sequence (variable beat count, since bradycardia/tachycardia during apnea changes how many beats fall in a minute) onto a fixed 60-point grid, in seconds. Same per-minute windows and beat-count threshold as `features.py`, so the CNN and classical baseline are evaluated on directly comparable minute sets.
- `src/ecg/rr_amp_sequence.py` — same idea, but produces a **second channel**: R-peak amplitude (from the cleaned ECG signal, an ECG-derived-respiration proxy), aligned and resampled the same way.
- `src/ecg/cnn_model.py` — `RRCNN`: 2 conv layers (16→32 channels) + global average pool + 2 FC layers, supports 1 or 2 input channels. `train_cnn()`/`predict_cnn()` helpers; `pos_weight_factor` parameter multiplies the natural class-balance loss weight (for the sensitivity-boosting experiment below).
- `src/ecg/cnn_apnea.py` — **v1**: 1-channel (RRI only), default class-balanced loss.
- `src/ecg/cnn_apnea_v2.py` — **v2**: 2-channel (RRI + R-amplitude), loss weighted 1.5× toward apnea (an attempt to fix v1's weak sensitivity).
- `src/ecg/cnn_apnea_ablation.py` — isolates v2's two changes: **Variant A** (2-channel, default weight) and **Variant B** (1-channel, 1.5× weight), each run separately.
- `src/ecg/cnn_apnea_final.py` — reproduces the **adopted final config** (2-channel, default weight = Variant A) for future reproducibility; not re-run after adoption since the ablation already produced the validated numbers.

All four main scripts use the same 35-fold LOSO split and the same exact 16,949-minute set as Phase 2 (explicitly joined against Phase 2's cached feature table to guarantee comparability — the raw RR-sequence extraction alone would keep 16,991 minutes; the difference is Phase 2's `hr_delta_prev` feature dropping each record's first minute via a pandas `.diff()`).

**Reproduce:**
```
python src/ecg/cnn_apnea.py             # v1
python src/ecg/cnn_apnea_v2.py          # v2
python src/ecg/cnn_apnea_ablation.py    # variants A and B
python src/ecg/cnn_apnea_final.py       # reproduces the adopted result (not yet re-run)
```
Each is a genuine 35-fold LOSO training run (fresh model per fold) — **not fast**. Observed wall-clock time varied a lot on this laptop CPU (from ~14 min to ~50 min for the same script), apparently due to thermal throttling under sustained load, not the code.

**Progression (per-minute, LOSO, same 16,949 minutes throughout):**

| Step | Input | Loss weighting | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Phase 2 baseline | hand-crafted features (logistic regression) | balanced | 0.711 | 0.637 | 0.758 |
| v1 CNN | RRI sequence, 1 channel | balanced | 0.775 | 0.653 | 0.851 |
| v2 CNN | RRI + R-amplitude, 2 channels | balanced × 1.5 | 0.798 | 0.754 | 0.825 |
| **FINAL — Variant A** | **RRI + R-amplitude, 2 channels** | **balanced (default)** | **0.812** | **0.756** | **0.846** |

**Ablation finding** (why the final config drops v2's loss weighting): isolating v2's two changes showed the **R-amplitude channel does essentially all the work**. Variant A (2ch, default weight) beats v2 on *every* metric; the 1.5× weighting alone (Variant B: 1ch, weighted) is actually a net loss vs. v1 (accuracy 0.775→0.758) — it buys a little sensitivity at a much larger specificity cost, and combining it with the amplitude channel (v2) doesn't compound gains, it just gives away specificity the amplitude channel wasn't asking to give away.

| Variant | Accuracy | Sensitivity | Specificity |
|---|---|---|---|
| v1 (1ch, default) | 0.775 | 0.653 | 0.851 |
| **A: 2ch, default (= FINAL)** | **0.812** | **0.756** | **0.846** |
| B: 1ch, 1.5× weight | 0.758 | 0.715 | 0.786 |
| v2: 2ch, 1.5× weight | 0.798 | 0.754 | 0.825 |

**Per-recording** (same crude threshold as Phase 2): 28/35 = 0.800, essentially unchanged across v1/v2/final — consistent with Phase 2's finding that this metric doesn't discriminate well regardless of the underlying per-minute model.

**A caught-and-fixed process bug worth recording:** the first CNN training attempt used 15 epochs and was clearly undertrained (68% train accuracy, predicting far fewer apnea minutes than the true rate even on its own training data) — caught by checking the loss curve before trusting the number, fixed by increasing to 30 epochs. Also: a bug in the delta-vs-baseline print statement (`cnn_accuracy` instead of `cnn_accuracy - baseline_accuracy`) produced an absurd "+0.775 point improvement" the first time v1 finished — caught before reporting it, since a jump that size is obviously wrong for this domain.

Outputs: `results/phase3_cnn/` (README + final predictions + historical v1), `results/phase3_cnn_v2/`, `results/phase3_cnn_ablation/`.

---

### Deployable ECG model (frozen after Phase 3)

**Goal:** every Phase 3 number is a cross-validation estimate — 35 models trained and thrown away, one per LOSO fold. None of those weights were ever saved. This step produces the first (and so far only) actual weight file: the model to run on our own QVAR recordings once hardware is back.

**Files:**
- `src/ecg/train_deployable_model.py` — trains **one** `RRCNN` (2-channel, default weighting — the adopted Phase 3 config) on **all 35 records, no held-out split**, then saves it.

**Reproduce:** `python src/ecg/train_deployable_model.py` (~20–90s; a single training pass, not LOSO).

**Result:** training-set fit accuracy 0.834 (not a generalization estimate — it has seen every minute it's checked against; expected to look better than the 0.812 LOSO number for exactly that reason).

Outputs, all in `results/phase3_cnn/deployable/`:
- `model_state_dict.pt` — the weights.
- `norm_stats.json` — per-channel mean/std (RR-interval: μ=0.926s, σ=0.386s; R-amplitude: μ=1.256, σ=0.865) required to normalize any new input before inference.
- `config.json` — architecture/training config + the Phase 3 LOSO numbers for reference.
- `README.md` — explains this is a deployment artifact not an evaluation, gives a load/use code snippet, and flags an **untested transfer risk**: the R-peak amplitude channel's scale comes from PhysioNet's calibrated ECG, while our QVAR channel is an electrostatic sensor with likely very different amplitude units and morphology. The RR-interval channel (time-based) should transfer more safely. This has not been checked against real QVAR data — there isn't any yet.

---

### §4b — ECG feature-extraction comparison study (exploratory, methods 3–8)

**Goal:** Phase 3 closed with one adopted feature representation (RR-interval +
R-peak amplitude, 2-channel 1D-CNN, 81.2%). This exploratory study asks
whether other ECG-derived feature representations do better, worse, or about
the same, under the identical evaluation protocol — not to reopen Phase 3
(still closed, no further tuning), but to characterize the representation
space before Phase 5's own-hardware work commits to one input format.

**Method:** 8 feature representations, all derived from the same single-lead
ECG, each evaluated with genuine 35-fold LOSO cross-validation on the same
16,949-minute set (a few fewer where a method's own features are undefined —
noted per method below):

| # | Method | Classifier |
|---|---|---|
| 1 | RR-intervals (= Phase 3 v1) | 1D-CNN |
| 2 | RR + R-peak amplitude (= Phase 3 **adopted final**) | 1D-CNN |
| 3 | Time-domain HRV bank (SDNN, RMSSD, pNN50, mean HR, HR range, triangular index) | best of 1D-CNN / logistic regression / small MLP |
| 4 | Frequency-domain HRV (VLF/LF/HF power, LF:HF ratio) | best of the same 3 classifiers |
| 5 | Nonlinear/Poincaré (SD1, SD2, SD1:SD2, sample entropy) | best of the same 3 classifiers |
| 6 | RR + QRS-area (whole-complex integral EDR proxy, vs. method 2's single-sample amplitude) | 1D-CNN |
| 7 | CWT scalogram of the RR series | 2D-CNN |
| 8 | Spectrogram of the raw ECG segment (the only method not derived from R-peaks at all) | 2D-CNN |

**Controlled-comparison design, and a correction made mid-study:** methods
1/2/6 share one 1D-CNN (`RRCNN`, unmodified from Phase 3) since they're
genuine within-minute time series; methods 7/8 share one 2D-CNN
(`ScalogramCNN`) since they're genuine images. Methods 3/4/5 are *scalar*
per-minute feature vectors (no time axis) — the first pass forced them
through `RRCNN` anyway (treating the vector as a 1-channel sequence) for a
strict single-classifier comparison, but this was flagged as handicapping
feature types that don't suit a convolutional architecture. Fixed by adding
logistic regression and a small MLP as alternatives and keeping the best
classifier per bank (logistic regression won all three, confirming the
concern — e.g. method 4's specificity recovered from 0.293 under the CNN to
0.772 under logistic regression). The master table reports each bank's best
classifier, not the CNN's number, and names which classifier won.

**Files:**
- `src/ecg/hrv_bank_features.py` — per-minute time/frequency/nonlinear HRV
  feature banks (methods 3/4/5), cached to `results/feature_study/hrv_bank_features.csv`.
- `src/ecg/feature_bank_cnn.py` — LOSO runner for methods 3/4/5 across all
  three classifiers (`cnn`/`logreg`/`mlp`); `cnn_model.py` gained
  `FeatureMLP`/`train_mlp`/`predict_mlp` for the MLP option.
- `src/ecg/qrs_area.py`, `rr_qrsarea_sequence.py`, `method6_qrsarea_cnn.py` — method 6.
- `src/ecg/cwt_scalogram.py`, `cnn2d_model.py`, `method7_cwt_cnn.py` — method 7
  (PyWavelets complex Morlet CWT; `scipy.signal.cwt`/`morlet2` were removed
  in this project's scipy version).
- `src/ecg/ecg_spectrogram.py`, `method8_spectrogram_cnn.py` — method 8
  (scipy STFT, log-magnitude).

**Reproduce:** each method has its own runner script (see the table above's
file list); `python src/ecg/feature_bank_cnn.py <time|freq|nonlinear>` for
methods 3–5. Each is a genuine 35-fold LOSO run — the 2D-CNN image methods
(7, 8) are notably slower (~31 min and ~101 min wall-clock respectively on
this machine's CPU) than the 1D methods (~5–8 min each).

**Result — master table** (`results/feature_study/master_results.csv`/`.md`;
per-minute, subject-independent LOSO):

| # | Method | Family | Classifier | Acc | Sens | Spec | Prec | F1 | Rec-Acc |
|---|---|---|---|---|---|---|---|---|---|
| 1 | RR-intervals | time | 1D-CNN | 0.775 | 0.653 | 0.851 | 0.732 | 0.691 | 0.800 |
| 2 | RR + R-amplitude (**adopted, Phase 3 final**) | time | 1D-CNN | **0.812** | 0.756 | 0.846 | 0.754 | 0.755 | 0.800 |
| 3 | Time-domain HRV bank | time | logistic regression | 0.702 | 0.620 | 0.754 | 0.611 | 0.615 | 0.743 |
| 4 | Frequency-domain HRV | frequency | logistic regression | 0.672 | 0.513 | 0.772 | 0.584 | 0.546 | 0.743 |
| 5 | Nonlinear/Poincaré | nonlinear | logistic regression | 0.707 | 0.673 | 0.728 | 0.617 | 0.644 | 0.743 |
| 6 | RR + QRS-area | time | 1D-CNN | 0.798 | 0.687 | 0.868 | 0.764 | 0.724 | **0.829** |
| 7 | CWT scalogram of RR | image | 2D-CNN | 0.749 | 0.491 | 0.911 | 0.774 | 0.601 | 0.771 |
| 8 | Raw ECG spectrogram | image | 2D-CNN | 0.534 | 0.673 | 0.448 | 0.432 | 0.526 | 0.800 |

All 8 per-minute accuracies sit well under the project's 0.95
leakage-suspicion threshold (§5) — none triggered it, including the
lopsided sensitivity/specificity splits in methods 4, 7, 8, each of which
has an identifiable, documented cause (below), not silent acceptance.

**Findings:**
1. **Real within-minute time series beats aggregated scalars, at any
   classifier.** Methods 1/2/6 (genuine per-beat sequences into the CNN) are
   the only ones over 0.79 accuracy; methods 3/4/5 (a minute collapsed to a
   handful of numbers first) top out around 0.70 — close to Phase 2's
   original classical floor (0.711) — regardless of which of the three
   classifiers is used.
2. **R-peak amplitude isn't uniquely special as an EDR proxy.** Method 6
   (QRS-area, a whole-complex integral) comes within 1.4 accuracy points of
   the adopted method 2 and posts the **best per-recording accuracy in the
   study** (29/35 = 0.829) — a genuinely competitive alternative, not just a
   near-miss.
3. **Frequency-domain HRV is the weakest scalar bank, as predicted before
   the run:** `hrv_bank_features.py` flags in its own docstring that VLF
   needs ~5.5 minutes of data to resolve and LF is borderline at a 60s
   window — method 4 is the worst-performing non-image method under every
   classifier tried.
4. **Image representations trade sensitivity for specificity, sharply.**
   Method 7 (CWT of RR — same information as method 1, reprojected) swings
   to spec=0.911/sens=0.491; method 8 (raw ECG spectrogram, the only method
   not derived from R-peaks) swings the other way (sens=0.673/spec=0.448)
   and is the weakest per-minute performer in the whole study, though its
   per-recording accuracy (0.800) still holds up.
5. **This does not reopen Phase 3.** The adopted 81.2% model (method 2)
   remains the best per-minute result in the study; nothing here changes
   the "Phase 3 is done, no further tuning" decision (§7). The practical
   takeaway is for Phase 5: QRS-area (method 6) is worth keeping as a
   fallback EDR signal if R-peak amplitude's amplitude-scale transfer risk
   (§6) turns out to be a real problem on QVAR hardware.

**Honest limitations specific to this study:**
- **35 subjects is a small dataset for 2D-CNNs on 2,000–6,000-pixel images**
  (methods 7/8) even more than it was for the original 1D-CNN — taken more
  seriously here since these architectures have more capacity relative to
  the data than RRCNN did.
- **Frequency-domain features (method 4) are flagged as unreliable, not
  merely weak** — report for completeness (it was an explicit ask), not
  because vlf_power in particular should be trusted as a real physiological
  estimate at this window length.
- **The scalar-bank-as-1D-CNN-input design was a deliberate, debatable
  adaptation**, corrected mid-study once flagged — see "Controlled-comparison
  design" above. The master table reports the winning classifier per bank,
  named explicitly, specifically so this doesn't get silently averaged away.
- Two of the largest intermediate caches (CWT scalograms, 125MB; ECG
  spectrograms, 250MB) exceed GitHub's 100MB file limit and are gitignored
  rather than committed — both regenerate automatically (~15–25s) the first
  time their runner script is run without a cache present.

Outputs: `results/feature_study/` — `master_results.csv`/`.md` (this
table), `hrv_bank_features.csv` (cached scalar features), and one
subdirectory per method (`method3_time_domain/{cnn,logreg,mlp}/` etc.) with
per-minute/per-recording predictions and, for methods 3–5, a
`classifier_comparison.md` showing all three classifiers' full metrics.

---

### Phase 4 — Effort branch (built, validated on synthetic data only)

**Goal:** implement the IMU → breathing signal pipeline from CLAUDE.md's architecture diagram. Pure signal processing, no training — there is no public dataset pairing IMU with expert apnea labels, so unlike the ECG branch this cannot be trained or cross-validated the same way; it can only be checked against known respiratory physics (synthetic data, done here) and, later, our own breath-hold recordings (Phase 5, needs hardware).

**Files:**
- `src/effort/envelope.py` — `accel_magnitude()` (3-axis → orientation-invariant magnitude) → `compute_motion_mask()` (flags gross body movement, a data-quality gate, via rolling std of raw magnitude) → `bandpass_filter()` (zero-phase Butterworth, 0.08–0.6 Hz = 4.8–36 breaths/min) → `hilbert_envelope()` (instantaneous amplitude + phase) → `instantaneous_breathing_rate_bpm()` (from phase derivative) → `effort_amplitude_trace()` (envelope smoothed over ~10s, for reporting a coarse trend). `compute_effort_signal()` runs the full chain; `EffortSignal` dataclass holds every intermediate signal.
- `src/effort/cessation.py` — `detect_cessations()`: flags contiguous stretches where effort drops below 30% of a rolling local baseline for ≥10s (a candidate breathing pause).
- `src/effort/synthetic_demo.py` — generates a synthetic 6-minute, 50 Hz accelerometer signal (15 bpm breathing + gravity + noise) with 4 inserted flat "breath-hold" gaps (20s, 15s, 12s, and a 7s gap deliberately below the 10s detection threshold), runs it through the full pipeline, and checks detected events against the known ground truth.

**Reproduce:** `python src/effort/synthetic_demo.py`

**Result:**
```
Estimated breathing rate: 15.0 bpm (true = 15.0 bpm)
gap 60-80s   (20s): detected    [OK]
gap 150-165s (15s): detected    [OK]
gap 220-227s (7s):  not detected [OK]  -- correctly below the 10s threshold
gap 300-312s (12s): detected    [OK]
```
All four gaps classified correctly (three real cessations found with sensible durations, the sub-threshold gap correctly ignored) and the breathing-rate estimate matched the synthetic ground truth exactly.

**A caught-and-fixed bug worth recording, same discipline as Phase 3's:** the first run detected only 1 of 3 real gaps. Investigated rather than tuned around — traced to feeding `detect_cessations()` the `effort_amplitude` trace, which is smoothed with a ~10s window for reporting purposes. That smoothing window is comparable to (or larger than) the 10–20s events being detected, and blurred away their edges enough to shrink a real 15s gap's *detected* duration to ~8.6s (under the 10s threshold). Fix: cessation detection now runs on the raw (unsmoothed) Hilbert envelope instead; `cessation.py` and `envelope.py`'s docstrings both now flag this explicitly so the mistake isn't repeated.

**What this validates and what it doesn't:** confirms the signal-processing chain is internally consistent and behaves as expected on a clean synthetic signal with known ground truth. It does **not** validate anything about real chest-wall accelerometer data — real IMU signal will have different noise characteristics, motion artifacts, and possibly a different useful frequency band than this idealized sine-wave-plus-noise model. That validation is Phase 5's job, once hardware is back.

Outputs: none persisted to `results/` yet (this is a code-correctness check, not an experiment with a result to archive) — running `synthetic_demo.py` reproduces the numbers above on demand.

---

## 5. Methodology decisions and why

**Subject-independent evaluation (leave-one-subject-out), always.** Every Apnea-ECG record is a different patient. If a model sees some of a patient's minutes during training and is tested on other minutes from the *same* patient, it can partly succeed just by memorizing that patient's personal baseline HR and HRV rather than learning the general apnea signature — the literature gap between mixed-subject splits (~100%) and subject-independent splits (~86%) on this exact task is the standard illustration of how badly that inflates results. Every LOSO fold in this project holds out one entire record; no subject is ever in both train and test within a fold. This is why we can trust the numbers above enough to report them.

**Per-minute and per-recording are reported separately, never conflated.** They're different tasks answering different questions: per-minute asks "did apnea happen in this specific 60-second window" (the granularity the AHI severity index is ultimately built from); per-recording asks "does this person have apnea at all" (a coarser triage question). Phase 2 showed directly why this separation matters: a per-minute model can be reasonably discriminative while a naive per-recording rollup of its predictions still fails to cleanly separate groups — reporting only one of the two numbers would have hidden that.

**Start simple, escalate only if it plateaus.** Classical CVHR (logistic regression on hand features) before a 1D-CNN, and a small 2-layer CNN before anything bigger (no CNN-LSTM, no Transformer). Two concrete payoffs so far: (1) the classical baseline gave a real, cheap number to beat, so the CNN's value could be measured rather than assumed; (2) building the CNN incrementally (1 channel → 2 channels → weighting) made the ablation possible — if all three changes had shipped in one leap, we'd never have learned that the loss-weighting idea didn't actually help.

**Suspiciously good numbers are treated as bugs to investigate, not results to report.** Every LOSO script has an explicit check that flags accuracy above 0.95 as needing a subject-leakage check before being trusted (none of our real numbers have tripped it — 81.2% is the highest, comfortably below the flag). This isn't paranoia for its own sake: the actual value of a >95% number in this specific setup would almost always be leaked test data, not a genuinely better model, given the literature's own ceiling is ~87–94% even with much more sophisticated architectures.

**Suspiciously *bad* results get the same treatment, not just suspiciously good ones.** When the effort branch's first synthetic test missed 2 of 3 known-real breathing-pause gaps, the response was to debug the actual signal values at the failure points (not to loosen the detection threshold until the test passed) — which surfaced a genuine bug (an over-smoothed input signal blurring event edges) rather than a threshold that happened to be miscalibrated. A test tuned to pass would have hidden that bug on real data, where the ground truth isn't known in advance.

---

## 6. Honest limitations

- **LOSO is a cross-validation protocol, not a saved model** — the 81.2%/75.6%/84.6% numbers describe the *method's* expected performance on an unseen subject, from 35 models trained and discarded, one per fold. A deployable model has now been frozen (see "Deployable ECG model" above), but its own accuracy on new data is unmeasured by construction (no held-out set) — the LOSO numbers are the best available estimate of how it should perform, not a measurement of this specific model.
- **The per-recording metric is a crude heuristic**, not a validated clinical rule: "flag apnea if >5% of predicted minutes are apnea." Phase 2's own investigation showed apnea-group and control-group recordings overlap heavily under this rule regardless of threshold choice — a proper AHI-style per-recording metric is Phase 7's job, not something solved here.
- **Sensitivity (75.6%) still misses about 1 in 4 true apnea minutes.** For a screening tool, missing real apnea is generally worse than a false alarm, so this remains the weakest part of the current model even after the iteration that targeted it specifically.
- **81.2% is below the 85–88% literature target** for 1D-CNN single-lead apnea detection (Chang et al. 2020). That target reflects more tuning and architecture work (e.g. more epochs, deeper nets, more careful hyperparameter search) that hasn't been attempted — per the "start simple" rule, we stopped at the first config that clearly beat the floor and made an honest, ablation-checked improvement, not the best config achievable.
- **No own-hardware data anywhere in this pipeline yet.** Every ECG-branch number in this document comes from the public PhysioNet dataset; the effort branch has only been run against a synthetic sine-wave-plus-noise signal. That's correct per the project's core constraint (never train on our own recordings), but it also means we genuinely don't yet know how either branch behaves on real sensor data — our own 240 Hz QVAR-derived ECG has different noise characteristics than a clinical single-lead recording, and a real chest accelerometer signal has motion artifacts and morphology a synthetic sine wave doesn't.
- **The deployable ECG model's R-peak amplitude channel may not transfer to QVAR data.** Its scale comes from PhysioNet's calibrated ECG; our QVAR channel is an electrostatic sensor, likely with very different amplitude units and morphology. Untested — flagged in `results/phase3_cnn/deployable/README.md`, not yet resolved.
- **The 100 Hz vs 240 Hz sample-rate mismatch is not yet reconciled in code.** RR-interval features are kept in time units specifically so this should transfer, but that claim hasn't been tested against real 240 Hz data.
- **The effort branch's frequency band (0.08–0.6 Hz) and cessation thresholds (30% collapse, 10s minimum, 60s baseline window) are physiologically-reasoned defaults, not fitted to any data** — there's no labelled IMU dataset to fit them against. They worked on the synthetic test by construction; real chest-motion data may call for different values.
- **35 subjects is a small dataset for a CNN.** The architecture was deliberately kept tiny for exactly this reason; a larger or deeper model would likely overfit further before it improved.

---

## 7. Current status and what's next

| Phase | Status |
|---|---|
| 0 — Environment | Done |
| 1 — Ingestion (loader + R-peaks) | Done |
| 2 — Classical CVHR baseline | Done (71.1% per-minute) |
| 3 — 1D-CNN | **Done, closed** (81.2% / 75.6% / 84.6% per-minute) |
| — Deployable ECG model | **Done** — frozen, untested on real QVAR data |
| §4b — ECG feature-extraction comparison study (8 methods) | **Done, exploratory** — confirms method 2 (adopted) is still best; QRS-area (method 6) is a viable fallback EDR |
| 4 — Effort branch (IMU → envelope → cessation detection) | **Built and validated on synthetic data only** — real-IMU validation blocked on hardware |
| 5 — Breath-hold validation on our own hardware | **Blocked** — needs SensorTile hardware; also where the 240 Hz reconciliation and the amplitude-channel transfer risk become live |
| 6 — Decision-level fusion (fused > ECG-only, the headline claim) | Blocked on 5 |
| 7 — Per-night report (events/hour, severity band, no-SpO2 caveat) | Blocked on 6 |
| 8 — Track B (MESA, multimodal + SpO2 + obstructive/central) | Stretch goal, blocked on NSRR data access |

Phase 4's synthetic-only build means the *code* for both branches now exists and is internally validated — what's left across Phases 4–7 is fundamentally the same hardware dependency (SensorTile, currently down, timeline unknown per CLAUDE.md): real IMU data to validate the effort branch against, and real QVAR recordings to test the deployable ECG model and the sample-rate reconciliation against. No further ECG-branch software work is planned per the "Phase 3 is done, no further tuning" decision.

---

## 8. File and directory map

```
apnea/
  CLAUDE.md                          Full project brief: motivation, hardware, data sources,
                                      architecture, methodology rules, phase plan. Read this first.
  docs/
    PROGRESS.md                      This file.

  data/
    physionet/                       PhysioNet Apnea-ECG database (328 files): a01-a20 (apnea),
                                      b01-b05 (borderline), c01-c10 (control) — each with
                                      .dat/.hea/.apn/.qrs — plus x01-x35 (withheld, no labels).
    recordings/                      (not yet created) — future home for our own SensorTile
                                      recordings; validation-only, never committed to git.

  src/
    ingest/
      physionet.py                   Loads one Apnea-ECG record's ECG signal + per-minute labels.
    ecg/
      rpeaks.py                      R-peak detection (neurokit2) + RR-intervals + R-peak amplitudes.
      features.py                    Per-minute CVHR/HRV feature extraction (7 features) for the
                                      classical baseline; build_dataset() aggregates across records.
      cvhr_baseline.py                Phase 2: logistic regression + LOSO cross-validation.
      rr_sequence.py                 Builds fixed-length 1-channel (RRI) sequences for the CNN (v1).
      rr_amp_sequence.py             Builds fixed-length 2-channel (RRI + R-amplitude) sequences
                                      for the CNN (v2 / final).
      cnn_model.py                   RRCNN architecture + train/predict helpers (1 or 2 channels,
                                      adjustable loss weighting).
      cnn_apnea.py                   Phase 3 v1: 1-channel CNN, LOSO.
      cnn_apnea_v2.py                Phase 3 v2: 2-channel CNN + 1.5x sensitivity-boosted loss, LOSO.
      cnn_apnea_ablation.py          Ablation isolating v2's two changes (variants A and B).
      cnn_apnea_final.py             Reproduces the adopted final config (2ch, default weight).
      train_deployable_model.py      Trains ONE model on all 35 records (no held-out split) and
                                      saves weights + norm stats -- the actual deployment artifact.
      hrv_bank_features.py           §4b methods 3-5: per-minute time/freq/nonlinear HRV feature banks.
      feature_bank_cnn.py            §4b methods 3-5: LOSO runner across 3 classifiers (cnn/logreg/mlp).
      qrs_area.py                    §4b method 6: QRS-area EDR proxy (whole-complex integral).
      rr_qrsarea_sequence.py         §4b method 6: 2-channel (RR, QRS-area) sequences.
      method6_qrsarea_cnn.py         §4b method 6: LOSO runner, same RRCNN as the adopted model.
      cwt_scalogram.py               §4b method 7: CWT scalogram of the RR series (PyWavelets).
      cnn2d_model.py                 §4b methods 7-8: ScalogramCNN (2D-CNN) + train/predict helpers.
      method7_cwt_cnn.py             §4b method 7: LOSO runner.
      ecg_spectrogram.py             §4b method 8: STFT spectrogram of the raw ECG segment.
      method8_spectrogram_cnn.py     §4b method 8: LOSO runner.
    effort/
      envelope.py                    Accel magnitude -> motion mask -> bandpass (0.08-0.6Hz) ->
                                      Hilbert envelope -> breathing rate + effort amplitude.
      cessation.py                   Flags envelope collapses >=10s (candidate breathing pauses).
      synthetic_demo.py              Generates a synthetic breathing signal with known gaps and
                                      checks the envelope+cessation pipeline detects them.
    fusion/                          Empty — Phase 6, blocked on hardware + real effort-branch data.
    eval/
      splits.py                      list_labelled_records(), leave_one_subject_out(), record_group().
      metrics.py                     binary_metrics() (accuracy/sensitivity/specificity/precision/F1).

  notebooks/                         Empty — reserved for exploration only, nothing checked in yet.

  results/
    phase2_cvhr_baseline/
      features.csv                  Cached per-minute feature table, all 35 records.
      per_minute_predictions.csv    LOSO per-minute predictions (the 71.1% result).
      per_recording_predictions.csv Per-recording rollup (the 74.3% result).
    phase3_cnn/
      README.md                    Phase 3 summary: progression table + ablation finding.
      final_per_minute_predictions.csv     Adopted final model (Variant A) predictions.
      final_per_recording_predictions.csv  Adopted final model per-recording rollup.
      per_minute_predictions.csv    Historical v1 predictions (kept for the progression table).
      per_recording_predictions.csv Historical v1 per-recording rollup.
      sequences.npz                 Cached 1-channel RRI sequences (v1).
      deployable/
        model_state_dict.pt         Frozen weights: the deployable model (trained on all 35 records).
        norm_stats.json             Per-channel mean/std to normalize inputs before inference.
        config.json                 Architecture/training config + Phase 3 LOSO numbers for reference.
        README.md                   Explains this is a deployment artifact, how to load/use it, and
                                     the untested QVAR amplitude-scale transfer risk.
    phase3_cnn_v2/
      per_minute_predictions.csv    v2 predictions.
      per_recording_predictions.csv v2 per-recording rollup.
      sequences.npz                 Cached 2-channel sequences (reused by cnn_apnea_final.py).
    phase3_cnn_ablation/
      variant_a_2ch_default_weight.csv   Ablation Variant A predictions (= the adopted final result).
      variant_b_1ch_weighted.csv          Ablation Variant B predictions.
    phase3_cnn_run.log                    Raw stdout from the v1 training run (fold-by-fold timing).
    phase3_cnn_v2_run.log                 Raw stdout from the v2 training run.
    phase3_cnn_ablation_run.log           Raw stdout from the ablation run.
    feature_study/                        §4b: the 8-method feature-extraction comparison study.
      master_results.csv / .md            The master table (all 8 methods, one row each).
      hrv_bank_features.csv               Cached per-minute time/freq/nonlinear features (methods 3-5).
      method3_time_domain/{cnn,logreg,mlp}/    Method 3 predictions per classifier + classifier_comparison.md.
      method4_freq_domain/{cnn,logreg,mlp}/    Method 4, same structure.
      method5_nonlinear/{cnn,logreg,mlp}/      Method 5, same structure.
      method6_qrsarea_edr/                Method 6 predictions + cached sequences.
      method7_cwt_scalogram/              Method 7 predictions (scalograms.npz cache gitignored, regenerable).
      method8_ecg_spectrogram/            Method 8 predictions (spectrograms.npz cache gitignored, regenerable).
```
