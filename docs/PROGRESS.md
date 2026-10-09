# Project Progress — Cardiorespiratory Sleep Apnea Detection

**Last updated:** 2026-10-09
**Status at a glance:** Phase 0–3 done (ECG branch, classical baseline + CNN), plus a frozen deployable ECG model. An exploratory 8-method ECG feature-extraction comparison study (§4b) confirmed Phase 3's choice was sound. The in-house QVAR validation track (§7, Mam's directive 3) is done. MESA Track B (§7b) is done: CANet (ECG + effort belt, cross-modal attention) achieves AUROC 0.780 vs 0.610 ECG-only baseline on 220 MESA subjects, subject-independent. §7c confirmed the MESA-trained CANet runs end-to-end on SensorTile hardware but the effort stream does not transfer (94% healthy flag rate, AUROC 0.997 belt-vs-accelerometer separability). §7d (new) — a four-attempt sensor-domain adaptation track — rigorously ruled out representation-space alignment (linear and nonlinear, with and without adversarial training) as a solution to this gap with current data. Phase 5 has now had its first session (§7e): the chest accelerometer shows a clear effort drop during voluntary breath-holds (effort ratio below 1 in 7/7 events under a conservative window), but automatic boundary detection was attempted three ways and did not generalise across all 7 events, so effort ratios are reported as ranges. §7f then tried supervised transfer from the frozen MESA CANet on those labels: no demonstrated benefit over simple hand-crafted features (pilot, n = 7 events), but the end-to-end wrapper works on our own recording. Next: a second session to validate a boundary detector and to rule out temporal drift.

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

**Most of this document is the ECG branch** (the left half of that diagram). The effort branch (right half) is signal-processing, not a trained model — it has now been built and checked against a synthetic signal (§3–4, Phase 4); real IMU data is now confirmed present and usable in all 15 in-house recordings (§7), but the effort-branch pipeline hasn't been run against it yet.

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
   the "Phase 3 is done, no further tuning" decision (§8). The practical
   takeaway at the time was that QRS-area (method 6) looked worth keeping
   as a fallback EDR signal if R-peak amplitude's transfer risk (§6) turned
   out to be real on QVAR hardware. **§7 (in-house QVAR validation)
   overturns this**: the amplitude-scale transfer risk did turn out to be
   real (~108x mean gap), but QRS-area's own PhysioNet-vs-QVAR gap
   (~268x) is worse, not better — it is not a usable fallback as-is.

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

## 7. In-house QVAR validation track (Mam's directive 3)

**What & constraint.** 15 SensorTile QVAR-ECG recordings across ~9 subjects (healthy 19–20yo, no apnea labels) were downloaded and processed as **validation-only** data, per CLAUDE.md §4 — never used to train anything. With no apnea labels, no sensitivity/accuracy claim is possible from this data and none is made below; everything here is signal-quality, distribution-comparison, and negative-control checks.

**Data quality** (`results/inhouse_validation/quality_report.csv`). 14/15 files plausible (mean HR 44.7–90.0 bpm, implausible-RR ≤0.23%); one too-short aborted capture (`user_1612bhavi_ecg_000`, 320 samples, 1.3s); one noisier file (`user_0802-anagesh_ecg_000`, 1.09% implausible-RR, just over the 1.0% comparability margin). Empirical sample rate is **238.1–243.2 Hz** across all 15 files, a systematic ~1–1.3% offset from the nominal 240 Hz — too small to trip the loader's 2% warning, but real and consistent across every file, which vindicates CLAUDE.md rule 6's choice to keep RR features in time units rather than trust a fixed sample count. The accelerometer channel is present and non-flat in **all 15 files** — the effort branch is now real-data-testable, correcting the earlier "blocked on hardware" status for that specific check (Phase 5's breath-hold protocol itself still hasn't been run).

**Domain gap, PhysioNet vs QVAR** (`results/inhouse_validation/domain_gap/`, 13 plausible in-house files vs all 35 labelled PhysioNet records, per-beat distributions):

| Channel | PhysioNet mean | QVAR mean | Ratio |
|---|---|---|---|
| RR-interval (s) | 0.908 | 1.066 | 1.17x |
| R-peak amplitude | 1.295 | 140.5 | ~108x |
| QRS-area | 0.047 | 12.60 | ~268x |

RR-interval transfers cleanly — the 17% mean gap is healthy-young vs older-clinical physiology, not a hardware artifact. R-peak amplitude and QRS-area both fail to transfer, by two to three orders of magnitude: QVAR is an uncalibrated electrostatic sensor, not a calibrated ECG amplifier, and cleaning doesn't fix a scale mismatch that large. **This overturns §4b's assumption that QRS-area (method 6) is a viable fallback EDR channel for real hardware** — its PhysioNet-vs-QVAR gap is larger than the amplitude channel's, not smaller.

**Frozen-model plausibility test** (`results/inhouse_validation/plausibility/`, negative control: healthy subjects should get a LOW predicted apnea rate; 9 of 13 plausible files had ≥1 usable minute, 2,027 minutes total, 4 files too short for even one 60s window). The full 2-channel deployable model (RR + R-peak amplitude) predicts an implausible **52.8%** overall apnea-minute rate — a transfer artifact, not a result. Neutralizing the amplitude channel (RR-only variant, amplitude set to its training mean) drops this to **21.3%** overall, and to single digits for 6 of 7 longer recordings (kishor 71.4%→10.9%, aadhithiyaa 51.6%→6.1%, jayendran 59.0%→11.1%), confirming the amplitude channel as the dominant failure mode — consistent with the ~108x domain gap above. **The frozen model does not transfer to QVAR as-is; it needs a QVAR-specific recalibration of the amplitude/EDR channel — not a swap to QRS-area, not a retrain.**

**Hudson/Hari anomaly investigation** (`results/inhouse_validation/hudson_debug/`) — two recordings stayed implausible even in the RR-only variant; investigated rather than tuned around, same discipline as Phase 4's cessation-detection bug (§3–4). The initial hypothesis (R-peak under-detection from low HR) was tested and **rejected**: both hudson files' implausible-RR% (0.019%, 0.0%) sit at or *below* the known-good comparison file's (kishor, 0.225%), with unimodal RR histograms and every visible beat correctly marked in the waveform plots. Actual causes, established from the diagnostics, not assumed:
- **Both hudson recordings**: a genuine, stable ~45 bpm resting HR, roughly 2 SD below PhysioNet's training mean (68±11.8 bpm) — a **model-generalization limitation** on real, atypical-but-plausible physiology, not a bug or a bad capture. Notable because fit/low-resting-HR young adults are exactly who a wearable screener would encounter.
- **`user_2406Hari`**: a real ~10–15s electrode-contact/EMG electrical artifact (amplitude spikes to 400–500 vs. a normal ~50–100), visible in the biggest-RR-gap waveform plot. The IMU-based motion gate (`src/effort/envelope.py::compute_motion_mask`) flagged only 0.09% of this file — it cannot see a purely electrical artifact with no chest movement. **Exposes a tooling gap: there is no ECG-signal-based quality gate, only an IMU-based one.**

**Net result.** Directive 3 is complete. Three concrete findings carry forward into Phase 5/6, none of them blockers on closing this track: (a) RR-interval transfers, R-peak amplitude does not and needs QVAR-specific recalibration (QRS-area is not a fallback — it's worse); (b) an out-of-distribution-bradycardia blind spot in the PhysioNet-trained model; (c) a missing ECG-domain signal-quality gate that IMU-based motion gating cannot substitute for.

---

## 7b. Track B — MESA public data and CANet (ECG + respiratory effort)

**Scope and constraint.** MESA (NSRR, expert-scored PSG) is the *training* distribution for the fusion model; our own recordings stay validation-only (CLAUDE.md §4). All numbers below are **subject-independent** (grouped 5-fold CV over subjects, no subject in train and test of a fold), on **30 s sleep epochs**, target = **respiratory event (apnea or hypopnea) vs normal** (event prevalence 25.4%). Obstructive-vs-central is descriptive only (central is 0.05% of apnea epochs / zero in 123 of 220 subjects: not learnable).

**Audit** (`results/mesa/audit/`, code `src/ingest/mesa*.py`). 250 subjects downloaded (EDF + NSRR XML + Profusion XML, 0 failures); 220 usable under the strict funnel (4 signals present, sleep >= 4 h, both belts alive and showing breathing in >= 50% of epochs, >= 1 scored event). Native rates: EKG 256 Hz, Thor/Abdo belts 32 Hz, SpO2 1 Hz. Class balance on usable subjects (sleep epochs): 74.9% normal / 19.6% hypopnea-only / 5.5% apnea (89.5% of apnea epochs obstructive, 10.5% central, mixed ~0). Per-subject medians [IQR]: obstructive 8 [2-26], central 0 [0-2], hypopnea 107 [57-153]. AHI is rule-dependent (median 11 under NSRR 4% vs 20 under 3%/arousal vs 21 scoring every event). Committed outputs are anonymised (`subjects_anon.csv`; mesaids never in git).

**ECG-only baseline** (`src/mesa/ecg_baseline.py`, `results/mesa/ecg_baseline/`). RR + R-amplitude (2 Hz, target epoch +/-2 neighbours = 150 s, per-subject label-free normalisation) -> 1D-CNN, weighted BCE, early stopping and Youden threshold from inner-validation *training* subjects only. Pooled over 156,826 test epochs: **AUROC 0.610, AUPRC 0.335** (chance 0.254), sens 0.62 / spec 0.53, F1 0.42. Subject-level AHI correlation r ~ 0.30 (a0h3/a0h4).

**CANet** (`src/mesa/canet.py`, `results/mesa/canet/`). Two streams — cardiac (identical to the baseline input) and respiratory effort (Thor + Abdo @ 32 Hz) — multi-scale 1D-CNN encoders (kernels 3/7/15), bidirectional cross-modal attention, GAP+GMP -> MLP; 169k parameters; same folds, same epochs, same threshold discipline (asserted in code). **No SpO2, no arousal in the headline model** (our wearable has neither).

| Model | AUROC | AUPRC | Sens | Spec | AHI r (a0h3 / a0h4) |
|---|---|---|---|---|---|
| ECG-only baseline | 0.610 | 0.335 | 0.62 | 0.53 | 0.31 / 0.30 |
| **CANet (ECG + effort)** | **0.780** | **0.490** | 0.82 | 0.61 | 0.74 / 0.70 |
| ablation: effort only | 0.747 | 0.422 | 0.84 | 0.57 | 0.71 / 0.68 |
| ablation: ECG + effort, concat (no attention) | 0.744 | 0.424 | 0.82 | 0.58 | 0.72 / 0.69 |
| CEILING: CANet + SpO2 (not on our wearable) | 0.813 | 0.541 | 0.85 | 0.64 | 0.82 / 0.77 |

Headline vs baseline: dAUROC +0.171 (95% CI +0.151..+0.191), dAUPRC +0.155 (+0.132..+0.184), paired subject bootstrap; every fold improves (CANet AUROC 0.76-0.80 vs 0.56-0.64). Per-class sensitivity at each model's own threshold: obstructive apnea 0.83 -> 0.97, central apnea 0.84 -> 0.98, hypopnea 0.56 -> 0.78, specificity (normal) 0.54 -> 0.61. Attention vs alternatives: +0.036 AUROC over concat fusion, +0.033 over effort-only (CIs exclude 0).

**Honest caveats.**
- Most of the ECG-only -> CANet gain comes from the effort signal (effort-only already 0.747); ECG fused by attention adds a smaller, real gain.
- The concat ablation is **not parameter-matched** (102k vs 169k), so the attention gain is not cleanly attributable to attention.
- **Single seed per fold**, no hyperparameter search; fold-to-fold SD ~0.015 AUROC.
- The "AHI" correlation is an epoch-rate proxy (flagged 30 s epochs per sleep hour), not an event-scored AHI, and there is **no SpO2**, so desaturation-based definitions cannot be matched. Specificity is only 0.61; sensitivity numbers use lenient Youden thresholds, so compare AUROC/AUPRC across models, not sensitivity alone.
- Hypopnea dominates the positives and is the weakest class; the usable cohort is skewed toward disease (median all-event AHI 21).
- MESA effort is a **respiratory inductance belt**; see §7c for what that means for our accelerometer.

## 7c. In-house CANet transfer (MESA-trained CANet on our QVAR + IMU hardware)

**Headline: CANet runs end to end on our hardware, BUT the belt-trained effort stream does NOT transfer to accelerometer input.** Healthy subjects get flagged on ~94% of epochs — a **transfer failure, not an apnea result**.

**Constraint.** The in-house cohort is healthy 19-20 y/o, unlabelled, **validation only — never trained on, never committed**. Nothing in this section is an apnea-detection accuracy and none is claimed. The study is characterization + plausibility + deployability (`results/inhouse_canet/`, code `src/inhouse/`, `src/mesa/train_final.py`). Nights are anonymised S01..; sleep state is unknown (no staging), rates are per recorded hour; 10 of 15 recordings are usable (5 are 1-32 s aborted captures).

**Deployability: yes.** QVAR -> R-peaks -> cardiac stream (MESA-identical grid; per-night normalisation, so the earlier ~108x amplitude gap is neutralised by construction) and accelerometer -> breathing trace (`src/effort/envelope.py`) -> 32 Hz, MESA belt preprocessing -> MESA-trained CANet -> per-epoch probability -> per-night rate, for all 10 usable nights. (Weights are refit on MESA only for this purpose: the CV runs did not keep weights; they are gitignored.)

**Belt vs accelerometer gap.** A classifier separates MESA belt epochs from our accelerometer epochs with **AUROC 0.99** (handcrafted features) / **0.997-0.998** (effort-encoder embedding); control (MESA half vs half) 0.51 / 0.47. Breathing *is* in the accelerometer (spectral peak near 0.2-0.25 Hz) but the signal is broadband and irregular: spectral entropy 0.64-0.66 vs 0.46 for belts (night medians at the 87-98th MESA percentile), zero-crossings 40-42 vs 30 per min, breathing-rate spread IQR 10 vs 4 bpm. Accel magnitude and principal axis look about equally far. Caveat: posture/activity of the recordings is unverified, so part of the gap may be behaviour rather than sensor.

**Healthy negative control** (flag fraction at the MESA operating points; reference = MESA out-of-fold low-AHI subjects, a0h4 < 5, n = 67):

| Model | In-house | MESA low-AHI median [p90] |
|---|---|---|
| ECG-only | 0.58 | 0.43 [0.67] |
| effort-only (accel) | 0.80-0.84 | 0.35 [0.55] |
| **headline CANet** | **0.92-0.94** | 0.28 [0.48] |
| effort-only, white-noise effort | 0.33 | 0.35 [0.55] |

The ablation isolates the **effort branch**: pure white-noise effort gives 33% (in MESA range) while real accelerometer effort gives 80-84% — the model reads our structured-but-wrong-morphology trace as disrupted breathing. The cardiac side is **mildly elevated** (58%, 4/10 nights above MESA p90); possible respiratory sinus arrhythmia in young hearts, **unverified**. Fusion compounds both. (Note the MESA thresholds flag ~28% even of MESA low-AHI subjects; that, not zero, is the right comparator.)

**Conclusion.** The effort branch needs **adaptation or an accelerometer-native effort model** before deployment; MESA belts remain the only validated effort input. Whether our accelerometer can even detect cessation is now the critical open question — see `docs/breath_hold_protocol.md` (Phase 5 recording session).

---


---

## 7d. Sensor-domain adaptation track — four attempts, consistent negative result

**Motivation.** §7c established that the MESA-belt-trained effort encoder is trivially separable from accelerometer input (AUROC 0.997 in embedding space) and produces ~94% false-positive flagging on healthy in-house nights. This track systematically attempts to close that gap without labels, using only the 10 usable in-house nights and the MESA belt data.

**All experiments are in `src/adaptation/`; outputs in `results/adaptation/`.**

---

### Step 1 — Embedding gap characterisation (`embed_gap.py`)

**Outputs:** `results/adaptation/embed_gap/` (embeddings.npz, embedding_space.png, pca_dim_gap.png)

Extracted the resp encoder's 128-dim GAP+GMP embeddings (60 MESA subjects × 30 epochs = 1,800 belt embeddings; 10 in-house nights × ~180 epochs = 1,811 accelerometer embeddings). PCA + t-SNE on the joint space.

**Key findings:**
- **PC1 carries 82.4% of variance.** Belt embeddings span the full PC1 range (−10 to +35); accelerometer embeddings are **collapsed into a tight cluster at the left end** (−8 to −2) — the encoder uses only a tiny corner of its representational space for accelerometer input.
- **Mean shift is small** (0.162 SD across 50 PCA dims), but separability is 0.997 — the gap is in manifold structure (variance collapse in PC1, distributional shape differences in minor PCs), not in mean offset. A linear shift/scale adaptor will not fix this.
- **Mean L2 distance MESA→in-house: 10.086 vs. within-MESA: 10.601 (ratio 0.95x).** Cross-domain distance is *smaller* than within-domain, confirming the gap is structural rather than a global offset.
- t-SNE shows partial overlap (not completely disjoint clusters), confirming the spaces are not entirely incompatible — adaptation is potentially viable, but requires more than a linear map.

---

### Step 2 — CORAL (Correlation Alignment, linear, two attempts) (`coral_adaptor.py`)

**Outputs:** `results/adaptation/coral/` (including FINDINGS.md)

CORAL (Sun et al. 2016) aligns second-order statistics (covariance) between domains — closed-form, no labels needed.

**Attempt 2a — CORAL at pooled (GAP+GMP, 128-dim) embedding level.**
- Covariance alignment succeeded: domain AUROC 0.994 → 0.218.
- But flag rate went 93% → 100% — got worse.
- **Diagnosis:** bypassed cross-attention entirely; the head received a representation it was never trained on. Alignment worked, model broke.

**Attempt 2b — CORAL at token level (64-dim, pre-attention), preserving full forward pass.**
- Domain AUROC: 0.954 → 0.934 (barely moved; regularisation sweep 0.01–10.0, best reg=1.0).
- Flag rate: ~93% → 96.4% median (no real improvement, within noise).
- **Diagnosis:** per-token CORAL assumes Gaussian blobs; belt vs. accelerometer tokens differ in *nonlinear temporal structure* (different relationship between signal shape and token content across 75 positions), not just covariance. Consistent with the spectral characterisation from §7c (spectral entropy, zero-crossing rate, breathing-rate IQR all structurally different).

**Conclusion:** linear alignment is insufficient at either level. See `results/adaptation/coral/FINDINGS.md`.

---

### Step 3 — Adversarial adaptor v1 (large capacity) (`train_adaptor.py`)

**Outputs:** `results/adaptation/adaptor/` (including FINDINGS.md)

Residual MLP per-token (hidden=128), trained adversarially against a belt-vs-accel discriminator, with a breathing-consistency loss (autocorrelation at the measured per-epoch respiratory lag) and an L2 residual anchor. Train nights: S03/S06/S07/S08/S11/S12/S13/S15 (2,411 epochs). Val nights held out: S04, S10.

**Result:** Training unstable — residual loss oscillated 1.6–13.2 without converging; discriminator confidence on belt tokens trended *upward* (toward 0.78) rather than toward the fooled equilibrium of 0.5. Held-out domain AUROC (at end of training): **1.0000** — worse than the pre-adaptation baseline (~0.93–0.95). Textbook adversarial overfitting: adaptor found a training-set-specific equilibrium that did not generalize to unseen nights.

> **Erratum (2026-10-09):** the held-out AUROC of 1.0000 reported for attempt 1 came from the pre-fix *resubstitution* measurement (classifier fit and scored on the same data; see Step 4, where the same bug returned 1.0 for the identity adaptor) and must not be read as a valid number; the finding for attempt 1 rests on the unstable training dynamics and the absence of a credible flag-rate benefit, not on that AUROC.

**Training curve confirms:** disc(belt) and disc(adapted) diverge throughout; no sign of the crossing-at-0.5 equilibrium that indicates genuine alignment. See `results/adaptation/adaptor/training_curves.png`.

---

### Step 4 — Adversarial adaptor v2 (constrained capacity, corrected measurement) (`train_adaptor_v2.py`)

**Outputs:** `results/adaptation/adaptor_v2/` (including FINDINGS.md)

Changes from v1: hidden dim 128→32, residual penalty 0.05→0.3 (6x stronger anchor), discriminator LR 3×10⁻⁴→1×10⁻⁴ (slower than adaptor to prevent discriminator winning the arms race), early-stopping on held-out domain AUROC checked every 40 steps.

**Measurement bug caught and fixed mid-run:** the held-out AUROC function originally fit a logistic regression and scored it on the *same* data (resubstitution), giving spurious AUROC=1.0 even for the identity (no-op) adaptor as a sanity check. Caught because the identity sanity check returned 1.0 — that's impossible for a genuine generalization measurement. Fixed to use a proper random train/test split inside the classifier. All numbers below are post-fix.

**Result:** Pre-adaptation sanity check (identity adaptor): **0.9984** (validates the measurement — matches the CORAL experiment's ~0.95 baseline range). Best held-out AUROC during training: **0.9981** (delta: −0.0003, within noise). Training stable but inert — alpha grew only 0.050→0.057 over 240 steps before early-stop triggered. Flag rate on held-out nights: S04 85.8%, S10 92.3% (essentially unchanged from pre-adaptation ~93–97%).

**Verdict: NO IMPROVEMENT.** Unlike v1 (actively overfit), v2 was stable but had too little capacity to move the representation at all with this data volume.

---

### Summary across all adaptation attempts

| Attempt | Mechanism | Train-domain effect | Held-out generalization | Flag rate change |
|---|---|---|---|---|
| CORAL pooled (128-dim) | linear covariance | AUROC 0.994→0.218 (strong) | Broke model (cross-attn bypassed) | 93%→100% (worse) |
| CORAL token (64-dim) | linear covariance | AUROC 0.954→0.934 (weak) | Consistent — no real improvement | ~93%→96.4% (noise) |
| Adversarial v1 (hidden=128) | nonlinear, large | Discriminator winning | AUROC 1.000 (overfit) — *invalid, see erratum under Step 3* | ~93%→96–100% |
| Adversarial v2 (hidden=32) | nonlinear, constrained | Stable but inert | AUROC 0.9984→0.9981 (noise) | ~93%→84–92% |

**Overall conclusion.** Four distinct mechanisms — linear and nonlinear, loosely and tightly constrained — produced consistent null or negative results. This is not a single failed attempt; it is evidence that **representation-space manipulation of the effort encoder's tokens cannot close the belt-to-accelerometer gap with 2,411 training epochs across 8 nights**. The gap is real (confirmed in §7c, characterized structurally in §7d Step 1) but requires either substantially more in-house accelerometer data, a labelled supervisory signal (breath-hold events from Phase 5), or direct retraining/fine-tuning of the effort encoder on accelerometer data. These results make Phase 5 (breath-hold protocol) not merely useful but **necessary** before any further adaptation work is meaningful.

**Note on cross-experiment AUROC comparability:** the CORAL token-space measurement used `GroupKFold` across all nights; the adaptor v2 measurement used a 50/50 split restricted to 2 held-out nights. These are not directly comparable; the difference in baseline values (0.9542 vs 0.9984) reflects different night subsets and protocols, not a real change in separability.

---

## 7e. Phase 5 — first breath-hold session (chest SensorTile, 7 voluntary holds)

**Constraint, as always.** One healthy volunteer, one session, one device (chest box: QVAR-ECG + accelerometer; the EOG/EMG boxes were placed elsewhere and are not analysed). Validation only, never trained on, raw recording not committed. This is a plausibility/detectability result, **not** an apnea-accuracy number. Code `src/breath_hold/analyze_session.py`, outputs and per-event plots in `results/breath_hold/` (write-up: `results/breath_hold/FINDINGS.md`).

**Session.** `data/recordings/breath_hold/user_MukuBreathHold_ECG_07_10_2026.csv` (new comma-delimited firmware format; loader fixed in 68d16f8). Recording 1795.6 s, empirical fs 243.1 Hz (+1.29% vs nominal, same offset as §7). Seven ~30 s holds (stopwatch laps 2,4,…,14); stopwatch total 1800.46 s vs recording 1795.6 s, i.e. a **4.8 s sync lag**, consistent with the reported 5–10 s manual start lag. Stopwatch boundaries are therefore good to a few seconds only, which is exactly why boundary placement matters for the ratio below.

**Effort ratio** = RMS of the 0.08–0.6 Hz bandpassed accelerometer magnitude inside the hold ÷ RMS over the 15 s before it (<1 means effort dropped). Reported under two honest windows, because automatic boundary detection did not work reliably (next subsection):

| Event | Inner window (nominal inset by 4.8 s sync lag) | Wide window (nominal stopwatch boundaries) | Detector (visually graded) |
|---|---|---|---|
| 1 | 0.25 | 0.62 | fell back to nominal — FAIL |
| 2 | 0.27 | 0.31 | 0.07 — OK |
| 3 | 0.64 | 0.49 | 0.05 — FAIL (truncated by a mid-hold burst) |
| 4 | 0.14 | 0.52 | 0.09 — OK |
| 5 | 0.46 | 0.78 | 0.16 — FAIL (no clean quiet stretch) |
| 6 | 0.10 | 0.33 | 0.04 — OK |
| 7 | 0.60 | 1.49 | 0.47 — FAIL (start ~13 s late) |
| mean / median | 0.35 / 0.27 | 0.65 / 0.52 | — |

**Read these as a range per event (inner → wide), not a point value.** Inner window: 7/7 events below 0.8, 5/7 below 0.5. Wide window: 6/7 below 0.8, 3/7 below 0.5. Direction is consistent in every event under the inner window; magnitude is not pinned down. All 7 hold windows in `session_overview.png` show a visibly flat effort stretch.

Why the two windows differ, event-specific and not forced: the 4.8 s lag means quiet periods sit earlier in recording time than the stopwatch says, so the wide window of Event 7 (quiet ends ≈1738 s, nominal end 1743.8 s) swallows the recovery-gasp burst — hence 1.49, which is boundary bleed, not "effort increased". Conversely in Event 1 the true quiet starts ≈5 s *after* nominal start (breath/settling), so the wide window includes breathing. The pre-hold baseline is anchored to the nominal start, so events whose quiet starts before nominal (Event 4, ≈6 s) have a slightly quiet-contaminated baseline, which biases ratios *upward* (conservative). Event 3 has a ~3.5 s mid-hold burst (swallow/twitch); Event 5 is not cleanly quiet at all.

**Boundary detection: attempted three ways, did not generalise (an honest result).**
1. *v2 (previous run) — the confirmed bug.* `refine_from_effort` ran on the **signed bandpassed trace**, taking `|x|` below a 30th-percentile threshold for **5 consecutive samples ≈ 20 ms** at 243 Hz. `|bandpassed|` crosses zero twice per breath, so every zero-crossing counted as "quiet" and the first match was always the left edge of the ±12 s search window: all 7 start offsets landed at −11.0…−11.9 s (the search bound, not 7 independent detections). Event 1's ratio of 3.12 came from a "hold" window that was mostly normal breathing.
2. *v3a — Hilbert envelope, threshold = 30% of pre-hold median envelope, quiet run ≥ 2 measured breath periods (period from each event's own baseline spectrum, 2.1–3.9 s), segments chained across ≤1 period of activity, ≥80% quiet, refuses to return a boundary on the search-window edge (falls back to nominal, flagged).* Removed the edge-pinning. Visually correct on Events 2, 4, 6 only.
3. *v3b — floor-relative threshold (noise floor from the event's own window + 30% of the way to baseline).* Changed nothing materially (noise floor ≈ 0), same 3/7. Per the agreed rule this was the last threshold adjustment; tuning further on one subject/session would overfit the detector to this session.

Visual verdict per event (all 7 plots read): OK = 2, 4, 6; FAIL = 1 (fell back; real quiet ≈997–1015 s not found), 3 (stops at ≈1240.6 s because of the mid-hold burst; quiet resumes to ≈1257 s), 5 (6 s fragment), 7 (start 1724.6 s vs visible quiet from ≈1710.5 s; end OK). The CSV carries `refinement_fell_back_to_nominal`, `fallback_reason` and a `detector_visual_verdict` column (a **manual** annotation, flagged as such in the code), so rows are self-describing. The detector's per-event numbers (0.04–0.09 on the three visually-good events) are shown for transparency; they are **not** a headline.

**HR direction: still unconfirmed.** Hold-minus-pre mean HR, computed on each version's boundaries: v1 +0.84 bpm (5/7 up), v2 −3.08 (1/7 up), v3 −1.35 bpm, SD 4.8 (3/7 up; per event +5.5, +1.7, +2.8, −3.8, −6.2, −7.1, −2.3). The sign changes with the boundaries, so no direction is established and no bradycardia/tachycardia claim is made. The expected physiological caveat stands (short voluntary awake holds often show anticipatory sympathetic rise), but it is a hypothesis, not a finding. Separately, Event 1 contains a ~153 bpm two-beat R-peak artifact near 1000 s that the single-beat despiker does not catch (two consecutive samples), which inflates that event's hold-mean HR; not fixed here.

**Open problem — status.** The *bug* is resolved (root cause identified and removed); a **reliable automatic boundary detector is not**. Needs a second session (ideally with a physical sync marker — 3 sharp taps per protocol — or a button/event marker to cut the 4.8 s lag uncertainty) to validate a detector against, plus the SHORT 5–7 s hold false-positive test from `docs/breath_hold_protocol.md`, which this session did not include.

**Session 2 preparation (2026-10-09).** A v2 protocol (`docs/breath_hold_protocol.md`: sync double-taps at the start and end, 9 holds of 5–60 s, rest ≥ max(90 s, 3× hold), annotation sheet, mandatory pre-session tap check with a STOPWATCH-ONLY fallback, and a pre-specified analysis plan) and the sync tooling (`src/breath_hold/sync.py`; validation gate `src/breath_hold/sync_validate.py`, results in `results/breath_hold/sync_validation/`) have been prepared. **No session-2 data exists yet**; nothing in this section changes until it is recorded.

**What this establishes / does not.** Establishes: on real hardware, chest accelerometer effort visibly collapses during voluntary holds (inner-window ratio < 0.8 in 7/7), the first labelled evidence that the raw accelerometer carries cessation information — a counterpoint to §7c, which showed the *belt-trained learned representation* doesn't transfer, not that the signal is uninformative. Does not establish: boundary precision, detector thresholds, HR behaviour, generalisation beyond one subject, or any false-positive rate.

---

## 7f. Supervised transfer: frozen MESA CANet features vs hand-crafted effort features, trained on our own breath-hold labels

**Headline (read this first).** The frozen MESA-trained CANet features do **not** beat a simple hand-crafted baseline at this data scale. Under identical leave-one-event-out CV, a linear head on the frozen 256-d features and a linear head on 8 hand-crafted effort features tie on pooled AUROC (**0.83 vs 0.83**; 95% event-bootstrap CIs 0.58–1.00 and 0.71–0.97). The hand-crafted head is the more consistent detector: per-fold AUROC min **0.78 vs 0.16**, and **0.5% vs 24%** false positives on the held-out baseline period. A shuffled-training-label control collapses both heads to chance (frozen 0.48 ± 0.10, hand-crafted 0.45 ± 0.09), so the pipeline is not leaking labels. **Caveat that applies to this headline and every comparison below:** the two feature sets are *not* apples-to-apples. The frozen features see a 150 s window (cardiac + effort, cross-attention) whereas the hand-crafted features see only the 30 s effort centre epoch. Scope of the claim: the shuffled-label control rules out label leakage; it does not make a 7-event result statistically firm (the CIs above span 0.58 to 1.00), so the supportable statement is "no demonstrated benefit of the frozen features, and worse consistency", not a proven ranking. One subject, one session, n = 7 events: **a pilot, not a validated accuracy.** Voluntary awake holds are also not apnea; nothing here is an apnea accuracy.

**Deliberate, scoped exception to CLAUDE.md §4 / methodology rule 5 ("never train on our own recordings").** Recorded 2026-10-09, *before any result existed*, so it is a decision and not a result-driven rationalisation. The advisor directed that the hold/rest labels from §7e be used as supervision. The exception is limited to **a linear (L2-regularised logistic regression) head fitted on frozen, MESA-trained CANet features and on hand-crafted effort features, using the 7 labelled holds of the single 2026-10-07 session.** It does not change how the ECG branch (PhysioNet), the MESA CANet (§7b) or any frozen weights are trained: no encoder, attention or MESA head weight is updated, and in-house data is still never used to train or tune a deep network (the §7d lesson: unsupervised deep adaptors overfit 8 nights; this is supervised and linear). The healthy in-house nights remain validation-only. With one subject there is no subject-independent claim; numbers are within-subject, leave-one-event-out.

### Design (fixed before any model was run)
- **Window** = one 150 s CANet input (target 30 s epoch ± 2 context epochs, the frozen model's native shape; 75 fixed tokens), labelled by its 30 s centre epoch. Stride 5 s. Per-recording normalisation exactly as §7c; accelerometer breathing trace fed to both belt channels.
- **Positive**: ≥ 15 s of the 30 s epoch inside the §7e *inner* hold (nominal inset by the 4.8 s sync lag; holds are only 19–21 s). **Rest** (primary negatives): epoch ≥ 15 s clear of every inner hold and in a between-hold gap. **Baseline** (secondary, test-only, never pooled): pre-Hold-1 windows whose whole 150 s input ends before Hold 1 starts (182 windows). Transition zones dropped. 28 positive / 48 rest windows, tagged with their event; **independent units = 7 events**.
- **CV**: leave-one-EVENT-out. Test = that event's positives + the rest windows nearest to it (each window tested once). Train = all other pos/rest windows minus any whose full 150 s input overlaps the held-out hold ± 15 s, minus any whose target epoch overlaps a test epoch. Folds: 4 test positives, 4–8 test rest, 24 train positives (all 6 other events), 32–40 train rest; purge removes 4–8 windows.
- **Heads**: StandardScaler + LogisticRegression, strong L2, C = 0.01 fixed a priori (not tuned), threshold 0.5. Frozen = mean+max pooled tokens after cross-attention, 2 streams × 128 = 256-d. Hand-crafted = the 8 `epoch_features` of the 30 s effort epoch (`src/inhouse/run_canet.py`). Extra diagnostics (RMS only; frozen effort-half / cardiac-half) are exploratory.

### Results (out-of-fold, C = 0.01)
| Features | Pooled AUROC (95% CI) | Sens / spec @0.5 | Per-fold AUROC mean / min | Baseline-period FPR |
|---|---|---|---|---|
| Frozen CANet, 256-d | 0.832 (0.58–1.00) | 0.71 / 1.00 | 0.84 / 0.16 | 0.24 |
| Hand-crafted, 8 features | 0.826 (0.71–0.97) | 0.43 / 1.00 | 0.94 / 0.78 | 0.005 (0–0.022 per fold) |
| RMS only | 0.599 | 0.00 / 1.00 | 0.71 / 0.00 | 0.00 |
| *Frozen effort half (diagnostic)* | 0.690 | 0.64 / 0.85 | 0.71 / 0.00 | 0.33 |
| *Frozen cardiac half (diagnostic)* | 0.865 | 0.86 / 0.92 | 0.90 / 0.31 | 0.09 |

Per-fold AUROC (held-out event): frozen 1.00 / 1.00 / 1.00 / 1.00 / **0.16** / 1.00 / **0.69\*** (events 1–7); hand-crafted 0.81 / 1.00 / 0.97 / 1.00 / 0.78 / 1.00 / 1.00\*. (\*Event 7 has up to 27% zero-padded input.) Reference: the untouched MESA head separates hold from rest at AUROC 0.74 with no training.

**C sensitivity (reported, not used for selection).** Pooled AUROC at C = 0.001 / 0.01 / 0.1 / 1: frozen 0.875 / 0.832 / 0.818 / 0.818; hand-crafted 0.709 / 0.826 / 0.900 / 0.920. The tie holds only at the pre-specified C; with weaker regularisation hand-crafted is clearly ahead. The diagnostic halves and the C sweep are extra looks at 7 events; only the frozen-256 vs hand-crafted-8 comparison at C = 0.01 was pre-specified.

**Event 7 (up to 27% / 40 s zero-padded input), pooled and separately.** Frozen: events 1–6 only AUROC 0.840 (sens/spec 0.83/1.00); Event 7 alone AUROC 0.688 (sens/spec 0.00/1.00). Hand-crafted: events 1–6 0.812 (0.42/1.00); Event 7 alone 1.000 (0.50/1.00). Pooled-with-all-7 numbers are in the table above. *Is it the padding?* Not established. A control that zero-padded the last 40 s of the inputs of Events 1–6 did not hurt them (AUROC 0.840 → 0.892), and Event 5, with no padding, fails equally (0.16), which argues against padding alone. But Event 7's own positive windows score lowest as padding rises (p = 0.40, 0.30, 0.26, 0.15 at pad 17–27%) while moving toward the recovery gasp over the same four windows, so the two are confounded; I cannot separate them. The hand-crafted head only reads the 30 s centre epoch, so padding cannot affect it.

**Leakage / dominance checks.** No pooled AUROC > 0.95. Five frozen folds are exactly 1.00, on 4 positives vs 4–8 rest each, which is easy to reach at that size. Dropping Event 5 lifts the pooled frozen AUROC to **0.96** (above the project's 0.95 caution level): the frozen result is "5 perfect events, 2 failing events", not a uniform detector. Dropping any one event moves hand-crafted between 0.79 and 0.90. Shuffled-training-label null: frozen 0.484 ± 0.099, hand-crafted 0.446 ± 0.086 over 20 repeats. Window-level purging and event-level folds are in place.

### Why RMS-only fails: label impurity (same class of bug as Phase 4 and §7e)
A 30 s centre epoch cannot isolate a 19–21 s hold: every positive epoch contains 9–15 s of surrounding breathing, including the inhale before and the recovery gasp after. The mean RMS of the positive epochs is therefore *not* lower than rest in Events 1, 2, 3 and 5 (Event 1: 1.87 vs 1.13, higher) and only clearly lower in Events 4, 6, 7 (e.g. Event 7: 0.77 vs 2.40). This is the same family of problem as the Phase 4 over-smoothed envelope and the §7e boundary detector: a window or boundary that does not match the physical event's extent. It also handicaps the hand-crafted baseline's absolute level (the 8-feature head's sensitivity at 0.5 is only 0.43 out-of-fold). The labels are impure by design (the frozen model's native input forces ≥ 30 s targets), not a measurement error to be tuned away.

### Deployment (`src/deploy/predict_session.py`)
`python3 -m src.deploy.predict_session <recording.csv|txt> [--stride 5] [--out DIR] [--events event_summary.csv]`. Raw recording → `load_sensortile` → R-peaks + accelerometer breathing trace → 150 s windows → frozen CANet → linear head **A**, and the 30 s hand-crafted features → linear head **B**; prints a per-window report and saves a probability-over-time plot and CSV. Both heads (`src/transfer/train_heads.py`, `results/transfer/deployed_heads.joblib`) are fitted on **all 7 holds**, so the wrapper labels predictions on the training session **IN-SAMPLE, not a performance estimate** (automatically, by file name) in the console and on the plot.

*In-sample demo on the training session* (`results/transfer/demo_breath_hold/`): head A's probability rises to ≥ 0.8 at all 7 holds, which looks convincing — but A also flags 29.8% of all windows in 25 segments, mostly in the awake baseline, consistent with its 24% held-out baseline FPR. Head B flags only 4 segments (Holds 2, 4, 6, 7), i.e. 3.2% of windows, and misses Holds 1, 3, 5 (in-sample sensitivity on training windows 0.36 vs 0.93 for A, specificity 1.0 for both). Because this is in-sample, the visual advantage of A is **not** evidence it is the better detector; the out-of-fold table above is. B's low sensitivity at 0.5 is the price of heavy L2 on 8 features: it is conservative, not sensitive.

*Held-out healthy nights (the only truly out-of-sample test; `results/transfer/healthy_nights_check.csv`).* 10 nights, 0.8–7.5 h, no labelled holds, stride 30 s, every window treated as "not a hold": pooled flag rate **3.1% (frozen) and 4.6% (hand-crafted)**; per-night 0.0–5.9% and 0.0–9.7%. Caveats: this is sleep, not the awake-supine breath-hold state (the §7c domain shift), so it does not measure the baseline-period behaviour above; nights are unlabelled so any true pause counts as an error; specificity-only, n = 10; the 5 s vs 30 s stride differ. It shows neither head fires constantly on healthy sleep; it does not validate hold detection.

### Open limitations (explicit)
1. **Temporal drift is not ruled out (open).** The 24% (frozen) vs 0.5% (hand-crafted) baseline-period false-positive rate comes from windows drawn from the first ~900 s of one session, while the training windows come from the final ~800 s. Slow drift in posture, electrode/skin contact, motion or heart-rate state across the session could drive part of that gap; the shuffled-label control does not test for it, and the head clearly does not separate purely by time (it is quiet on healthy nights and on much of the baseline), but I have not excluded drift. A second session with holds spread across the recording and a rest-only control stretch would address it.
2. **Not apples-to-apples** (150 s cardiac + effort vs 30 s effort only); a hand-crafted baseline on the same 150 s window, or a frozen model on a 30 s effort-only window, was not run.
3. **Label impurity** from the 30 s target epoch vs 19–21 s holds (above); the detector boundary problem from §7e still stands.
4. **n = 7 events, one subject, one session**; per-fold test sets of 4 positives; pooled bootstrap CIs are wide; Events 5 and 7 behave as outliers for the frozen features and Event 7's cause (padding vs event-specific) is unresolved.
5. **Multiple looks**: five feature sets × four C values on 7 events; only one comparison was pre-specified. The frozen cardiac half scoring 0.87 is unexplained (the tokens have passed through cross-attention with effort) and is not claimed as a cardiac effect.
6. **Voluntary awake holds ≠ apnea**; these are hold/rest classifiers, not apnea accuracy. The 5–7 s SHORT-hold false-positive test was not recorded.

**Reproduce.** `python3 -m src.transfer.windows` (counts) · `python3 -m src.transfer.run_transfer` (CV, `results/transfer/results.json`, `oof_*.csv`) · `python3 -m src.transfer.train_heads` · `python3 -m src.transfer.healthy_check` · `python3 -m src.deploy.predict_session ...`. Use the system `python3` (torch + GPU); the project `venv` has no torch.

---

## 7g. Why not an LSTM/RNN for the hold classifier? (technical note answering an advisor question; no experiment run; wording open to the advisor's input)

**Question.** Why did §7f use frozen-feature *linear* heads and not a temporal model (LSTM/RNN) for hold-vs-rest? **Short answer.** (i) The model we already use is not a non-temporal model; (ii) the documented obstacle is distribution shift between belt and accelerometer data, not a lack of temporal modelling; (iii) with 7 independent labelled events no recurrent model could be trained *and evaluated* in a way that would tell us anything, whichever way it came out. We did not run one. Reasoning below, with the numbers it rests on.

### 1. What the evidence from §7d does and does not say — including a correction
- **Data scale there:** 2,411 unlabelled 30 s epochs from 8 nights (train S03/S06/S07/S08/S11/S12/S13/S15; held-out S04, S10). §7f has **7 independent events** (76 training windows, 28 of them positive, strongly correlated within an event): 2,411 / 7 ≈ 344x more training units than events, or ≈ 32x more than windows.
- **What failed there:** attempt 1 (residual MLP, hidden 128, ≈ 16.6k adaptor parameters plus an ≈ 8.3k-parameter discriminator) trained unstably (residual loss oscillating 1.6–13.2, discriminator confidence 0.46–0.78, never reaching the 0.5 equilibrium) and gave only a small, non-credited drop in flag rate (median 87.8% on held-out nights vs ~93–97% before). Attempt 2 (hidden 32, ≈ 4.2k parameters, 6x stronger anchor) was stable but inert (held-out AUROC 0.9984 → 0.9981, flag rate 86–92%). Capacity was therefore tried at two sizes with two different failure signatures (unstable; inert) on 2,411 epochs.
- **ERRATUM, found while writing this note (since recorded in §7d Step 3 and as a pointer note in the attempt-1 FINDINGS.md, commit `33c586c`).** §7d/`results/adaptation/adaptor/FINDINGS.md` report attempt 1's "held-out domain AUROC 1.0000, worse than the ~0.93–0.95 baseline" as evidence of overfitting. In `src/adaptation/train_adaptor.py` (lines ~310–322) that AUROC is computed by fitting the logistic regression and scoring it **on the same data** — the exact resubstitution error that §7d Step 4 later identified and fixed for attempt 2, where it returned 1.0 even for the identity (no-op) adaptor. The 1.0 for attempt 1 is therefore not interpretable (a no-op would have scored 1.0 too) and **must not be cited as overfitting evidence**; the corrected pre-adaptation baseline under the fixed measurement is 0.9984. What still stands for attempt 1 is the unstable training dynamics and the absence of a credible flag-rate improvement. Note that `results/adaptation/adaptor_v2/FINDINGS.md` (its "Change from attempt 1" paragraph) still repeats the attempt-1 "1.0 vs ~0.93-0.95" comparison; it is left as the historical record and is covered by the same erratum.
- **Honest limits of the analogy.** (a) The adaptors were *unsupervised, adversarial*; an LSTM would be *supervised*, so its failure would look different (memorising events) rather than adversarial instability. (b) A tiny LSTM can have **fewer** parameters than the adaptors (1 layer, 2 inputs, hidden 8 ≈ 350 parameters; hidden 16 with 3 inputs ≈ 1.3k), so "an LSTM has more capacity than the adaptor" is not automatically true. The argument is not parameter count; it is the number of independent events available to constrain *any* trainable temporal model.

### 2. CANet is already a temporal model; the open problem is transfer, not temporality
CANet (`src/mesa/canet.py`, 169k parameters, trained on 156,826 MESA epochs from 220 subjects, §7b) is not a static model. Each stream is a multi-scale 1D CNN (parallel kernels 3/7/15, stacked blocks with pooling; the effort stream has a stride-4 stem) that turns a **150 s** window into **75 tokens (≈ 2 s each)**, followed by bidirectional cross-modal attention across all 75 tokens of the other stream, then mean+max pooling. It models temporal structure at 2 s to 150 s scales and is exactly the model §7f froze. What fails on our hardware is documented in §7c/§7d: the belt-trained effort encoder's representation of *accelerometer* input is separable from belt input at **AUROC 0.997**, healthy nights are flagged on **~94%** of epochs (white-noise effort gives 33%, real accelerometer effort 80–84%), and four unsupervised adaptation attempts did not close it. A recurrent layer on top does not address a **sensor-domain shift**; a recurrent model trained from scratch on 7 events cannot learn the temporal structure that 156,826 MESA epochs taught CANet. (A recurrent head on the frozen 75×128 tokens would have tens of thousands of parameters and be strictly worse-posed.)

### 3. Should a small, heavily regularised recurrent model be tried as a sanity check? Recommendation: no, not now
If tried, the only defensible version would be: 1-layer GRU/LSTM, hidden 8 (≈ 350–800 parameters), dropout, weight decay, **fixed** epochs (no early stopping, because with 6 training events there is no honest validation split), the same 5 s-stride windows, **same leave-one-event-out folds and purge** as §7f, ≥ 10 seeds with seed spread reported, success criterion fixed in advance (beat the hand-crafted head's per-fold minimum **and** pooled AUROC with CI). Why it is not worth running, with numbers from §7f:
1. **Resolution.** At 7 events the pooled AUROC CIs are already 0.58–1.00 (frozen) and 0.71–0.97 (hand-crafted), i.e. ±0.21 and ±0.13; each fold tests 4 positive windows from one event. A recurrent model would have to differ from those by more than that spread to be distinguishable.
2. **Uninterpretable either way.** A poor result cannot separate "LSTMs are unsuited" from "7 events is not enough"; a good result would be suspect (non-convex training, seed luck, and we have already taken 5 feature sets x 4 regularisation values of extra looks on this data, §7f limitation 5).
3. **The linear heads are already the data-limited regime.** The frozen-256 head has 257 parameters, > 7 events, needed C = 0.01, and still failed 2 of 7 folds (0.16 and 0.69). Adding a non-convex recurrent model on the same labels is moving further into the regime where the §7f comparison has no power.
4. **Label noise propagates.** §7e left boundary detection unresolved and §7f documents 30 s-epoch label impurity (19–21 s holds); a temporal model with more freedom fits that noise rather than the hold.
5. **Cost/benefit.** It would cost little compute but near-zero information, and adds a way to mislead (a lucky seed) in a project whose standard is to report honest modest results.
If the advisor still wants it, run it only under the pre-registered protocol above, report all seeds, and describe the outcome as a "feasibility smoke test at n = 7", not as a comparison.

### 4. What would make an LSTM/RNN (or a temporal model trained on our own data) reasonable — concrete conditions
(Derived from the documented CI widths; scaling estimates, not proven thresholds.)
- **More independent events.** CI half-width scales roughly as 1/sqrt(n_events): going from 7 to ≈ 28 events would roughly halve it (frozen ±0.21 → ≈ ±0.10), the minimum to detect a ~0.1 AUROC difference between models. Target ≥ 30 independent labelled holds.
- **More subjects, subject-independent evaluation** (project rule 1): ≥ 10 subjects with ≥ 6 holds each (≥ 60 events) so leave-one-subject-out is possible; today everything is within one subject and one session.
- **Repeat sessions per subject** with rest controls spread across the recording (resolves the §7f temporal-drift limitation) and 3 sync taps / event markers (reduces the 4.8 s boundary uncertainty from §7e) plus the SHORT 5–7 s hold false-positive test.
- **Trustworthy event boundaries** (a validated detector from §7e's second session), so a recurrent model is not trained on boundary noise.
- **A baseline to beat:** the hand-crafted head's pooled AUROC and per-fold minimum from §7f, evaluated by the same leave-one-event/-subject protocol with a purge.
- **Pretraining or labelled accelerometer data at scale** if a temporal model is to learn breathing dynamics itself: MESA has belts only; there is no public labelled accelerometer apnea data (CLAUDE.md §6), so a recurrent model on our hardware needs far more of our own labelled recordings, or CANet fine-tuning (not just a head) once the in-house labelled set is large enough to validate it.

---
## 8. Current status and what's next

| Phase | Status |
|---|---|
| 0 — Environment | Done |
| 1 — Ingestion (loader + R-peaks) | Done |
| 2 — Classical CVHR baseline | Done (71.1% per-minute) |
| 3 — 1D-CNN | **Done, closed** (81.2% / 75.6% / 84.6% per-minute) |
| — Deployable ECG model | **Done** — frozen; tested for plausibility (not accuracy) on real QVAR data, does not transfer as-is (see §7) |
| §4b — ECG feature-extraction comparison study (8 methods) | **Done, exploratory** — confirms method 2 (adopted) is still best; QRS-area (method 6) is **no longer** considered a viable fallback EDR (see §7 — QVAR domain gap is worse for QRS-area than for amplitude) |
| — In-house QVAR validation track (Mam's directive 3) | **Done** — see §7: 15-recording data-quality report, PhysioNet-vs-QVAR domain gap, frozen-model plausibility test, hudson/Hari anomaly investigation |
| 4 — Effort branch (IMU → envelope → cessation detection) | **Built and validated on synthetic data only** (but see §7c: the MESA-belt-trained CANet effort stream does not transfer to our accelerometer) — real IMU signal is present in all 15 in-house recordings (§7) and the chest accelerometer envelope visibly collapses during breath-holds (§7e); cessation *detector* thresholds still unvalidated on real data |
| 5 — Breath-hold validation on our own hardware | **First session done (§7e, 2026-10-07 recording, one subject, 7 holds).** Accelerometer effort drops during holds (ratio range per event in §7e; inner-window ratio < 0.8 in 7/7). Automatic boundary refinement did NOT generalise (3 approaches, 3/7 events visually correct) -> ratios reported as a range. HR direction unconfirmed. Second session needed to validate a detector; short-hold false-positive test not part of this session. |
| 6 — Fusion (fused > ECG-only, the headline claim) | **Shown on MESA** (§7b: CANet AUROC 0.780 vs 0.610 ECG-only, subject-independent) using belts; on our own accelerometer hardware **blocked on 5** (transfer failure, §7c) |
| 7 — Per-night report (events/hour, severity band, no-SpO2 caveat) | Blocked on 6 |
| 8 — Track B (MESA) | **Done for the headline path** (§7b): audit, ECG-only baseline, CANet, ablations, SpO2 ceiling; obstructive/central kept descriptive only (central not learnable) |
| §7d — Sensor-domain adaptation track | **Done (negative result, documented).** Four attempts (CORAL pooled, CORAL token, adversarial v1, adversarial v2) all failed to close the belt→accelerometer gap with current data. Consistent finding: representation-space manipulation insufficient at this data scale. Phase 5 breath-hold protocol is now the critical unblocked path. See `results/adaptation/` and individual FINDINGS.md files. |
| §7f — Supervised transfer (frozen MESA CANet vs hand-crafted effort, own hold labels) | **Done, pilot-scale, negative for the transfer story.** Frozen features tie hand-crafted on pooled AUROC (0.83 vs 0.83) but are less consistent per event (min fold 0.16 vs 0.78) and give 24% vs 0.5% baseline false positives; 150 s vs 30 s context, not apples-to-apples; n = 7 events, one subject. End-to-end wrapper `src/deploy/predict_session.py` works on our own recording (in-sample on the training session; healthy-night flag rate 3–5%). See §7f. |
| §7g — Why not an LSTM/RNN? (technical note) | **Written 2026-10-09; no experiment run; open to advisor input.** CANet is already temporal (multi-scale conv + cross-attention over 75 tokens / 150 s); the open problem is belt-to-accelerometer transfer; 7 events cannot support or evaluate a recurrent model (argument rests on sample size, not parameter count). Records the §7d attempt-1 AUROC erratum (committed separately). |

Phase 4's synthetic-only build means the *code* for both branches now exists and is internally validated; §7 confirms real IMU data is present and usable in all 15 in-house recordings, so the effort branch's remaining gap is running it against real data, not hardware access. What's still genuinely blocked is Phase 5's breath-hold protocol specifically (a recording session that hasn't happened) and the ECG branch's QVAR-specific amplitude recalibration flagged in §7. No further ECG-branch *architecture* work is planned per the "Phase 3 is done, no further tuning" decision — the amplitude-channel fix is a recalibration/normalization task, not a new model.

---

## 9. File and directory map

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
    recordings/
      inhouse_ecg/                   15 in-house SensorTile QVAR-ECG+IMU .txt recordings (§7);
                                      validation-only, gitignored, never committed.

  src/
    ingest/
      physionet.py                   Loads one Apnea-ECG record's ECG signal + per-minute labels.
      sensortile.py                  Loads one SensorTile .txt recording (QVAR+IMU), all channels
                                      kept; empirical sample-rate + timestamp-gap checks (§7).
      inhouse_quality_report.py      §7: per-file R-peak plausibility report over all 15 recordings.
      domain_gap.py                  §7: PhysioNet-vs-QVAR distribution comparison (RR, HR,
                                      R-amplitude, QRS-area).
      hudson_debug.py                §7: waveform/RR-histogram diagnostics for the hudson/Hari
                                      plausibility-test anomaly.
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
      inhouse_plausibility.py        §7: runs the frozen deployable model (2ch + RR-only variant)
                                      over the in-house recordings as a negative control.
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
    inhouse_validation/                   §7: in-house QVAR validation track (all sub-results here).
      quality_report.csv / README.md      Per-file plausibility report, all 15 recordings.
      domain_gap/                         PhysioNet-vs-QVAR distribution summary CSV + histogram
                                           PNGs (RR, HR, R-amplitude, QRS-area) + README.
      plausibility/                       Frozen-model negative-control predictions (2ch + RR-only)
                                           per subject + README.
      hudson_debug/                       Waveform/RR-histogram diagnostic plots + summary.csv +
                                           README for the hudson/Hari anomaly investigation.
    mesa/audit/                           §7b: 250-subject MESA audit (anonymised subjects_anon.csv, summary, funnel, plots).
    mesa/ecg_baseline/                    §7b: ECG-only baseline (grouped 5-fold CV) metrics/config/AHI points (no ids).
    mesa/canet/                           §7b: CANet headline + ablations + SpO2 ceiling, head_to_head.md/.csv, per-variant subdirs.
    inhouse_canet/                        §7c: deployability / domain_gap / negative_control for the MESA-trained CANet on QVAR+IMU.
    adaptation/
      embed_gap/                          §7d step 1: embeddings.npz, embedding_space.png, pca_dim_gap.png, run.log.
      coral/                              §7d step 2: CORAL params, alignment plots, reg sweep, flag rates, FINDINGS.md.
      adaptor/                            §7d step 3: adversarial v1 training history, curves, checkpoint, FINDINGS.md.
      adaptor_v2/                         §7d step 4: adversarial v2 (corrected measurement), training history,
                                           curves, checkpoint, FINDINGS.md, summary.json.
  src/adaptation/                         §7d: sensor-domain adaptation track.
    embed_gap.py                           Step 1: effort-encoder embedding visualisation (PCA + t-SNE),
                                            gap characterisation. Outputs -> results/adaptation/embed_gap/.
    coral_adaptor.py                       Step 2: CORAL (linear covariance alignment) at pooled and
                                            token levels. Negative result — see FINDINGS.md.
    train_adaptor.py                       Step 3: adversarial residual MLP adaptor v1 (hidden=128).
                                            Negative result (overfit) — see FINDINGS.md.
    train_adaptor_v2.py                    Step 4: adversarial adaptor v2 (hidden=32, stronger anchor,
                                            balanced LRs, early-stop guardrail). Negative result (inert).
                                            Fixed a measurement bug (resubstitution AUROC) mid-run.
  src/mesa/                               MESA cardiac + respiratory streams, ECG baseline, CANet, final-model training.
  src/inhouse/                            In-house stream builder (QVAR+IMU -> CANet inputs) and the transfer study driver.
  src/transfer/                           §7f: windows.py (labelled window index + LOEO folds), run_transfer.py (CV), train_heads.py, healthy_check.py.
  src/deploy/predict_session.py           §7f: end-to-end wrapper (raw recording -> per-window hold probability, two heads, plot).
  results/transfer/                       §7f outputs: results.json, oof_*.csv, window_index.csv, fold_sizes.csv, deployed_heads.joblib, demo_breath_hold/, healthy_nights_check.csv.
  src/breath_hold/analyze_session.py      §7e: breath-hold session analysis (R-peaks, despiked HR, effort envelope,
                                           3-way effort-ratio windows, boundary-detector attempt + flags).
  results/breath_hold/                    §7e outputs: event_summary.csv, event_XX.png, session_overview.png, summary.json, FINDINGS.md.
  docs/breath_hold_protocol.md            Monday's breath-hold recording checklist (Phase 5).
```
