# Cardiorespiratory Sleep Apnea Screening — Full Presentation Content

**Purpose of this file:** the complete, ordered fact base for the faculty-mentor Word document and PowerPoint. Every number below was read from the repo's result files (source path given in each section) or recomputed from the prediction CSVs. Nothing is from memory. Plain wording will be rewritten later; keep every number.

**Reading rule used throughout:** every result states its *evaluation type*. "LOSO" = leave-one-subject-out. "Grouped 5-fold" = 5-fold cross-validation where whole subjects are held out. All reported results are subject-independent (no person is ever in both training and test).

---

## 1. What the project is

- A **wearable sleep-apnea screening system** (screening/triage, NOT a diagnostic device): it flags people who should go for a proper clinical sleep study.
- **One chest-worn sensor** records two signals:
  - **ECG** (heart's electrical activity) — recorded through the QVAR channel of an ST SensorTile.box PRO + STEVAL-MKI242A adapter wired to chest electrodes (Lead-II placement).
  - **Respiratory effort** (chest-wall movement) — from the same device's onboard IMU (accelerometer + gyroscope).
- **Scientific claim:** fusing heart + breathing signals beats using either alone.
- **Architecture:** two branches, **decision-level fusion**.
  - ECG branch (trained on public data): ECG → R-peaks → RR-interval series → features → classical detector or 1D-CNN → probability of apnea per minute.
  - Effort branch (originally signal-processing only): IMU → acceleration magnitude → motion mask → band-pass 0.08–0.6 Hz (4.8–36 breaths/min) → Hilbert envelope → breathing rate and effort amplitude.
  - Fusion → per-night report (events, AHI-like rate, severity).
  - Reason for decision-level (not one end-to-end model): the branches train on different data, and each stays explainable.
- Context: student project (2 students) with an ST Microelectronics industry collaboration, under faculty guidance; part of a larger sleep-analysis programme ("Pinnacle"). Our track = apnea. (A sister team does EEG-free sleep staging — a different track, not covered here.)
- Hardware: 5 SensorTile units. QVAR (always ECG here) and IMU sampled at a nominal 240 Hz.

## 2. Why it matters

- Sleep apnea affects about **1 in 7 adults**; about **80% of moderate-to-severe cases are undiagnosed** (project brief figures).
- The only definitive test is overnight **polysomnography (PSG)**: expensive, uncomfortable, lab-bound, long waitlists.
- A cheap wearable screen closes the gap between "has apnea" and "gets diagnosed".
- Apnea is a **cardiorespiratory** event. ECG-only detection tops out around 85–88% in the literature and cannot tell apnea *types* apart; adding respiratory effort improves both.
- Severity scale (AHI = events per hour): <5 normal, 5–15 mild, 15–30 moderate, >30 severe.
- Obstructive apnea = airway blocked, chest effort continues. Central apnea = effort stops. Only an effort signal can distinguish them.

## 3. The core constraint

- Our own volunteers are **healthy 19–20-year-olds who essentially do not have apnea**.
- Therefore:
  - **Training data must be public, expert-labelled clinical data** — the only place real apnea exists for us.
  - **Our own recordings are validation-only. We never train on them.** (Any experiment training on them would be wrong by design.)
  - Validation plan on our hardware: **voluntary breath-holds** (awake, timestamped) as safe, perfectly-labelled apnea-like events, plus healthy overnight recordings as negative controls.
- Methodology rules followed: subject-independent evaluation always; per-minute and per-recording reported separately; classical baseline is the floor the CNN must beat; start simple (classical → 1D-CNN → only then anything larger); any suspiciously perfect number (>0.95 accuracy) is treated as a bug and checked for subject leakage first; sample rates verified empirically.

## 4. Data used

### 4.1 PhysioNet Apnea-ECG (training/evaluation for the ECG branch)
*Source: docs/PROGRESS.md §2; results/phase2_cvhr_baseline, results/phase3_cnn.*
- Single-lead ECG, **100 Hz**, with expert **per-minute apnea/normal labels**.
- **35 labelled records** used: a01–a20 (apnea group), b01–b05 (borderline), c01–c10 (control/healthy). 35 further withheld test records (x01–x35) are unlabelled and not used.
- **16,949 usable minutes** across the 35 records (96 minutes dropped: fewer than 3 beats, or a record's first minute where a delta feature is undefined).
- Loader sanity checks (reproduced): record a01 = 470 apnea / 19 normal minutes, 29,992 R-peaks, mean HR 60.9 bpm; record c01 = 0 apnea / 484 normal, 27,719 R-peaks, mean HR 58.0 bpm; implausible RR intervals 0.00% and 0.02%.
- CVHR sanity check on a01 (cyclic variation of heart rate): apnea-minute mean HR-swing **36.37 bpm vs 26.63 bpm** for normal minutes (SDNN 138.5 ms vs 80.9 ms) — the expected apnea fingerprint is present in the data.

### 4.2 MESA via NSRR (training/evaluation for the fusion model, "Track B")
*Source: results/mesa/audit/summary.md, summary.json, usable_funnel.csv, docs/MESA_NOTES.md.*
- MESA Sleep ancillary study (Exam 5, home PSG). Elderly cohort: in our 250-subject subset, age median 70 (range 55–92), 50% male.
- Signals per subject: **ECG** (256 Hz), **thoracic and abdominal respiratory-inductance belts** (32 Hz each), **SpO2** (1 Hz), plus expert-scored respiratory events.
- **250 subjects downloaded (EDF + event XML), 0 download/parse errors → 220 usable.**
- **Usable-subject funnel** (cumulative):

| Criterion | Subjects |
|---|---|
| Complete (EDF + XML) | 250 |
| ECG + SpO2 + Thor + Abdo all present | 250 |
| XML has sleep-stage events | 250 |
| Sleep time ≥ 4 h | 224 |
| Both belts not flat (flat fraction < 0.5) | 222 |
| Both belts show breathing in ≥ 50% of epochs | 220 |
| ≥ 1 scored apnea/hypopnea event | **220** |

- **Scored events (250 subjects):** obstructive apnea 6,071; central apnea 849; mixed apnea 4; hypopnea 27,746. Subjects with ≥1 central apnea: 106 (≥5: 30); with ≥1 obstructive: 208.
- **Epoch class balance** (30 s epochs, sleep epochs only, all 250): apnea 5.7%, hypopnea 19.8%, any respiratory event 25.4%. Usable 220 subjects: apnea 5.5%, hypopnea 19.6%, event 25.1%. The model-training set (156,826 sleep epochs after dropping epochs with <10 valid beats) has event prevalence **25.4%** (= chance AUPRC 0.254).
- Within apnea epochs: 89.5% obstructive, 10.5% central, mixed ≈ 0 (PROGRESS §7b). Central apnea is rare and concentrated: zero central events in 123 of 220 usable subjects (PROGRESS §7b), so obstructive-vs-central is not learnable and is kept descriptive only.
- **AHI depends on the hypopnea scoring rule** (always quote the rule). Audited subset, NSRR official: ahi_a0h3a (3% desaturation or arousal) median 20.2 [IQR 10.8–37.3]; ahi_a0h4 (4% desaturation) median 10.8 [IQR 3.6–23.5]. Event-XML all events median 21.6 [13.5–32.2]. Severity counts, a0h4: <5 → 80, 5–15 → 76, 15–30 → 49, ≥30 → 45 (n=250). Subset is representative: full MESA cohort (n=2,057) a0h3a median 18.4, a0h4 median 9.1.
- Label cross-check: our XML-derived AHI vs NSRR official, Pearson r = 0.714 (a0h3) and 0.748 (a0h3a); sleep-time check vs NSRR: median absolute difference 0.0 min, max 1.0 min.
- Belt quality: median breathing rate 14 bpm on both belts; median agreement between belts |r| = 0.82; breathing-visible epoch fraction median 0.857 (thoracic) / 0.887 (abdominal); cleaner belt per subject: thoracic 65, abdominal 183, tie 2.
- Important: MESA effort belts are **respiratory inductance plethysmography (RIP) bands** — they measure chest/abdomen circumference change, not acceleration. They are a *proxy* for our accelerometer, not the same signal.
- Mismatch to state: MESA is old (≈70), our volunteers are 19–20.
- Figures: `mesa_audit_overview.png`, `mesa_usable_funnel.png`.

### 4.3 Our own SensorTile recordings (validation only)
*Source: docs/PROGRESS.md §7 and §7c; results/inhouse_validation, results/inhouse_canet.*
- **15 recordings** (≈9 volunteers; healthy, 19–20 y, no apnea labels). QVAR-ECG + IMU at ~240 Hz.
- **Never trained on, never committed to git** (data/recordings/ is gitignored). Volunteers are anonymised as S01… in all outputs.
- Data quality (inhouse_validation): 14/15 files plausible (mean HR 44.7–90.0 bpm; implausible-RR ≤ 0.23%), one aborted 1.3 s capture, one noisier file (1.09% implausible RR).
- **Empirically measured sample rate: 238.1–243.2 Hz**, a consistent ~1–1.3% offset from the nominal 240 Hz — real across every file, which is why RR features are kept in time units (seconds) rather than sample counts.
- The accelerometer channel is present and non-flat in all 15 files.
- For the CANet transfer study (§9): **10 of 15 recordings usable (≥ 600 s; 5 are 1–32 s aborted captures)**, 0.8–7.5 h each.

---

## 5. Phase 2 — classical baseline (the floor)
*Source: results/phase2_cvhr_baseline/*.csv (recomputed); docs/PROGRESS.md.*
- Method: 7 hand-crafted per-minute features (mean HR, **HR-swing** = max−min instantaneous HR in the minute, SDNN, RMSSD, pNN50, mean RR, change in mean HR vs previous minute) → **logistic regression** (class-balanced, standardised features).
- Evaluation: **35-fold LOSO, per-minute**, 16,949 minutes.
- **Accuracy 0.711 | sensitivity 0.637 | specificity 0.758** | precision 0.621 | F1 0.629.
- **Per-recording** (flag a recording "apnea" if >5% of its minutes are predicted apnea, vs official group): **26/35 = 0.743**; all 9 errors were control recordings flagged as apnea. Not fixable by choosing a better threshold (apnea and control recordings overlap heavily, e.g. control c08 = 45% flagged vs apnea b05 = 12%). Reported as a known-weak diagnostic, not a claim.
- Literature context: classical CVHR detector r = 0.91 vs AHI (Hayano 2011); a from-scratch version was expected at 72–80% per-minute, so 71.1% is the honest floor.

## 6. Phase 3 — 1D-CNN
*Source: results/phase3_cnn/README.md and prediction CSVs (all recomputed and matching).*
- Input: each minute's RR-interval series resampled onto a fixed 60-point grid (in seconds); second channel = **R-peak amplitude** (an ECG-derived-respiration proxy). Model `RRCNN`: 2 conv layers (16→32 channels) + global average pool + 2 FC layers. Deliberately tiny (35 subjects is small).
- Evaluation: 35-fold LOSO, same 16,949 minutes as Phase 2 (so directly comparable).

**Progression (per-minute, LOSO):**

| Step | Input | Loss weighting | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Phase 2 baseline | hand-crafted features, logistic regression | balanced | 0.711 | 0.637 | 0.758 |
| v1 CNN | RR-interval, 1 channel | balanced | 0.775 | 0.653 | 0.851 |
| v2 CNN | RR + R-amplitude, 2 channels | balanced × 1.5 | 0.798 | 0.754 | 0.825 |
| **FINAL (Variant A)** | **RR + R-amplitude, 2 channels** | **balanced (default)** | **0.812** | **0.756** | **0.846** |

- **Final per-recording: 28/35 = 0.800** (unchanged across v1/v2/final).
- Gain over Phase 2: accuracy +0.101, sensitivity +0.119, specificity +0.088 (computed from the table).
- **Ablation finding** (v2 changed two things at once, so they were isolated):

| Variant | Accuracy | Sensitivity | Specificity |
|---|---|---|---|
| v1 (1 ch, default weight) | 0.775 | 0.653 | 0.851 |
| **A: 2 ch, default weight (= FINAL)** | **0.812** | **0.756** | **0.846** |
| B: 1 ch, 1.5× weight | 0.758 | 0.715 | 0.786 |
| v2: 2 ch, 1.5× weight | 0.798 | 0.754 | 0.825 |

  **The R-amplitude channel does essentially all the work.** The 1.5× loss weighting alone (B) is a net loss vs v1 (accuracy 0.775 → 0.758: buys a little sensitivity for a larger specificity cost), and stacking it on the amplitude channel (v2) is worse than A on every metric. Decision: keep the amplitude channel, drop the extra weighting.
- Honest limits: sensitivity 0.756 still misses about 1 in 4 true apnea minutes (weak for a screening tool). 0.812 is below the 85–88% literature target for a 1D-CNN (Chang 2020) because we stopped at the first config that beat the floor ("start simple"). LOSO is a cross-validation protocol: 35 models were trained and discarded.
- Process caveats that were caught: first CNN run was undertrained (15 epochs → 30); a printing bug produced an absurd "+0.775" improvement and was caught before reporting. 0.812 is well under the 0.95 leakage-suspicion threshold.
- A **deployable model** (same architecture, trained once on all 35 records) was frozen; its training-set fit (0.834) is not a generalisation estimate.
- Figure: `phase3_progression.png`.

## 7. The eight-method ECG feature study (master table)
*Source: results/feature_study/master_results.csv and master_results.md.*
- Question: would another ECG representation beat the adopted one? Same single-lead ECG, same 35-fold LOSO, same 16,949-minute set (method 4: 16,946; method 5: 16,207 minutes, where features were undefined for a few).
- Scalar HRV banks (methods 3–5) were run through three classifiers (1D-CNN, logistic regression, small MLP) and the best is shown; **logistic regression won all three**.

| # | Method | Family | Classifier | Acc | Sens | Spec | Prec | F1 | Per-recording acc |
|---|---|---|---|---|---|---|---|---|---|
| 1 | RR intervals (= Phase 3 v1) | time | 1D-CNN | 0.775 | 0.653 | 0.851 | 0.732 | 0.691 | 0.800 |
| 2 | RR + R-peak amplitude (**adopted**) | time | 1D-CNN | **0.812** | 0.756 | 0.846 | 0.754 | 0.755 | 0.800 |
| 3 | Time-domain HRV bank (SDNN, RMSSD, pNN50, mean HR, HR range, triangular index) | time | logistic regression | 0.702 | 0.620 | 0.754 | 0.611 | 0.615 | 0.743 |
| 4 | Frequency-domain HRV (VLF/LF/HF power, LF:HF) | frequency | logistic regression | 0.672 | 0.513 | 0.772 | 0.584 | 0.546 | 0.743 |
| 5 | Nonlinear / Poincaré (SD1, SD2, SD1:SD2, sample entropy) | nonlinear | logistic regression | 0.707 | 0.673 | 0.728 | 0.617 | 0.644 | 0.743 |
| 6 | RR + QRS-area (whole-complex EDR proxy) | time | 1D-CNN | 0.798 | 0.687 | 0.868 | 0.764 | 0.724 | **0.829** |
| 7 | CWT scalogram of RR series | image | 2D-CNN | 0.749 | 0.491 | 0.911 | 0.774 | 0.601 | 0.771 |
| 8 | Spectrogram of raw ECG | image | 2D-CNN | 0.534 | 0.673 | 0.448 | 0.432 | 0.526 | 0.800 |

**Findings:**
1. Real within-minute **time series** (methods 1, 2, 6) are the only ones above 0.79 accuracy; collapsing a minute into a few numbers first (3, 4, 5) caps at about 0.67–0.71, near the Phase 2 floor (0.711).
2. R-peak amplitude is not uniquely special: QRS-area (method 6) is within 1.4 accuracy points (0.798 vs 0.812) and has the best per-recording accuracy (29/35 = 0.829).
3. Classifier choice matters for weak features: forcing scalar banks through the CNN cost 7–22 accuracy points (method 4 specificity collapsed to 0.293 under the CNN; logistic regression recovered it to 0.772).
4. Frequency-domain HRV is weakest among non-image methods (as flagged beforehand: VLF needs ~5.5 min of data, we have 1).
5. Image representations swing sharply between sensitivity and specificity (method 7: 0.491/0.911; method 8: 0.673/0.448); method 8, the only one not derived from R-peaks, is the weakest per-minute.
6. **Method 2 stays the best per-minute result, so Phase 3 is not reopened.** No method triggered the 0.95 leakage flag.
- Later overturned point: QRS-area looked like a fallback if amplitude failed to transfer to our hardware; the in-house study (§9, domain-gap table) showed QRS-area's gap is worse, so it is **not** a usable fallback.
- Limits: 35 subjects is small for 2D-CNNs; method 4 reported for completeness only.
- Figure: `feature_study_comparison.png`.

---

## 8. MESA: ECG-only baseline vs CANet (the fusion result)
*Source: results/mesa/ecg_baseline/, results/mesa/canet/ (README.md, head_to_head.md/.csv, headline_vs_ablations.txt, metrics.json, fold_metrics.csv).*

**Setup (identical for every model):** 220 subjects; **grouped 5-fold CV** (per fold 150 / 26 / 44 subjects for train / inner-validation / test; early stopping and operating threshold chosen from inner-validation subjects only); **30-second sleep epochs**, 156,826 test epochs pooled; target = **respiratory event (apnea or hypopnea) vs normal**, prevalence 25.4%. Headline model uses **no SpO2 and no arousal** (our wearable has neither).

### 8.1 ECG-only baseline
- Input: RR + R-amplitude (2 Hz grid, target epoch ±2 neighbours = 150 s window, per-subject label-free normalisation) → 1D-CNN, weighted BCE.
- Pooled: **AUROC 0.610** (0.6088 unrounded; per-fold mean 0.610 ± 0.030), **AUPRC 0.335** (chance 0.254; per-fold 0.339 ± 0.055), sensitivity 0.622, specificity 0.535, precision 0.313, F1 0.416.
- Confusion (epochs): TP 24,768; TN 62,630; FP 54,396; FN 15,032.
- Subject-level AHI correlation (epoch-rate proxy, n=220): **r = 0.31 vs a0h3, 0.30 vs a0h4** (Spearman 0.31/0.31). For reference, the true event-epoch rate itself correlates r = 0.69 / 0.67 — the ceiling of this proxy.
- Reading: ECG alone is weak on MESA — a much harder, older, hypopnea-heavy cohort than PhysioNet (where per-minute accuracy was 0.812; the tasks and cohorts are different and not directly comparable).

### 8.2 CANet (Cross-Attention Network) — headline
- Two streams: **cardiac** (identical input to the baseline) and **respiratory effort** (thoracic + abdominal belts at 32 Hz). Multi-scale 1D-CNN encoders (kernel sizes 3/7/15), **bidirectional cross-modal attention** between the streams, global average + max pooling → MLP. **169k parameters.** Same folds, same epochs, same threshold discipline (asserted in code).
- Pooled results: **AUROC 0.780, AUPRC 0.490**, sensitivity 0.822, specificity 0.613, precision 0.419, F1 0.555. Confusion: TP 32,701; TN 71,769; FP 45,257; FN 7,099. Per-fold AUROC 0.7616 / 0.8042 / 0.7822 / 0.7801 / 0.7726 (mean 0.780 ± 0.016); per-fold baseline AUROC 0.564–0.642. **Every fold improves.**
- **Gain over ECG-only (paired bootstrap over subjects, 200 reps):** ΔAUROC **+0.171** (95% CI +0.151 to +0.191); ΔAUPRC **+0.155** (95% CI +0.132 to +0.184).
- Subject-level AHI correlation (epoch-rate proxy): **r = 0.74 (a0h3) / 0.70 (a0h4)**, up from 0.31 / 0.30.

### 8.3 Ablations and ceiling (same folds, same epochs)

| Model | Params | AUROC | AUPRC | Sens | Spec | F1 | AHI r (a0h3 / a0h4) |
|---|---|---|---|---|---|---|---|
| ECG-only baseline | ~0.1M | 0.609 | 0.335 | 0.622 | 0.535 | 0.416 | 0.31 / 0.30 |
| **CANet (ECG + effort), headline** | 169k | **0.780** | **0.490** | 0.822 | 0.613 | 0.555 | 0.74 / 0.70 |
| Ablation: effort only | 54k | 0.747 | 0.422 | 0.837 | 0.570 | 0.540 | 0.71 / 0.68 |
| Ablation: ECG + effort, concatenation (no attention) | 102k | 0.744 | 0.424 | 0.819 | 0.582 | 0.538 | 0.72 / 0.69 |
| CEILING: CANet + SpO2 (**not on our wearable**) | 250k | 0.813 | 0.541 | 0.853 | 0.641 | 0.587 | 0.82 / 0.77 |

- Headline vs concat fusion: ΔAUROC **+0.036** (CI +0.029 to +0.045), ΔAUPRC +0.066 (CI +0.046 to +0.084).
- Headline vs effort-only: ΔAUROC **+0.033** (CI +0.023 to +0.042), ΔAUPRC +0.068 (CI +0.044 to +0.092).
- Vs ECG-only baseline: effort-only +0.138 AUROC (CI +0.116 to +0.158); concat +0.136 (CI +0.113 to +0.154); SpO2 ceiling +0.204 (CI +0.183 to +0.224).
- SpO2 ceiling = what adding oxygen saturation would buy (+0.033 AUROC over headline). It is an upper bound, not something our hardware can supply.
- Interpretation: most of the gain comes from the effort signal (effort-only is already 0.747); fusing ECG through attention adds a smaller but statistically clear gain on top (+0.033 to +0.036 AUROC, CIs exclude 0).

### 8.4 Per-class sensitivity (each model at its own validation-chosen threshold)

| Model | Obstructive apnea | Central apnea | Hypopnea | Specificity (normal) |
|---|---|---|---|---|
| ECG-only baseline | 0.829 | 0.840 | 0.564 | 0.535 |
| **CANet (headline)** | **0.969** | **0.981** | **0.780** | 0.613 |
| Effort only | 0.987 | 1.000 | 0.795 | 0.570 |
| Concat | 0.982 | 0.994 | 0.773 | 0.582 |
| + SpO2 ceiling | 0.914 | 0.945 | 0.835 | 0.641 |

- Headline gains: obstructive apnea **0.83 → 0.97**, central apnea **0.84 → 0.98**, hypopnea **0.56 → 0.78**, normal-epoch specificity 0.54 → 0.61.
- Reading: apnea (a complete stop in breathing) is easy once effort is available; hypopnea (partial reduction) is the harder, dominant class.

### 8.5 Honest caveats (state all of these)
1. Most of the ECG-only → CANet gain comes from the **effort signal**; ECG via attention adds a smaller, real gain.
2. The concat ablation is **not parameter-matched** (102k vs 169k), so the attention gain cannot be cleanly attributed to attention itself.
3. **Single seed per fold**, no hyperparameter search; fold-to-fold SD ≈ 0.016 AUROC (0.0157 headline; 0.0302 baseline).
4. "AHI" here is an **epoch-rate proxy** (flagged 30-s epochs per sleep hour), not an event-scored AHI. With no SpO2, desaturation-based definitions cannot be matched.
5. **Specificity is only 0.61.** Sensitivities use lenient Youden thresholds, so compare AUROC/AUPRC across models, not sensitivity alone.
6. Hypopnea dominates the positives and is the weakest class. The usable cohort is skewed to disease (median all-event AHI ≈ 21).
7. Obstructive-vs-central is descriptive only (central too rare to learn).
8. **MESA effort comes from RIP belts, not accelerometers** — see §9 for what that does to transfer.
9. Elderly cohort (median age 70) vs our 19–20-year-old volunteers.
- Figures: `canet_vs_baseline.png`, `per_class_gain.png`, `mesa_ecg_baseline_overview.png`.

---

## 9. In-house hardware test: does MESA-trained CANet work on our device?
*Source: results/inhouse_canet/ (README.md, deployability/, domain_gap/, negative_control/), docs/PROGRESS.md §7 and §7c.*

**Framing: this is a TRANSFER finding, NOT an apnea result.** The cohort is healthy, unlabelled, validation-only, never trained on, never committed. No apnea accuracy is claimed anywhere in this section. Sleep state is unknown (no staging); rates are per recorded hour; 10 of 15 recordings usable, 0.8–7.5 h each.

### 9.1 Earlier in-house checks (ECG branch, PhysioNet-trained CNN)
*Source: docs/PROGRESS.md §7 (results/inhouse_validation).*
- **Domain gap, PhysioNet vs our QVAR** (13 plausible in-house files vs 35 PhysioNet records, per-beat means):

| Channel | PhysioNet mean | QVAR mean | Ratio |
|---|---|---|---|
| RR interval (s) | 0.908 | 1.066 | 1.17× |
| R-peak amplitude | 1.295 | 140.5 | ~108× |
| QRS-area | 0.047 | 12.60 | ~268× |

  RR interval transfers (the 17% gap is young-healthy vs older-clinical physiology). Amplitude and QRS-area do not (QVAR is an uncalibrated electrostatic sensor, not a calibrated ECG amplifier).
- **Frozen-model plausibility** (negative control: healthy people should get a LOW apnea rate; 9 of 13 plausible files had ≥1 usable minute, 2,027 minutes total): the 2-channel PhysioNet CNN predicts an implausible **52.8%** apnea-minute rate; neutralising the amplitude channel drops it to **21.3%**. The amplitude channel is the dominant failure.
- Two recordings with a ~45 bpm resting heart rate stayed implausible even without amplitude: a model-generalisation blind spot (PhysioNet training mean 68 ± 11.8 bpm). One recording had a 10–15 s electrode-contact artifact the IMU-based motion gate cannot see → there is no ECG-quality gate yet.

### 9.2 Deployability: yes
- QVAR → R-peaks → cardiac stream (same grid as MESA; **per-night normalisation neutralises the ~108× amplitude-scale gap by construction**) and accelerometer → breathing trace (`envelope.py`) → 32 Hz with MESA's belt preprocessing → MESA-trained CANet → per-epoch probability → per-night rate.
- **Runs end to end for all 10 usable nights.** Technically deployable; the numbers are not meaningful (see 9.4). (Weights refit on MESA only for this purpose.)

### 9.3 Belt vs accelerometer domain gap
- A classifier separates MESA belt epochs from our accelerometer epochs with **AUROC 0.99** on handcrafted features (0.9939 for acceleration magnitude, 0.9904 for principal axis) and **0.997–0.998** on the effort-encoder embedding (0.9967 / 0.9979). **Control (MESA half vs MESA half): 0.51 (0.5066) / 0.47 (0.4741).** 0.5 = indistinguishable.
- Breathing IS in the accelerometer (spectral peak near 0.2–0.25 Hz; median ~16–18 bpm, vs 14 for belts) but the signal is broadband and irregular:

| Feature | Belts (MESA) | In-house accelerometer |
|---|---|---|
| Spectral entropy | 0.46 (0.4646) | 0.64–0.66 (magnitude 0.661); night medians at the 87th–98th MESA percentile |
| Zero-crossings per minute | 30 | 40–42 |
| Breathing-rate spread (IQR) | 4 bpm | 10 bpm |

  Amplitude after normalisation is comparable (RMS ≈ 0.9–1.0; belt median 1.02, accelerometer 0.96 / 0.90), so it is the **shape**, not the size, that differs.
- Caveat: posture/activity of these recordings is unverified; part of the gap may be behaviour rather than sensor.

### 9.4 Healthy negative control
Fraction of epochs flagged (median over 10 nights) at the MESA operating thresholds, vs MESA out-of-fold low-AHI subjects (a0h4 < 5, n = 67):

| Model | In-house healthy | MESA low-AHI median [90th pct] | Nights above MESA 90th pct |
|---|---|---|---|
| ECG-only | 0.58 | 0.43 [0.67] | 4 / 10 |
| Effort-only (accelerometer magnitude / principal axis) | 0.84 / 0.80 | 0.35 [0.55] | 10 / 10 |
| **Headline CANet (magnitude / principal axis)** | **0.94 / 0.92** | 0.28 [0.48] | 10 / 10 |
| Headline, effort set to zero / white noise | 0.68 / 0.62 | — | 7 / 10 and 9 / 10 |
| Effort-only, **white-noise** effort | **0.33** | 0.35 [0.55] | 0 / 10 |

(Unrounded: ECG-only 0.581; effort-only 0.837/0.803; headline 0.944/0.917; headline effort=0 0.685; effort=white noise 0.625; effort-only white noise 0.327.)
- **Verdict: implausibly high — a transfer failure, not a result.** Healthy 19–20-year-olds should sit at or below the MESA low-AHI level; the headline model flags >90%. The model flags every night above the MESA 90th percentile (10/10). Note that even MESA low-AHI subjects are flagged ~28% of the time at these thresholds; that, not zero, is the right comparator.
- **The noise ablation isolates the effort branch:** pure white-noise effort gives 33% (inside the MESA range) while real accelerometer effort gives 80–84%. The model reads our structured-but-wrong-morphology trace as disrupted breathing.
- The **cardiac side is mildly elevated** (58%, 4/10 nights above the MESA 90th percentile). Possible respiratory sinus arrhythmia in young hearts — **unverified**. Fusion compounds both.
- **Conclusion:** the belt-trained effort stream does not transfer to accelerometer input. MESA belts remain the only validated effort input; the effort branch needs adaptation or an accelerometer-native model. Whether our accelerometer can detect cessation at all is now the critical open question.
- Figures: `inhouse_domain_gap_separability.png`, `inhouse_domain_gap_spectrum.png`, `inhouse_domain_gap_amplitude_shape.png`, `inhouse_negative_control_summary.png`, `inhouse_negative_control_per_night.png`.

---

## 10. Current status (phase table)
*Source: docs/PROGRESS.md §8.*

| Phase | Status |
|---|---|
| 0 Environment; 1 Ingestion | Done |
| 2 Classical CVHR baseline | Done — 0.711 per-minute |
| 3 1D-CNN | Done, closed — 0.812 / 0.756 / 0.846 |
| Deployable ECG model | Frozen; plausibility-tested only, does not transfer as-is |
| 8-method feature study | Done, exploratory — method 2 still best |
| In-house QVAR validation | Done |
| 4 Effort branch (IMU) | Built; validated on synthetic data only (synthetic demo: estimated 15.0 bpm vs true 15.0 bpm; 3 of 3 real gaps of 20/15/12 s detected; 7 s gap correctly ignored). Not yet run against real breath-holds |
| 5 Breath-hold validation | **The critical experiment; protocol written, session pending** |
| 6 Fusion (fused > ECG-only) | Shown on MESA (CANet 0.780 vs 0.610 AUROC); on our hardware blocked on Phase 5 |
| 7 Per-night report (events/h, severity, no-SpO2 caveat) | Blocked on 6 |
| 8 MESA track | Done for the headline path |

## 11. What's next
*Source: docs/breath_hold_protocol.md.*
1. **Monday: breath-hold recording session.** Question: *can the accelerometer effort signal on our hardware detect cessation of breathing at all?* A breath-hold is quiet, not noisy — it should show up as an effort-envelope collapse. If it does not on a well-placed device, the accelerometer is the problem, not the model.
   - Subject supine; strap the SensorTile **firmly and flat to the mid-sternum** (placement is the main variable; a loose box gives weak effort — the failure mode already seen).
   - Procedure (~12–15 min per subject): start recording; 3 sharp taps within 10 s as a sync marker; 2 min quiet breathing; **5–6 voluntary breath-holds of 15–25 s** with 90 s breathing between; **one deliberate 5–7 s hold that should NOT be detected** (false-positive test); 2 min recovery breathing; timestamped clock log is the ground truth.
   - Safety: voluntary and short only; skip anyone with respiratory/cardiac conditions; stop for any dizziness or discomfort.
   - Scope: one clean subject validates the protocol; expand to 3–5 subjects the same week for a reportable result. Result type: a detectability/plausibility result, **not** an apnea-accuracy number.
2. **Adapt the effort branch to the accelerometer** (belt-like calibration, or an accelerometer-native / MESA-independent effort model), depending on what the breath-holds show.
3. Remaining engineering gaps found so far: QVAR-specific amplitude recalibration (or the per-night normalisation used for CANet); an ECG-based signal-quality gate; the 100 Hz vs ~240 Hz reconciliation is handled by working in time units.
4. Then Phase 6 on our hardware and the Phase 7 per-night report, naming the no-SpO2 hypopnea limitation.

## 12. Headline messages (for the summary slide)
1. ECG alone, from public data, reaches **0.812 per-minute accuracy** (LOSO) — above the classical floor of 0.711.
2. On the larger, harder MESA data, **adding breathing effort lifts AUROC from 0.610 to 0.780** (ΔAUROC +0.171, CI +0.151 to +0.191): the fusion claim holds, with belts.
3. Most of that gain is the effort signal; fusion adds +0.033 to +0.036 AUROC beyond effort-only and concatenation (the attention gain is not parameter-matched).
4. On our own hardware the pipeline runs end to end, but the belt-trained effort stream does **not** transfer to accelerometer input (0.99 separability; healthy flag rate 94%; isolated to the effort branch by the noise ablation). This is a transfer finding, not an apnea result.
5. Next: Monday breath-holds decide whether the accelerometer can detect cessation at all.

---

## 13. Glossary

| Term | Plain-English meaning |
|---|---|
| **Sleep apnea** | Repeated stops in breathing during sleep (≥10 s). |
| **Apnea** | A (near-)complete stop of airflow for ≥10 s. |
| **Hypopnea** | A partial reduction in breathing (≥30–50% for ≥10 s, depending on the rule) — shallower breathing, not a full stop. |
| **Obstructive apnea** | Airway blocked; the chest keeps trying to breathe. |
| **Central apnea** | The brain stops sending the breathing signal; chest effort stops too. |
| **Mixed apnea** | Starts central, ends obstructive. |
| **AHI (Apnea-Hypopnea Index)** | Apnea + hypopnea events per hour of sleep; <5 normal, 5–15 mild, 15–30 moderate, >30 severe. |
| **Polysomnography (PSG)** | The overnight lab sleep study; the clinical gold standard. |
| **Screening vs diagnosis** | Screening flags who should get tested; diagnosis is the confirmed medical conclusion. |
| **ECG (electrocardiogram)** | Recording of the heart's electrical activity. |
| **QVAR** | The SensorTile's electrostatic sensing channel; we wire it to chest electrodes to record ECG. |
| **IMU** | Motion sensor chip (accelerometer + gyroscope); here it records chest movement. |
| **Accelerometer** | Sensor measuring acceleration; on the chest it picks up breathing motion. |
| **R-peak** | The tall spike of each heartbeat in the ECG. |
| **RR-interval** | Time between consecutive R-peaks (one heartbeat to the next), in seconds. |
| **HRV (heart-rate variability)** | How much the beat-to-beat timing varies. |
| **SDNN / RMSSD / pNN50** | Standard HRV measures: overall spread of RR intervals / beat-to-beat change size / share of successive beats differing by >50 ms. |
| **CVHR (cyclic variation of heart rate)** | Apnea's cardiac fingerprint: heart rate slows during the pause, then surges when breathing resumes. |
| **HR-swing** | Max minus min instantaneous heart rate within a minute; a simple CVHR measure. |
| **EDR (ECG-derived respiration)** | Breathing inferred from how ECG amplitude/shape changes with each breath. |
| **R-amplitude / QRS-area** | Two EDR measures: height of the R-peak / area under the whole heartbeat complex. |
| **VLF / LF / HF power** | Heart-rate fluctuation power in very-low / low / high frequency bands; HF mostly reflects breathing. |
| **Poincaré (SD1, SD2), sample entropy** | Nonlinear HRV measures: short- and long-axis spread of the beat-to-beat scatter plot / how unpredictable the series is. |
| **Respiratory sinus arrhythmia (RSA)** | Heart speeding up on inhale and slowing on exhale; strong in young people. |
| **Bradycardia / tachycardia** | Slow / fast heart rate. |
| **Epoch** | One analysis window: 60 s for PhysioNet labels, 30 s for MESA. |
| **Per-minute vs per-recording** | Judging each minute separately vs judging a whole person/night; different tasks, always reported separately. |
| **Sensitivity (recall, true-positive rate)** | Of the truly-apnea minutes/epochs, the fraction caught. |
| **Specificity** | Of the truly-normal ones, the fraction correctly left alone. |
| **Precision / F1** | Of the flagged, the fraction that were truly apnea / the balance of precision and sensitivity. |
| **Accuracy** | Fraction of all predictions that were right (misleading when classes are imbalanced). |
| **AUROC** | Area under the ROC curve: chance of ranking a random positive above a random negative; 0.5 = coin flip, 1.0 = perfect. Does not depend on the threshold. |
| **AUPRC** | Area under the precision–recall curve; more informative with rare positives; chance level = the positive rate (0.254 here). |
| **Youden threshold** | The cut-off that maximises sensitivity + specificity − 1, chosen on validation data only. |
| **Confidence interval (95% CI)** | Range the true value plausibly lies in; if a difference's CI excludes 0, it is statistically clear. |
| **Paired subject bootstrap** | Re-sampling subjects many times (200 here) to get a CI for a difference between two models on the same subjects. |
| **LOSO (leave-one-subject-out)** | Each fold holds out one entire person; train on the rest, test on them; repeat for every person. |
| **Grouped k-fold** | Cross-validation that keeps all of a person's data in the same fold. |
| **Subject-independent evaluation** | No person appears in both training and test; the honest way to estimate performance on new people. |
| **Subject leakage** | Test-person data sneaking into training, inflating results; the reason >0.95 accuracy is treated as a bug. |
| **Ablation** | Removing or changing one component to see what it contributed. |
| **Baseline / floor** | The simple method every later model must beat. |
| **1D-CNN** | Convolutional neural network that slides small filters along a time series. |
| **2D-CNN** | Same idea on images (here scalograms/spectrograms). |
| **Scalogram (CWT)** | Image of how signal frequency content changes over time, from a continuous wavelet transform. |
| **Spectrogram** | Image of signal frequency content over time (short-time Fourier transform). |
| **Logistic regression** | Simple classical classifier that weighs input features into a probability. |
| **MLP** | Small fully-connected neural network. |
| **Cross-attention** | Mechanism letting one signal stream "look at" the other and weight which parts matter, so ECG and breathing inform each other. |
| **CANet** | Our Cross-Attention Network: ECG stream + effort stream with bidirectional cross-attention. |
| **Decision-level fusion** | Combining the branches' separate outputs rather than their raw inputs; keeps each interpretable. |
| **Concatenation (concat) fusion** | Simply joining the two streams' features end to end, with no attention. |
| **Parameters** | Learned weights in a model; a rough size measure. |
| **RIP belt (respiratory inductance plethysmography)** | Elastic chest/abdomen band that measures breathing from circumference change; used in MESA's PSG. |
| **Thor / Abdo** | MESA's thoracic (chest) and abdominal belt channels. |
| **SpO2** | Blood-oxygen saturation; falls during many apneas. Not on our wearable. |
| **Desaturation** | A drop in SpO2 (3% or 4% are the common AHI rules). |
| **Arousal** | A brief awakening seen in EEG, often ending an event. |
| **MESA / NSRR** | Multi-Ethnic Study of Atherosclerosis sleep study / National Sleep Research Resource (the gated data repository). |
| **PhysioNet Apnea-ECG** | Public dataset of single-lead overnight ECGs with per-minute apnea labels. |
| **Domain gap / domain shift** | Difference between the data a model was trained on and the data it meets in use. |
| **Transfer failure** | A model working on its training-type data but failing on a new device/population. |
| **Negative control** | A test where the right answer is "nothing" (healthy people should not be flagged). |
| **Flag fraction / flagged fraction** | Share of epochs the model labels as an event. |
| **Spectral entropy / zero-crossings** | How spread-out a signal's frequencies are (low = regular) / how often it crosses zero (more = busier, less regular). |
| **Hilbert envelope** | Smooth outline of a signal's instantaneous amplitude; a collapse = breathing effort stopped. |
| **Motion mask** | A quality gate flagging gross body movement. |
| **Sample rate (Hz)** | Samples per second (PhysioNet 100 Hz, MESA ECG 256 Hz, belts 32 Hz, SensorTile ≈240 Hz). |
| **Breath-hold** | Voluntary pause in breathing; our safe stand-in for apnea. |
