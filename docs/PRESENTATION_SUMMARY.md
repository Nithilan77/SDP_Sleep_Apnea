# Presentation summary — Cardiorespiratory sleep-apnea screening

*Synthesis of `docs/PROGRESS.md` (§1–§7g, §8–§9), `CLAUDE.md` and the `FINDINGS.md`/`README.md` files under `results/`. Every number below is taken from a documented result and was cross-checked against the saved artifact (source in brackets). Nothing here is new analysis. Where a claim is weak it is stated as weak. Written for a talk; tone and emphasis can be adjusted for the live presentation.*

---

## 1. Goal and the constraint that shapes everything

**Goal.** A wearable *screening* tool (not a diagnostic) that flags people who should get a proper sleep study, using **one chest-worn sensor** (ST SensorTile.box PRO): ECG (the QVAR channel wired to chest electrodes) plus **respiratory effort** from the onboard accelerometer/IMU. The scientific claim to test: **fusing heart + breathing beats either alone.** [CLAUDE.md §1; PROGRESS §1]

**Why it matters.** ~1 in 7 adults have sleep apnea and ~80% of moderate–severe cases are undiagnosed; the only definitive test (overnight polysomnography) is expensive and lab-bound. [CLAUDE.md §2]

**The constraint.** Our volunteers are healthy 19–20-year-olds who essentially do not have apnea. So:
- the models can only be **trained on public, expert-labelled clinical data** (PhysioNet Apnea-ECG, MESA via NSRR);
- our own recordings are **validation only**: healthy-night negative controls and voluntary awake **breath-holds** as safe, labelled apnea-*like* events;
- every headline number is **subject-independent** (never train and test on the same person). [CLAUDE.md §4, §8]
- *One deliberate, scoped exception* (§6 below): linear heads on our own breath-hold labels, advisor-directed, recorded before results existed. [PROGRESS §7f]

---

## 2. What is built and validated

### ECG branch (PhysioNet Apnea-ECG, 35 labelled records, 16,949 labelled minutes, leave-one-subject-out) [results/phase3_cnn/README.md; verified from `per_minute_predictions.csv`]

| Step | Per-minute accuracy | Sensitivity | Specificity |
|---|---|---|---|
| Classical CVHR baseline (7 hand-crafted features, logistic regression) | 0.711 | 0.637 | 0.758 |
| 1D-CNN on RR-intervals | 0.775 | 0.653 | 0.851 |
| **Final: 1D-CNN on RR + R-peak amplitude** | **0.812** | **0.756** | **0.846** |

- *Why it matters:* the classical floor was honest (0.711); a small 2-layer CNN beats it by ~10 points without escalating to anything bigger. An ablation showed the **R-peak-amplitude channel does essentially all the work** (the extra loss weighting alone made things worse). 0.812 is a little below the 85–88% literature target, which we state rather than hide. [PROGRESS §3–4, §6]
- A separate 8-method feature study (time/frequency/nonlinear HRV, QRS-area, wavelet and spectrogram images) found **no method that beat the adopted one per minute**; real per-beat sequences beat summary scalars. [results/feature_study/master_results.md]
- Per-recording accuracy (28/35 = 0.800) is a crude rule and is reported separately, not as a headline. [PROGRESS §3]

### Fusion model "CANet" (MESA, 220 subjects, 156,826 sleep epochs, grouped 5-fold by subject) [results/mesa/canet/README.md; verified from `metrics.json`]

| Model | AUROC | AUPRC | Sens / Spec |
|---|---|---|---|
| ECG-only baseline | 0.609 | 0.335 | 0.62 / 0.54 |
| **CANet (ECG + respiratory effort, cross-attention)** | **0.780** | **0.490** | 0.82 / 0.61 |
| effort only (ablation) | 0.747 | 0.422 | 0.84 / 0.57 |
| ECG+effort, no attention (ablation) | 0.744 | 0.424 | 0.82 / 0.58 |
| + SpO2 (ceiling, **not on our wearable**) | 0.813 | 0.541 | 0.85 / 0.64 |

- *Why it matters:* this is the project's central claim shown on real clinical data — **fusion gives +0.171 AUROC over ECG-only** (paired subject-bootstrap 95% CI +0.151…+0.191). Per-class sensitivity at each model's own threshold: obstructive apnea 0.83 → 0.97, central 0.84 → 0.98, hypopnea 0.56 → 0.78.
- *Honest caveats we state:* most of the gain comes from the effort signal (effort-only is already 0.747); the attention-vs-concat gain (+0.036) is not parameter-matched (169k vs 102k parameters); single seed per fold; the "AHI" correlation is an epoch-rate proxy (r ≈ 0.74/0.70 vs 0.31/0.30 for ECG-only); no SpO2; the MESA effort signal comes from **respiratory belts, not accelerometers**. [PROGRESS §7b]

---

## 3. The central open problem: it does not transfer to our hardware as-is

Running the MESA-trained model on our own SensorTile recordings (10 usable healthy nights, 0.8–7.5 h each) works end to end, **but the effort stream does not transfer**. [PROGRESS §7c; results/inhouse_canet/]

- A simple classifier tells MESA **belt** epochs from our **accelerometer** epochs with **AUROC 0.997** on the encoder's embedding (0.99 on hand-crafted features; control belt-vs-belt 0.51 / 0.47).
- Healthy volunteers are flagged on **~94%** of epochs by the fused model (0.92–0.94), vs a MESA low-AHI median of 0.28. Effort-only gives 0.80–0.84; replacing our effort with white noise drops it to 0.33 — so it is the *structured-but-wrong-morphology* accelerometer trace being read as disrupted breathing.
- Breathing **is** in our accelerometer (spectral peak near 0.2–0.25 Hz) but the signal is broader-band and less regular than a belt (spectral entropy 0.64–0.66 vs 0.46).
- Separately, the ECG-only model transfers partly: the RR-interval channel transfers (mean RR 1.07 s vs 0.91 s, physiology), but R-peak amplitude is ~**108×** off (uncalibrated electrostatic sensor), so the frozen 2-channel model predicts an implausible 52.8% apnea rate on healthy recordings (21.3% with the amplitude channel neutralised). [PROGRESS §7]

This is the problem the rest of the work responds to.

---

## 4. What we tried to close the gap (and what each attempt taught us)

All four attempts were **label-free** (our healthy data has no labels) and used 10 in-house nights (8 train / 2 held-out for the adaptors). [PROGRESS §7d; results/adaptation/*/FINDINGS.md]

| Attempt | Why it made sense | Result | What it narrowed down |
|---|---|---|---|
| Embedding-gap study | Understand the gap before fixing it | Mean shift small (0.16 SD) but separability 0.997; accelerometer embeddings collapse into a corner of the belt space | Gap is in *manifold structure*, not an offset — a shift/scale fix won't work |
| CORAL, pooled embedding | Cheap closed-form covariance alignment | Domain AUROC 0.994 → 0.218 (alignment worked) but flag rate 93% → 100% | Aligned the wrong thing: bypassed cross-attention, fed the head an input it never saw |
| CORAL, token level | Same idea in the model's real input space | Domain AUROC 0.954 → 0.934; flag rate ~96% (no gain) | Gap is *nonlinear temporal structure*, not covariance |
| Adversarial adaptor v1 (hidden 128) | Nonlinear, trainable | Unstable training, no credible flag-rate gain (median 88% vs ~93–97%) | Too little data for adversarial training |
| Adversarial adaptor v2 (hidden 32, stronger anchor) | Constrain capacity | Stable but inert: held-out AUROC 0.9984 → 0.9981; flag rate 86–92% | Nothing to learn from 2,411 unlabelled epochs |

**Takeaway.** Four distinct mechanisms, linear and nonlinear, loosely and tightly constrained, all gave null or negative results — evidence that representation-space manipulation **without labels** cannot close the belt-to-accelerometer gap at this data scale, and that **a labelled supervisory signal from our own hardware** was the missing ingredient. That is what the breath-hold session supplies.

> **Correction to carry into the talk.** §7d originally cited adaptor v1's "held-out domain AUROC 1.0, worse than baseline" as proof of overfitting. That number came from fitting and scoring a classifier on the *same* data (the resubstitution bug found and fixed for v2, where it gave 1.0 even for a no-op adaptor). **Do not quote the v1 AUROC.** The unstable training and absence of a credible flag-rate gain still stand. (Recorded as an erratum under §7d Step 3 and in §7g, committed separately as `33c586c`.)

---

## 5. The breath-hold session: first labelled events from our hardware

One volunteer, one ~30 min session, chest SensorTile, **7 voluntary ~30 s holds** (stopwatch laps), 4.8 s lag between stopwatch total and recording. [PROGRESS §7e; results/breath_hold/FINDINGS.md]

**What we are confident about.**
- **Accelerometer effort visibly collapses during holds.** Under the conservative inner window (nominal inset by the 4.8 s lag) the effort ratio (hold RMS ÷ pre-hold RMS) is **<0.8 in 7/7 events and <0.5 in 5/7** (mean 0.35). A quiet stretch is visible in every hold window of the session overview plot. This is the first labelled evidence that our raw accelerometer carries cessation information — a counterpoint to §7c, which showed the *belt-trained representation* doesn't transfer, not that the signal is empty.

**What we are not confident about — stated as such.**
- **Effort ratios are a range, not a point value** (per event: inner → wide window): 0.25–0.62, 0.27–0.31, 0.49–0.64, 0.14–0.52, 0.46–0.78, 0.10–0.33, 0.60–1.49. (Event 7's 1.49 is the recovery-gasp bleed into the stopwatch window, not increased effort.)
- **Automatic boundary detection did not generalise.** A bug was found and fixed (the detector ran on the *signed* bandpassed signal and counted any 5 samples ≈ 20 ms near zero as "quiet", so every start landed at the ±12 s search edge). Three principled detectors were then tried; each is correct on only 3/7 events (2, 4, 6) by visual check. We stopped tuning rather than overfit one session.
- **HR direction is unconfirmed.** The sign of the hold-minus-pre HR change flips with the choice of boundaries (+0.84 / −3.08 / −1.35 bpm across three versions). No bradycardia claim is made.
- One subject, one session; no short-hold false-positive test was recorded.

---

## 6. The transfer-learning experiment (advisor's suggestion, §7f)

**Setup.** Freeze the MESA-trained CANet; train only a **linear (L2-regularised logistic regression) head** on its pooled features, using the 7 labelled holds; compare with a linear head on **8 hand-crafted effort features**; leave-one-**event**-out CV with a leakage purge; baseline period and healthy nights held out. [PROGRESS §7f; results/transfer/results.json]

**Result — plainly.** The frozen MESA features **do not demonstrably beat** the hand-crafted baseline at this data scale:

| | Pooled AUROC (95% CI) | Worst-event AUROC | False-positive rate on baseline period |
|---|---|---|---|
| Frozen CANet features | 0.832 (0.58–1.00) | 0.16 | 24% |
| Hand-crafted (8 features) | 0.826 (0.71–0.97) | 0.78 | 0.5% |

- They **tie** on pooled AUROC; the hand-crafted head is **more consistent per event** (frozen fails Events 5 and 7: 0.16 and 0.69). A shuffled-label control gives chance for both (0.48 / 0.45), so the pipeline is not leaking labels — **but that rules out leakage, not small-sample instability** (the frozen CI spans 0.58–1.00). The supportable claim is *"no demonstrated benefit of the frozen features, and worse consistency,"* not a proven ranking.
- **Not apples-to-apples:** the frozen features see a 150 s window (cardiac + effort); the hand-crafted ones only a 30 s effort epoch. With weaker regularisation hand-crafted pulls clearly ahead (0.92 at C = 1 vs 0.82); the tie holds only at the pre-specified C = 0.01.
- n = 7 events, one subject — a **pilot, not a validated accuracy**; voluntary awake holds are not apnea.

**Why this is still a meaningful finding.** It is a clean, honestly-bounded answer to "does pretrained transfer help at this scale?" (not yet), it identifies *where* it fails (events with atypical effort), and it was obtained without leakage by design. Simple effort statistics already carry most of the hold signal on our hardware.

**The concrete deliverable: an end-to-end deployed pipeline.** `python3 -m src.deploy.predict_session <recording>` takes a raw SensorTile file and outputs per-window hold-likelihood from **both** heads side by side, with a plot. Heads were trained on all 7 holds, so output on that same session is **in-sample** (labelled as such on the console and on the image). The only out-of-sample check is the 10 healthy nights: **3.1% (frozen) and 4.6% (hand-crafted)** of windows flagged (sleep vs awake-supine is a different state; unlabelled; specificity-only). [results/transfer/healthy_nights_check.csv]

---

## 7. Why not an LSTM/RNN? (§7g, a reasoned decision, not a gap)

- **CANet is already temporal:** multi-scale 1D convolutions (kernels 3/7/15) plus bidirectional cross-attention over **75 tokens spanning 150 s** (≈2 s each), trained on 156,826 epochs. [src/mesa/canet.py; PROGRESS §7g]
- **The open problem is sensor-domain transfer** (belt → accelerometer), not missing temporal modelling; a recurrent layer does not fix a distribution shift.
- **7 independent events cannot train or evaluate a recurrent model:** the existing pooled-AUROC CIs are ±0.21 (frozen) and ±0.13 (hand-crafted), so a plausible model difference could not be detected; a poor or a good result would both be uninterpretable. A tiny LSTM *could* have fewer parameters than our failed adaptors (≈350–1,300 vs 4.2k–16.6k), so the argument rests on **sample size, not parameter count**.
- **What would justify it:** ≈30+ independent labelled holds (≈4× today, to roughly halve the CI), ≥10 subjects for subject-independent evaluation, repeat sessions with sync taps and a validated boundary detector, and a baseline to beat (the hand-crafted head's per-fold minimum). *These are scaling estimates derived from the documented CI widths, not proven thresholds.* (Open to the advisor's input; PROGRESS §7g.)

---

## 8. What runs today, and what you can credibly claim

**Runs end to end now (own hardware, own data):**
- SensorTile recording (old `.txt` or new `.csv` firmware format) → R-peaks + accelerometer breathing trace → frozen MESA CANet → two linear heads → per-window hold probability + plot (`src/deploy/predict_session.py`).
- The frozen PhysioNet ECG model (RR + amplitude, 0.812/0.756/0.846 LOSO) also exists as a deployable artifact — but it **does not transfer to QVAR as-is** (amplitude scale), per §7.

**Credible claims**
1. On public data the fusion approach works: CANet AUROC 0.780 vs 0.609 ECG-only, subject-independent, with CIs.
2. The ECG-only branch reaches 0.812/0.756/0.846 per minute (LOSO), above a documented classical floor.
3. On our own hardware, the pipeline runs end to end, and accelerometer effort visibly drops during voluntary holds (ratio <0.8 in 7/7 events under the conservative window).
4. Rigorous negative results: four label-free adaptation attempts failed; pretrained features do not demonstrably beat hand-crafted features at n = 7 events.

**Do not claim**
- Any apnea accuracy on our hardware (healthy volunteers; voluntary holds are not apnea).
- That fusion beats ECG-only *on our hardware* (shown on MESA belts only).
- That the transfer-learning heads "work" as a detector in general: pilot, one subject/session, in-sample demo, 24% baseline false positives for the frozen head.
- A bradycardia/HR signature, precise hold boundaries, or a validated cessation detector.
- Anything about hypopnea without SpO2 (named limitation of the wearable).

**Known limitations to say out loud:** one subject / one session / 7 events; boundary detector unresolved; **temporal drift in the session not ruled out** as a contributor to baseline false positives; the Event 7 failure (up to 27% zero-padded input) is unresolved as padding-vs-event-specific; MESA is an elderly cohort (mean ~68) vs our 19–20-year-olds.

---

## 9. Next steps

1. **Second (and third) breath-hold session(s), more subjects:** 3 sharp sync taps / event markers, the 5–7 s SHORT-hold false-positive test, holds spread across the recording with a rest-only control stretch (addresses drift), and a rest-period comparison. → validates a boundary detector on held-out events.
2. **Subject-independent evaluation** (≥10 subjects) once enough events exist.
3. With a larger labelled in-house set: **fine-tune the effort encoder** (not just a head) and revisit adaptation with labels; a temporal model trained on our own data becomes viable around the sizes in §7 above.
4. **QVAR-specific recalibration of the ECG amplitude channel**, and an ECG-domain signal-quality gate (IMU motion gating cannot see electrical artifacts). [PROGRESS §7]
5. **Phases 6–7** on our hardware: decision-level fusion and a per-night report (events/hour, severity, no-SpO2 caveat) — blocked on a validated effort branch.
6. Housekeeping already done (separate commits): §7f false-positive correction (`0679da6`), §7d erratum (`33c586c`), README/CLAUDE.md status notes (`5ebf3e3`). `docs/presentation_handoff/*` (the 2026-10-04 mentor-review deck material) still predates §7d–§7g and should not be reused as-is.
