# PROJECT BRIEF — Cardiorespiratory Sleep Apnea Detection

This is the complete context for the project. Read this fully before doing anything.
It explains what we're building, why, the constraints, the architecture, and exactly
where we are so far.

---

## 1. The one-paragraph summary

We are building a **wearable sleep apnea screening system** that detects apnea (breathing
pauses during sleep) by fusing two signals from a single chest-worn sensor: **ECG** (heart
electrical activity) and **respiratory effort** (chest movement from a motion sensor). It is
a screening/triage tool, not a diagnostic device — it flags people who should get a proper
clinical sleep study. The scientific core is that fusing heart + breathing signals beats
using either alone.

## 2. Why this project exists (motivation)

- Sleep apnea affects ~1 in 7 adults; ~80% of moderate–severe cases are undiagnosed.
- The only definitive test is overnight polysomnography (PSG) — expensive, uncomfortable,
  lab-bound, long waitlists.
- A cheap wearable screen closes the gap between "has apnea" and "gets diagnosed."
- Apnea is a **cardiorespiratory** event. ECG-only detection tops out ~85–88% and cannot
  distinguish apnea *types*. Adding respiratory effort improves both.

## 3. Who we are / context

- Student project (2 students: Nithilan + Guru), academic, with an industry collaboration
  (ST Microelectronics). Guided by faculty (Dr. A. Shahina, Dr. K. S. Gayathri).
- Part of a larger sleep-analysis programme ("Pinnacle"). A sister team (Priyan, Harikaran)
  does **EEG-free sleep staging** — a DIFFERENT track, not our scope. Our track is apnea.
- The seniors' prior work (Pinnacle Review-I) built a CNN-LSTM sleep-stage classifier on
  EEG/EOG/EMG (Sleep-EDF dataset). We reuse their fusion approach and evaluation discipline.

## 4. THE CORE CONSTRAINT (this shapes everything — read twice)

**Our own test subjects are healthy 19–20 year olds. They essentially do not have apnea.**

Consequences:
- **Training data must be public, expert-labelled clinical data** — that's the only place
  real apnea exists for us.
- **Our own recordings are for VALIDATION ONLY. Never train on them.**
- We validate using **voluntary breath-holds** (awake, timestamped) as safe, perfectly-
  labelled apnea-like events, plus healthy overnight recordings as negative controls.
- Any experiment that trains on our own recordings is wrong by design.

> **Scoped exception (recorded 2026-10-09 in `docs/PROGRESS.md` §7f, before any result existed; advisor-directed):** linear (L2 logistic-regression) heads on *frozen* MESA-trained CANet features and on hand-crafted effort features were fitted on the 7 labelled holds of one breath-hold session. No encoder/attention weights and no deep network are trained on in-house data; the ECG and MESA branches are unchanged. See the §7f paragraph for the full scope and why.

## 5. Hardware

- **ST SensorTile.box PRO** + **STEVAL-MKI242A** adapter. We have 5 units.
- The adapter exposes **QVAR** (an electrostatic sensor channel) + a dry-electrode analog
  front-end. We wire QVAR to chest electrodes to record **ECG** (Lead-II placement).
- The onboard **IMU** (accelerometer + gyro) captures **chest-wall motion** = respiratory
  effort, for free in the same recording.
- One SensorTile records ONE biosignal channel (QVAR) at a time. For this project QVAR is
  always ECG. The IMU runs alongside regardless.
- **Our QVAR samples at 240 Hz.** (PhysioNet training data is 100 Hz — a mismatch we must
  reconcile; see §8.)
- Also in the ecosystem: Muse S Athena EEG headband (used by the staging track, not ours).

## 6. Data sources

| Source | Signals | Rate | Role |
|---|---|---|---|
| **PhysioNet Apnea-ECG** | single-lead ECG + per-minute apnea labels, 70 recordings | 100 Hz | **TRAIN** the ECG model |
| **Our SensorTile recordings** | QVAR-ECG + IMU (accel/gyro/temp) | 240 Hz | **VALIDATE** only |
| **MESA (via NSRR, gated)** | ECG + effort belts + SpO2 + EEG + event-type labels, ~2000 subj | varies | Track B (stretch) |

PhysioNet Apnea-ECG record naming:
- `a01`–`a20` = apnea group, `b01`–`b05` = borderline, `c01`–`c10` = control
  (these 35 have `.apn` per-minute labels = our labelled learning set)
- `x01`–`x35` = withheld test recordings
- Each record: `.dat` (ECG signal), `.hea` (header), `.apn` (per-minute A/N labels),
  `.qrs` (precomputed R-peaks)

**IMPORTANT — do not confuse datasets:** Sleep-EDF (folders named sleep-cassette,
sleep-telemetry, SC-subjects, ST-subjects) is a DIFFERENT dataset for sleep STAGING —
that's the sister team's. Ours is Apnea-ECG (files a01.dat, c01.apn, etc.).

**There is no public apnea dataset with IMU data.** This is why the effort branch is
signal-processing (no training needed), not a trained model. MESA's respiratory effort
belts are the closest labelled proxy for chest motion (Track B).

## 7. Architecture — two branches, decision-level fusion

```
                       ┌─ ECG BRANCH (trained on PhysioNet) ──────────────────┐
QVAR/ECG ─bandpass─▶ R-peaks (Pan–Tompkins) ─▶ RR-interval series
                       ─▶ CVHR features ─▶ [classical baseline | 1D-CNN] ─▶ P(apnea)/min
                                                                                  │
                       ┌─ EFFORT BRANCH (IMU, signal processing only) ─────┐      ▼
IMU ─▶ accel magnitude ─▶ motion-mask ─▶ bandpass 0.08–0.6Hz               ─▶ DECISION FUSION
    ─▶ Hilbert envelope ─▶ breathing rate + effort amplitude               ─▶      │
                                                                                  ▼
                                                  confident event / flagged / rejected
                                                                                  ▼
                                            per-night report: events, AHI-like rate, severity
```

**Why decision-level fusion (not one big end-to-end model):** the two branches train on
different data (ECG on PhysioNet; effort on nothing/MESA), and decision-level keeps each
branch interpretable — we can always explain *why* an event was flagged.

**Key terms:**
- **CVHR** (Cyclic Variation of Heart Rate): during apnea HR drops (bradycardia), then
  surges on breathing resumption (tachycardia). The cardiac fingerprint of apnea.
- **RR-interval:** time between consecutive R-peaks (heartbeats). Time-based, so transfers
  across sample rates once rates are known.
- **Epoch:** analysis window. Apnea-ECG labels are **per-minute** — match that.
- **AHI:** Apnea-Hypopnea Index = events/hour. <5 normal, 5–15 mild, 15–30 moderate, >30 severe.
- **Obstructive** apnea = effort continues (blocked airway). **Central** = effort stops.
  Only the effort/IMU signal distinguishes them.

## 8. Non-negotiable methodology rules

1. **Subject-independent evaluation always.** Report leave-one-subject-out / subject-wise
   splits. Mixed-subject splits inflate accuracy badly (literature: 100% per-recording vs
   ~86% subject-independent). Never headline the inflated number.
2. **Report per-minute AND per-recording separately** — different tasks.
3. **Beat the baseline or justify it.** Classical CVHR detector is the floor; the CNN must
   beat it or it adds nothing.
4. **Start simple:** classical CVHR → 1D-CNN → (only if plateaued) CNN-LSTM. No Transformer
   to start.
5. **Never train on our own recordings.**
6. **Reconcile sample rates** before feature extraction: PhysioNet 100 Hz vs QVAR 240 Hz.
   Prefer working in TIME units (seconds) for RR features so rate differences cancel;
   resample to a common grid for the CNN.
7. **Treat suspiciously perfect numbers as a bug** (e.g. 100%, perfect separation) — check
   for subject leakage between train and test FIRST.
8. **Verify true sample rate empirically** on our own recordings (samples ÷ duration from
   the microsecond timestamp) — never trust the nominal 240 Hz.

## 9. Reference performance (literature targets, not promises)

| Method | Score | Role |
|---|---|---|
| CVHR / ACAT (Hayano 2011) | r=0.91 vs AHI; 83% sens / 88% spec | baseline target |
| 1D-CNN single-lead (Chang 2020) | 87.9% per-minute, 97.1% per-recording | primary model target |
| CNN-Transformer-LSTM (2025) | 91.6–94.1% per-segment | escalation ceiling only |
| Chest accelerometer (Ryser 2022) | 76% sens / 70% spec cessation detection | effort branch realism |

A from-scratch CVHR baseline will likely land ~72–80% per-minute untuned. That's the
honest floor, not a failure.

## 10. SensorTile file format (for when we ingest our own recordings)

Tab-delimited `.txt`, 16 columns:
```
Timestamp [us]  A_X[LSB] A_Y[LSB] A_Z[LSB]  A_X[mg] A_Y[mg] A_Z[mg]
G_X[LSB] G_Y[LSB] G_Z[LSB]  G_X[dps] G_Y[dps] G_Z[dps]  T[LSB] T[degC]  QVAR[LSB]
```
Use the physical-unit columns (mg, dps, degC), not raw LSB. QVAR is the single biosignal
channel (wired as ECG). Prior team has a txt→CSV→EDF pipeline (pyEDFlib) we can adapt.

## 11. Environment

> **Update 2026-10-09:** the environment below describes the original Windows laptop and is out of date. Work is now on a Linux machine (Python 3.12.3, NVIDIA RTX 5000 Ada GPU). The system `python3` has torch 2.13.0+cu130 and is used for everything that imports torch; the project `venv/` has numpy/scipy/pandas/matplotlib/scikit-learn/neurokit2/mne but no torch. `requirements.txt` pins different versions from those installed. See `docs/HANDOFF_2026-10-09.md` §8.1.

- **Currently: Windows laptop, Python 3.11.9, CPU only** (GPU lab machine temporarily
  unavailable — timeline unknown). Everything through the CVHR baseline needs no GPU;
  a small 1D-CNN trains on CPU (slower but fine). When the GPU machine returns, copy the
  whole folder over and continue there for heavy training.
- venv active. Installed: numpy, scipy, pandas, matplotlib, scikit-learn, wfdb, neurokit2,
  torch (CPU), tqdm.
- Project root: `D:\Nithilan\SEM 4\Pinnacle - Sleep Pattern Analysis\apnea`
- Use RELATIVE paths (`data/physionet`) and run scripts from the project root.

## 12. Repo layout

```
apnea/
  data/physionet/     # Apnea-ECG (downloaded)
  data/recordings/    # our own SensorTile recordings (later; never commit)
  src/ingest/         # physionet loader; later: sensortile txt→edf, sync
  src/ecg/            # rpeaks, rr, cvhr features, models
  src/effort/         # IMU envelope, effort amplitude
  src/fusion/         # decision-level combiner
  src/eval/           # metrics, subject-wise splits, plots
  notebooks/          # exploration only
  results/            # metrics + figures, one subdir per experiment
```

## 13. WHERE WE ARE NOW (status)

> **Update 2026-10-09:** this section is a historical snapshot from project start and is out of date: all `src/` folders now contain code and results exist for Phases 0-5 and the MESA/transfer tracks. Current status: `docs/PROGRESS.md` (§8 table) and the complete record in `docs/HANDOFF_2026-10-09.md`. The original text below is kept unchanged.

DONE:
- Environment set up (Windows laptop, Python 3.11.9, venv active, packages installed:
  numpy, scipy, pandas, matplotlib, scikit-learn, wfdb, neurokit2, torch-CPU, tqdm).
- PhysioNet Apnea-ECG downloading (or downloaded) to data/physionet.

NOT YET DONE ON THIS MACHINE:
- All src/ folders are EMPTY. No code exists on this Windows laptop yet.
- The loader, R-peak, and feature code were written/verified earlier on a different
  (Linux) machine and must be RECREATED here from scratch, using relative Windows paths.
- Nothing has been run on this machine yet.

WHAT WAS PROVEN EARLIER (on the Linux machine — reproduce, don't assume):
- Loader worked: 100 Hz confirmed; a01 = 470 apnea / 19 normal minutes; c01 = 0 / 484.
- R-peak detection clean: a01 ~30k beats @ 62.6 bpm, 0% implausible; c01 similar.
- These are expected results to reproduce, NOT existing files.

IMMEDIATE NEXT STEPS (in order):
1. Confirm data/physionet download finished (expect a01.dat, c01.apn, etc. — NOT
   sleep-cassette/telemetry folders, which would be the wrong dataset).
2. Recreate src/ingest/physionet.py (loader), run it, confirm it matches the expected
   numbers above.
3. Recreate src/ecg/rpeaks.py, run it, confirm clean R-peaks.
4. Create src/ecg/features.py (per-minute CVHR features), run it, and confirm the CVHR
   signal is real: apnea minutes should show LARGER mean HR-swing than normal minutes
   on a01. That is the green light for Phase 2 modelling.

## 14. PLAN — phases in order (each is a checkpoint; report before moving on)

- **Phase 0 — Environment.** DONE.
- **Phase 1 — Ingestion.** PhysioNet loader + R-peaks DONE. (Own-recording ingestion later.)
- **Phase 2 — CVHR baseline.** Per-minute features → classical CVHR detector → first
  accuracy number (~75–80% expected). Report per-minute + per-recording, subject-independent.
- **Phase 3 — 1D-CNN.** RR-interval CNN, target ~85–88% per-minute. Must beat Phase 2.
  Subject-disjoint validation.
- **Phase 4 — Effort branch.** IMU → envelope → cessation detection. Signal-processing only,
  no training. (Needs our own recordings or synthetic test data.)
- **Phase 5 — Breath-hold validation.** First result on our own hardware: does the pipeline
  catch timestamped breath-holds? (Needs recordings + device sync.)
- **Phase 6 — Decision-level fusion.** Show fused > ECG-only (the headline claim). Plus
  healthy-night negative control.
- **Phase 7 — Per-night report.** Aggregate to events/hour + severity band. Name the no-SpO2
  hypopnea limitation in the output.
- **Phase 8 — Track B (MESA).** After NSRR access: multimodal + SpO2 + obstructive/central.

Phases 2–3 are pure software — do them now on the laptop. Phases 4–7 need our own
recordings (hardware), which run in parallel once the SensorTile sessions happen.

## 15. Ethics / data handling

Human-subject recordings. Never commit raw recordings or subject identifiers to git.
Consent forms exist from prior team work — reuse/adapt. Keep identifiers out of results
and figures.

## 16. How to work

Work phase by phase. At the end of each phase, STOP and report: what was produced, the
numbers (with split type stated), anything surprising, and what the next phase needs. Do
not silently proceed past a failed check. Prefer an honest modest result with a clear
caveat over an impressive unexplained one.
