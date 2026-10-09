# Breath-hold recording protocol — v2 (second session; approved 2026-10-09)

> **Status.** Approved 2026-10-09 (hold set, order A, rest rule, 3 s boundary tolerance, 2 s window inset, building the sync tooling). This replaces the v1 checklist (v1 is preserved in git history, commit `38c3b55`). Nothing has been recorded yet. One consequence of the approved 2 s inset is flagged for decision in §9.3. Every change from v1 is tied to a specific problem observed in session 1 (`docs/PROGRESS.md` §7e, §7f). The analysis plan (§9) is written **before any data exists** so that it cannot be tuned to the results.

## 0. Why we are doing this (unchanged purpose)
Collect **timestamped voluntary breath-holds** (awake, supine) as safe, labelled apnea-*like* events, to answer: **can the accelerometer effort signal on our hardware detect cessation of breathing, how precisely can we time it, and does the deployed pipeline work on data it has never seen?** This is **validation only** (healthy volunteers, voluntary holds): it yields a plausibility/detectability result, **not** an apnea-accuracy number. Recordings are never committed (`data/recordings/` is gitignored); no names in anything committed (subject codes only: `P01`, `P02`, …).

## 1. What session 1 taught us → what v2 changes

| # | Problem in session 1 | Evidence | v2 change |
|---|---|---|---|
| 1 | Event times known only to ~5 s: stopwatch total 1800.46 s vs recording 1795.6 s (**4.8 s lag**), so hold boundaries needed a 4.8 s inset and automatic refinement (§7e) failed on 4/7 events | `results/breath_hold/summary.json`, §7e | **Two sharp taps** on the box right after the stopwatch starts (and again at the end) → a measurable anchor in the accelerometer; lap times are then converted to recording time by the *detected* tap, not by a guessed lag (§5, §9.1–9.2) |
| 2 | One subject, one session → no subject-independent statement is possible | §7f limitation 4 | **Script written for several subjects** with identical order/calls/kit; subject code, nothing else changes (§7) |
| 3 | Event 7's 150 s model input was up to 27% zero-padded (recording ended 52 s after Hold 7) | §7f, `results/transfer/window_index.csv` (`pad_frac`) | **≥ 150 s of quiet buffer after the last hold** before the end taps, then 30 s more before stopping (§5) |
| 4 | All holds ≈ 30 s → no dose-response and no short-hold false-positive test (the planned SHORT hold was never recorded) | §7e limits, v1 protocol | **Holds of 5, 10, 30, 45 and 60 s** (two 5 s, two 10 s, three 30 s, one 45 s, one 60 s) (§5) |
| 5 | Rests of ~90 s after ~30 s holds (3:1); a longer hold needs proportionally longer recovery | session 1 laps | **Rest = max(90 s, 3 × the preceding hold)**, so always ≥ the hold and the same 3:1 ratio for ≥ 30 s holds (§5) |
| 6 | Unusual events (cough, strap shift, swallow mid-hold — Event 3's ~3.5 s burst) were unexplained after the fact | §3.2 of the handoff | **Light-weight time-stamped annotation sheet** with 6 one-letter codes (§6) |
| + | The §7f temporal-drift question: baseline false positives came from the *start* of the session, training windows from the *end* | §7f limitation 1 | **Quiet rest-only stretches at both ends** (4 min at the start, 2.5 min at the end) and holds spread across the recording (§5, §9.5) |
| + | The deployed heads were trained on session 1, so session 1 can never test them | §7f deployment | Session 2 is the **first genuinely out-of-sample test** of the deployed heads, run unchanged (§9.4) |

## 2. Roles and kit
**People:** *Subject* (supine, follows the calls) and *Logger* (stopwatch, laps, calls, annotation sheet). The Logger must not also be the Subject. Optional second helper for the pilot tap check.

**Kit (per subject):**
- [ ] SensorTile.box PRO + STEVAL-MKI242A, charged; logging **QVAR (ECG-style) + IMU at 240 Hz**, new `.csv` format (`src/ingest/sensortile.py` reads it); only the **chest box** is needed (EMG/EOG boxes are not used — if they record anyway, they are not analysed).
- [ ] Chest strap or tape; fresh electrodes (3 pads; **dried gel = noisy ECG**), alcohol wipes, razor.
- [ ] Phone/stopwatch with **LAP** (shows total time and lap split); a second clock showing wall time.
- [ ] Printed **protocol card** (§5 table) and **annotation sheet** (§6), pen.
- [ ] Consent form signed (reuse the prior team's form); volunteer screened (§8). Subject code assigned (`P01`…).
- [ ] Mat/bed, blanket; quiet room; phone silenced.

## 3. Placement (unchanged from v1 — this is what makes effort visible)
**Electrodes (3-tab QVAR cable, Lead-II).** Prep every site: alcohol-wipe, let dry, shave if hairy, fresh electrode.

| Tab | Role | Site |
|---|---|---|
| 1 | negative | right chest, just below the collarbone |
| 2 | positive | left lower ribcage, below the heart |
| 3 | reference | right lower ribcage / hip |

**SensorTile.** Strap **firmly and flat to the mid-sternum** (box face parallel to the chest, not tilted, not on soft fabric) so the IMU follows chest-wall motion. Write down: strap vs tape, cable direction, posture (supine, arms by sides), and take a **photo** of the placement (no face). Use the **same placement for every subject and session**. Note the top-face of the box: that is where the taps go (§4).

## 4. Pre-session tap check — MANDATORY checklist (5 min; do not start the real recording until it is ticked)
**Why:** session 1 contains no taps, so the real strength of a tap on a strapped box is unknown. The detector only accepts a double-tap that is *firm* (§9.1), so the taps must be calibrated on the actual subject, strap and box, every session.

**Pass criteria, fixed in advance (all must hold):** **3 trial double-taps**, at least 15 s apart; **each of the 6 taps ≥ 60 σ above the noise floor** (σ = robust noise of the high-passed accelerometer, printed by the tool); **each tap ≥ 2× any other transient within ±10 s**. Target (not required): ≥ 100 σ.

Checklist — tick each item (initials + time on the card):
- [ ] 1. Box strapped exactly as in §3 (same strap, same sternum position), subject supine and still.
- [ ] 2. Start a **separate short test recording** on the box (this file is *not* part of the analysis). Write the wall-clock time.
- [ ] 3. **Wait 10 s hands-off** (no touching, no talking).
- [ ] 4. **Trial 1:** two sharp, short taps ≈ 0.5 s apart with a **fingertip/knuckle on the box's top face** — firm, like knocking on a door; not a push, not a slap on the chest. Then **wait ≥ 20 s** hands-off. (The accelerometer saturates above ±2 g, so do not hit it hard enough to hurt; a soft touch will be invisible.)
- [ ] 5. **Trial 2:** same double-tap. Wait ≥ 20 s hands-off.
- [ ] 6. **Trial 3:** same double-tap. Wait 10 s. Stop the test recording; copy the file off the box.
- [ ] 7. Run: **`python3 -m src.breath_hold.sync --check <test_file.csv>`** (the project `venv` also works; no torch needed).
- [ ] 8. Read the output. Required: **`TAP CHECK PASSED: 3 accepted trial double-tap(s)`** with each tap's sigma ≥ 60 and dominance ≥ 2×. Write the three trial sigmas on the card.
- [ ] 9. **If PASSED:** tick this box, keep the same strap, same finger, same firmness for the real taps, and go to §5.
- [ ] 10. **If FAILED:** change one thing (tap firmer, tap nearer the box edge, tighten the strap), delete nothing, and repeat items 2–8 (**up to 3 attempts**). Do not weaken the criteria.
- [ ] 11. **If still FAILED after 3 attempts:** follow the fallback in §10 (**STOPWATCH-ONLY sync**) — write "SYNC: STOPWATCH-ONLY" at the top of the card **before** the session starts.

What the tool has been validated on (details in §9.1): synthetic taps injected into the real session-1 noise are accepted ≈ 90% of the time at ~0.5 g nominal peak, ≈ 99–100% at ≥ 0.6–0.8 g, and ≈ 0% at ≤ 0.2 g — i.e. the tap must be a **firm knock**; the check output tells you the real strength in σ.

## 5. Procedure — the exact schedule
All times are on the **protocol clock `P`**, where **P = 0:00 is the second sync tap**. The stopwatch is started *before* the taps, so the Logger converts: **target stopwatch time = P + (stopwatch reading at Lap 1)**. Fill the right-hand column of the card during the first 4-minute baseline (it takes ~1 minute; do the arithmetic once, write it down, do not recompute during holds).

**Calls (identical every time, calm voice):**
- ~15 s before a hold: *"Next hold in fifteen seconds. Breathe normally."*
- At the hold start (press LAP on the word): *"Breathe out normally … and hold."*
- At the hold end (press LAP on the word): *"Breathe."*
- Hold **after a normal exhale** (as in session 1) — **no deep breath in, no pre-hyperventilation.**
- Subject may stop **at any time**: raise a hand; Logger presses LAP, writes `A` (abort) and the reason (§6). Analysis uses the **actual** lap duration, not the planned one.

### 5.1 Start (≈ 0:40 before P = 0)
1. Place/strap the box, check placement and electrode contact. Subject lies still.
2. **Start the recording on the box.** Write the wall-clock time.
3. **Wait 10 s completely still** — no hands on the box, no talking (lets the start-up/handling transients die away; in session 1 an unexplained double-transient at ≈ 46 s, soon after the start, is exactly what a detector must not mistake for taps).
4. Logger **starts the stopwatch**, immediately says *"Tap, tap."* and the **subject (or Logger, same person every time) gives the double-tap** (§4 style). Logger **presses LAP on the second tap** → **Lap 1 = SYNC (P = 0:00)**. Write the stopwatch reading at Lap 1.
5. Both stay still, breathing normally. Write the sync time on the card.

### 5.2 Schedule (order A — use this order for **every** subject)
Rest after a hold = **max(90 s, 3 × hold)**. Holds ≥ 30 s are never first; the longest hold comes after two ≥ 30 s holds.

| Item | What | P start | P end | Duration | Rest after | Laps | Target stopwatch (fill in) |
|---|---|---|---|---|---|---|---|
| Lap 1 | **SYNC taps** (2nd tap) | 0:00 | | | | 1 | |
| Baseline | quiet breathing (rest-only control, start) | 0:00 | 4:00 | 240 s | — | — | |
| H1 | hold | 4:00 | 4:30 | 30 s | 90 s | 2 (start), 3 (end) | |
| H2 | hold | 6:00 | 6:10 | 10 s | 90 s | 4, 5 | |
| H3 | hold | 7:40 | 8:25 | 45 s | 135 s | 6, 7 | |
| H4 | hold | 10:40 | 10:45 | 5 s | 90 s | 8, 9 | |
| H5 | hold | 12:15 | 12:45 | 30 s | 90 s | 10, 11 | |
| H6 | hold | 14:15 | 15:15 | 60 s | 180 s | 12, 13 | |
| H7 | hold | 18:15 | 18:25 | 10 s | 90 s | 14, 15 | |
| H8 | hold | 19:55 | 20:00 | 5 s | 90 s | 16, 17 | |
| H9 | hold | 21:30 | 22:00 | 30 s | — | 18, 19 | |
| Buffer | quiet breathing, **no holds** (rest-only control, end; ≥ 150 s so no input is zero-padded) | 22:00 | 24:30 | 150 s | — | — | |
| End taps | **second double-tap** (press LAP on the 2nd tap) | 24:30 | | | | 20 | |
| Stop | 30 s still, then **stop the recording** | 24:30 | 25:00 | 30 s | | | |

Totals: 9 holds (2 × 5 s, 2 × 10 s, 3 × 30 s, 1 × 45 s, 1 × 60 s; 225 s of holding), protocol time 25:00, 20 laps. Rest/hold ratios: 3:1 for ≥ 30 s holds, 9:1 and 18:1 for the short ones (their recovery is dominated by the 90 s floor, which also gives a ≥ 60 s clean pre-hold baseline). Allow ~45 min per subject including setup, the tap check (§4), and teardown.
*Optional repeat session (not required):* order B = the same holds in **reverse** order, to separate order/fatigue effects from duration effects.

### 5.3 After
Stop the recording; write the wall-clock stop time and the stopwatch total. Remove electrodes, check skin. Copy the recording off the box that day; **do not commit it**.

## 6. Real-time annotation (lightweight; never interrupt the protocol)
**One printed sheet, one pen, held by the Logger.** Columns: `stopwatch (mm:ss)` | `code` | `hold #` | `note`. Read the time off the stopwatch display *without pressing anything* (never stop the stopwatch; the LAP button is for protocol laps only).

| Code | Meaning |
|---|---|
| **C** | cough / sneeze / swallow noticed |
| **S** | strap/box shift, electrode peel, cable tug |
| **P** | change of position (turned, lifted head, moved arms) |
| **T** | talking / laughing / noise in the room |
| **A** | hold **aborted** early (write why) |
| **O** | anything else unusual |

Rules: (a) the Logger writes **a code and a time only** — details afterwards; (b) **during a hold the Logger writes nothing and says nothing** except the end call (anything seen is jotted in the next rest, with the hold # and "during hold"); (c) the Subject does **not** speak during holds; during rests the Subject may report one word ("swallowed", "itch") which the Logger logs with the last lap time; (d) the sheet's times are cross-checked against detected transients afterwards (§9.6), never used to *edit* the signal. If it is unclear whether something is worth logging, log it with `O`.

## 7. More than one subject — keeping it repeatable
- **Identical script, kit, placement, order A, calls and card for every subject**; only the subject code and the filled-in stopwatch column change. Same Logger and same room if at all possible.
- **Target 3 subjects, if available by the session date** (the §7g scaling estimates need ≥ 10 subjects for subject-independent evaluation — session 2 is a step toward that, not a substitute). Fewer subjects is acceptable and is reported as such. Subject 1 may be the same volunteer as session 1 (within-subject, different day) — record it as such; the other subjects are cross-subject.
- **The Logger is always someone other than the subject** (the person running the session keeps the stopwatch, laps, calls and annotation sheet), so hold timing and logging never compete for one person's attention. **Every session is dated at the time of recording** (date written on the card and in the file name).
- Per-subject checklist (tick on the card): consent ☐ screening ☐ placement + photo ☐ tap check passed ☐ recording started ☐ 10 s still ☐ sync taps (Lap 1) ☐ 9 holds logged ☐ buffer ☐ end taps (Lap 20) ☐ stop ☐ annotation sheet scanned ☐ file copied ☐.
- Data naming: `P01_session2_<yyyymmdd>.csv`, `P01_session2_laps.csv`, `P01_session2_annotations.csv` (names/ids never in the file names).

## 8. Safety
- Voluntary, short holds; the subject can stop **at any time**; **no coaching to push on**. The 45 s and 60 s holds are optional per subject: they are *attempted*, the planned duration is a ceiling, an early stop is logged as `A`, and aborting is a valid result.
- **Skip anyone with a respiratory or cardiac condition** (asthma, COPD, arrhythmia, etc.), who is pregnant, or who feels unwell/dizzy. No pre-hyperventilation, no deep breath in before holds (reduces the risk of light-headedness).
- Stop the session immediately for dizziness, chest discomfort, tingling or distress; resume only if the subject wants to. A second person should be in the room.
- Voluntary holds are **not** apnea and the recording is **not** a medical test; say so in the consent briefing.

## 9. Analysis plan (pre-specified; written before any session-2 data exists)

### 9.1 Detecting the sync taps automatically (and what we have verified about it)
**Important:** the effort pipeline's band is **0.08–0.6 Hz** (`src/effort/envelope.py`). A tap is a ~10 ms transient that this band **removes completely**. Detection therefore runs on a *separate* high-frequency path:
1. 3-axis accelerometer (g) → subtract mean → **4th-order Butterworth high-pass at 10 Hz, zero-phase** → vector magnitude → robust z-score (median/MAD over the whole recording).
2. Candidate taps = isolated sharp peaks: z ≥ 25, peak width (FWHM) ≤ 60 ms.
3. A **sync marker** = a pair of candidates 0.25–1.2 s apart, amplitude ratio ≤ 3×, **isolated** (no third transient ≥ 25 within 0.6 s either side), and **dominant**: both taps ≥ 60 σ **and** ≥ 2× the largest other transient within ±10 s.
4. **Where to look:** start pair in the first 60 s of the recording; end pair in the last 120 s. Tap time = the second tap's peak sample (that is where Lap 1 / Lap 20 were pressed).
5. **QVAR cross-check (secondary only):** report the QVAR high-passed response at the tap times; not required for acceptance.

**What has been verified about the final detector** (`src/breath_hold/sync.py`; validation gate `src/breath_hold/sync_validate.py`, full output `results/breath_hold/sync_validation/validation.md` and `detection_table.csv`; all on session 1's real noise, which contains no deliberate taps):
- Accelerometer: **243.10 Hz**, ±2 g full scale (0.061 mg/LSB), 1.7% repeated samples. High-passed noise robust σ = **0.0023 g** (z = 25 → 0.058 g, z = 60 → 0.14 g, z = 100 → 0.23 g in the high-passed magnitude). The signal regularly reaches z ≈ 12–19 (z > 12 for 1.6% of the time; probably heartbeat vibration, not verified), so a bare threshold is not enough. Natural transients are common: 40 sharp peaks above z = 25 in 30 min (peak up to 0.44 g at ≈ 293 s), clustered shortly after the start (≈ 46 s), ≈ 290 s, ≈ 687 s, ≈ 1075 s and ≈ 1664–1669 s. None were annotated, so their causes are guesses — hence §6.
- **False alarms (final logic): 0 accepted markers anywhere in session 1** — 13 candidate pairs were seen and each was rejected for a stated reason (weak < 60σ, not dominant < 2×, not isolated, amplitude ratio); in particular none in the first 60 s or last 120 s. (A simpler earlier rule — isolated double-transient at z ≥ 25 — gave 3 false doubles at ≈ 46, 1075 and 1664 s; the 60σ + 2× dominance gate is what removes them. An even earlier check with a z > 12 isolation threshold reported 0 false doubles vacuously.)
- **Injection test of the FINAL logic** (synthetic double-taps, 40–90 Hz ring-down, τ = 6 ms, random 3-D direction and sub-sample phase, 200 per cell, injected at quiet sites of the real recording in the start window [10–30 s] and end window [last 110–10 s]); "accepted" = passes every rule including the ≥ 60σ and ≥ 2× gates:

| nominal peak (g) | candidate pair seen (start / end) | **accepted marker (start / end)** | median weaker-tap σ (accepted) |
|---|---|---|---|
| ≤ 0.08 | 0% / 0% | **0% / 0%** | — |
| 0.12 | 9% / 8% | **0% / 0%** | — |
| 0.20 | 88% / 80% | **0% / 0%** | — |
| 0.30 | 100% / 100% | **4% / 12%** | 70 / 65 |
| 0.40 | 100% / 100% | **46% / 63%** | 76 / 71 |
| 0.50 | 100% / 100% | **90% / 90%** | 85 / 80 |
| 0.60 | 100% / 100% | **97% / 99%** | 97 / 99 |
| ≥ 0.80 | 100% / 100% | **100% / 100%** | 127–199 |

  **Timing:** detected second-tap time minus the injected tap's own largest-magnitude sample: median |error| 0.0 ms, 95th percentile 0.0 ms, mean bias ≤ +0.1 ms (the detector lands on the same sample, i.e. resolution ±1 sample = ±4.1 ms). *Caveat:* my synthetic taps have their largest sample at the first sample, so this mostly shows the detector does not shift the peak; real ring-down may move it by a sample or two. Either way it is negligible beside the Logger's lap-press latency (~0.2–0.5 s, unmeasured).
- **This final logic is much stricter than the prototype** used to design the plan (which accepted 86% of 0.2 g taps): the 60σ + 2× dominance gate is what removes the false alarms, and what makes the real tap strength matter. **Pre-session tap-check pass rate in simulation** (3 trial double-taps): 0% at ≤ 0.3 g, 22% at 0.4 g, 76% at 0.5 g, 94% at 0.6 g, 97% at 0.8 g, 93% at 1.2 g (the residual failures are trials placed beside real large transients). **Stress test:** taps injected 6 s before the real z = 64 transient are accepted 0% at 0.2–0.4 g and 56% at 0.8 g (dominance rule working as designed — so keep the 10 s before the real taps still).
- **The simulation proves nothing about real taps**, only what strength is needed; the pilot check (§4) measures real strength in σ.
- **Lap alignment self-test** (synthetic laps with known truth, `validation.md`): recovered lap times within 5–10 ms with no clock drift and with 2–3 s drift (start+end interpolation); with 0.2 s drift the start-only rule (|drift| < 0.5 s) leaves up to 0.2 s error by design. End-to-end CLI runs on a synthetic tapped recording reproduced all 20 lap times to ≤ 10.3 ms (tap modes) and ≤ 6.6 ms (stopwatch-only mode, which also stamps every output `STOPWATCH-ONLY`).

### 9.2 Aligning the lap table to the recording
- `t_rec(tap)` is the detected second-tap time (start). Each lap's recording time = `t_rec(tap) + (lap_total − lap1_total)`. This replaces the session-1 guess of a 4.8 s global lag.
- **Drift/clock check:** the end pair gives a second anchor. Report `(t_rec(end tap) − t_rec(start tap)) − (lap20_total − lap1_total)`; if |drift| < 0.5 s, use the start anchor only; otherwise interpolate linearly between the two anchors. (Decision rule fixed now.)
- **Residual timing error** is now dominated by the Logger's button-press reaction (~0.2–0.5 s, *unmeasured*) and the Subject's start latency, not by a multi-second lag. We will *measure* it afterwards (observed effort-collapse onset minus lap start, per hold) and report the distribution, without using it to adjust anything.

### 9.3 Windows (fixed in advance)
- **Primary hold window** = lap start/end converted to recording time, **inset by 2 s at each end** (to cover onset latency and avoid the recovery gasp). Sensitivity: inset 0 s and 5 s. The wide and inner windows of session 1 are retired.
- **Flagged consequence of the approved 2 s inset (decision needed before the first recording):** with 2 s trimmed at each end, a **5 s hold has a 1 s primary window** and a 10 s hold a 6 s window; one second is shorter than a breath cycle (≈ 3–4 s at 15–20 breaths/min), so an effort ratio over it is not interpretable. In STOPWATCH-ONLY mode (5 s inset) the 10 s holds have no window at all. The tool computes insets of 0, 2 and 5 s for every hold and flags `primary_window_too_short` (< 3 s). **Proposed rule, pending approval:** for holds whose primary window is < 3 s, report the **0 s-inset** window as the value used for the dose-response comparison, labelled *short-hold (sub-breath-cycle window)*, and report the 2 s-inset value alongside without interpreting it. Until approved, both are reported and neither is claimed.
- **Pre-hold baseline** = the 15 s ending **5 s before** the hold start (so it cannot contain the pre-hold inhale or the first hold seconds).
- **Rest windows** and the **rest-only controls** (baseline 0:00–4:00 and buffer 22:00–24:30) are defined from the lap table the same way; windows for the deployed heads use the §7f construction (150 s input, 30 s centre epoch, 5 s stride, label thresholds unchanged).

### 9.4 Pre-specified analyses (what we will report, whatever the outcome)
1. **Effort ratio per hold** (hold RMS ÷ pre-hold RMS, 0.08–0.6 Hz bandpassed magnitude), per subject and pooled, vs **hold duration** (5/10/30/45/60 s): report every value and the within-subject rank correlation of ratio with duration; the **hypothesis is not assumed** (ratios may be flat in duration; the 5 s holds may sit near 1).
2. **Short-hold false-positive test:** run `src/effort/cessation.py::detect_cessations` unchanged (≥ 10 s minimum, 30% of rolling baseline): report whether the two 5 s holds are flagged (expected not) and whether the 10 s holds are (borderline by construction).
3. **Boundary detector validation on held-out events:** run the **unchanged** `src/breath_hold/analyze_session.py` detector (v3a/v3b, commit `1b0b674`) on the new holds *before* looking at the plots, and report the fraction whose refined start and end are within **3 s** of the lap-derived boundary, by hold duration. *(The 3 s tolerance is proposed here; the team should approve or change it before the first recording.)* No detector changes before reporting.
4. **Out-of-sample test of the deployed heads:** run `python3 -m src.deploy.predict_session` (frozen CANet head A and hand-crafted head B, trained on session 1, `results/transfer/deployed_heads.joblib`) **unchanged** on each session-2 file; report pooled and per-hold AUROC and sens/spec at 0.5 for the §7f window labels, and the false-positive rate on the two rest-only stretches. The same-volunteer session is within-subject/different-day; other subjects are cross-subject. No retraining before this result is reported; only afterwards may heads be refit (and then with leave-one-session/-subject-out).
5. **Drift control:** compare the heads' false-positive rate on the **start** rest-only stretch (0:00–4:00) with the **end** stretch (22:00–24:30), and on rest windows in between — directly addresses the §7f temporal-drift limitation.
6. **Annotation cross-check:** list annotated events (C/S/P/T/A/O) against detected high-frequency transients and the motion mask; windows overlapping an annotated event are flagged (not silently dropped) and results are reported with and without them.
7. **HR during holds:** per hold, hold-minus-pre mean HR on the primary window, by duration; **no direction claim** unless it is consistent in sign across ≥ 3 subjects (stated now so the result cannot be spun). Report whether the Event-1-style two-beat R-peak artifact recurs.

### 9.5 What we will not do
No tuning of detector, heads or windows on session-2 data before the pre-specified results are reported; no pooling of subjects into one "accuracy"; no claim beyond voluntary holds; if taps fail (§10) the session is analysed in the session-1 style and labelled as such.

## 10. Risks and fallbacks

### 10.1 IF THE TAP CHECK FAILS → STOPWATCH-ONLY sync (explicit instruction)
Follow in order:
1. Complete the retries in §4 (up to 3 attempts). If it still fails, **the session proceeds on STOPWATCH-ONLY sync** — do not weaken the pass criteria and do not skip the session on this account.
2. **Before the session starts**, write **"SYNC: STOPWATCH-ONLY"** in large letters at the top of the protocol card and the annotation sheet.
3. Run the session exactly as in §5 **including the start and end double-taps and Laps 1 and 20** (the taps may still be recorded and may be detectable; the fallback is about not *relying* on them). **Also record the stopwatch total at the instant the recording is stopped** (hand-over item 5) — the stopwatch-only alignment needs it.
4. Name the files `…_SWONLY_…` (e.g. `P01_session2_SWONLY_<yyyymmdd>.csv`).
5. Analyse with `python3 -m src.breath_hold.sync --align <file> --laps <laps.csv> --stopwatch-only --stop-total <seconds>`. The tool maps lap totals to recording time by `lag = stopwatch total at stop − recording duration` (the session-1 method, uncertain by several seconds), uses the **5 s inset**, and writes `sync_mode = STOPWATCH-ONLY` and a `sync_label` column into every output row (`aligned_windows.csv`, `lap_times_recording.csv`). If no start marker can be found at analysis time the tool falls back to this mode automatically and says so.
6. **Labelling is mandatory downstream**, the same way in-sample/out-of-sample is labelled in `docs/PROGRESS.md` §7f's artifacts: every plot title, CSV and PROGRESS/FINDINGS table that uses this session's windows must carry the text **"STOPWATCH-ONLY sync (boundaries uncertain by several seconds)"**; these results are **never pooled with tap-synchronised sessions without the label**; and the boundary-detector validation of §9.4-3 is **not claimed** for such a session (its reference boundaries are themselves uncertain by seconds).
7. Record in the session notes why the check failed (so the next session can fix it).

### 10.2 Other risks
| Risk | Fallback |
|---|---|
| Two natural transients mimic the taps | Dominance rule (§9.1) + the end pair + the first tap is expected ~10–30 s after the recording starts and must lie in the first 60 s; if two dominant pairs remain, use the one consistent with the Logger's start→Lap 1 interval; if ambiguous, use the end pair and report the ambiguity. |
| Subject cannot hold 45 or 60 s | Log `A`, analyse the actual duration; do not repeat on the day. |
| Strap/electrode shifts mid-session | Log `S`; re-secure during a rest only; do **not** restart the recording (a restart breaks the lap alignment). |
| Box battery/storage | Fully charge, confirm free space, test-record 30 s beforehand. |
| Lap pressed late/missed | Annotate `O`; the hold's boundaries then come from the effort signal only and are flagged. |
| File-format surprise | Verify `load_sensortile` on the tap-check file (it is read by the tap check itself) before the real session. |

## 11. Hand-over after each subject
1. The recording `.csv` (never committed). 2. `…_laps.csv` with columns **`lap,total`** (stopwatch TOTAL at each lap press as seconds or mm:ss.xx; 20 rows, Lap 1 = sync; optional extra column `call`) — the format `sync.py` reads. 3. `…_annotations.csv` (the sheet). 4. Placement photo. 5. Stopwatch total and wall-clock start/stop. 6. Tap-check result. 7. Subject code, whether same volunteer as session 1.

## 12. Decisions
**Approved 2026-10-09:** (1) hold set, order A and the rest rule; (2) 3 s boundary-agreement tolerance (§9.4-3) and 2 s window inset (§9.3); (3) building the sync tooling — **built**: `src/breath_hold/sync.py` (`--check`, `--detect`, `--align`) and `src/breath_hold/sync_validate.py` (validation gate, results in `results/breath_hold/sync_validation/`); (4) target 3 subjects if available, the Logger is never the subject, sessions dated at recording.

**Still to decide before the first recording:** (a) the short-hold window rule in §9.3 (use the 0 s-inset window for holds whose 2 s-inset window is < 3 s, labelled as such); (b) the session date, the Logger(s) and the subject list; (c) whether the first subject is the session-1 volunteer.
