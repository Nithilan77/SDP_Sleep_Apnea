# Breath-hold recording protocol (Phase 5) — Monday session checklist

## Why we are doing this
Collect **timestamped voluntary breath-holds** (awake, supine) as safe, perfectly-labelled apnea-*like* events, to answer one
question: **can the accelerometer effort signal on our hardware detect cessation of breathing at all?**
After the in-house CANet study (`docs/PROGRESS.md` §7c) this is the critical question: the belt-trained effort stream does not transfer
to our accelerometer (belt vs accelerometer 0.99-separable; healthy flag rate ~94%). A breath-hold is quiet, not noisy — it should show
up as an effort-envelope collapse. If it doesn't on a well-placed device, the accelerometer is the problem, not the model.

This is validation only: the recordings are **never used for training** and are **never committed** (`data/recordings/` is gitignored).
Healthy volunteers, voluntary holds: this yields a plausibility/detectability result, **not** an apnea-accuracy number.

## Before the session
- [ ] SensorTile.box PRO + STEVAL-MKI242A charged; QVAR logging to ECG-style channel + IMU at 240 Hz (same setup as the earlier recordings).
- [ ] Fresh electrodes (3 pads; **dried gel = noisy ECG**), alcohol wipes, razor, tape / chest strap.
- [ ] A clock/phone visible to the logger for exact clock times; the timestamp log table below (print or copy it).
- [ ] Consent form signed (reuse the prior team's form); volunteer screened (see Safety).

## Electrode placement (3-tab QVAR cable, Lead-II)
Prep every site: **alcohol-wipe, let dry, shave** if hairy, then fresh electrode.
| Tab | Role | Site |
|---|---|---|
| 1 | negative | right chest, just below the collarbone |
| 2 | positive | left lower ribcage, below the heart |
| 3 | reference | right lower ribcage / hip |

## SensorTile placement — the main variable to get right
**Strap the box FIRMLY and FLAT to the mid-sternum** (chest strap or tape; box face parallel to the chest, not tilted, not on soft fabric)
so the IMU follows chest-wall motion at its maximum. A loose, pocketed or dangling box gives weak effort — **exactly the failure mode we
just found**. Note in the log how it was attached (strap/tape), orientation (cable direction), and the posture (supine, arms by sides).

## Procedure (~12–15 min per subject)
1. Subject supine, relaxed, arms at sides. **Start the recording; write down the clock time.**
2. **Sync marker:** within the first 10 s give **3 sharp taps** on the box. Log the clock time of the taps.
3. **2 min quiet breathing** (baseline).
4. **5–6 breath-holds of 15–25 s each:** breathe out normally, then hold (comfortable, never strained). Call "start"/"stop" so the logger
   records the exact clock time of each. **90 s of normal breathing between holds.**
5. Include **ONE deliberate short hold of 5–7 s** that should **NOT** be detected (false-positive test); log it as `SHORT`.
6. **2 min quiet breathing** (recovery baseline).
7. Stop the recording; log the stop clock time.

## Safety
- Voluntary and short holds only; the subject can stop **at any time**, no coaching to push on.
- **Skip anyone with a respiratory or cardiac condition** (asthma, COPD, arrhythmia, etc.) or who feels unwell/dizzy.
- Stop the session immediately for dizziness, chest discomfort or distress.

## Timestamp log (this is the ground truth)
| Item | Clock time (hh:mm:ss) | Notes |
|---|---|---|
| Session start (recording started) | | file name: |
| Sync taps (3 taps) | | |
| Hold 1 start / end | / | |
| Hold 2 start / end | / | |
| Hold 3 start / end | / | |
| Hold 4 start / end | / | |
| Hold 5 start / end | / | |
| Hold 6 start / end (if done) | / | |
| SHORT hold (5–7 s) start / end | / | should NOT be detected |
| Session stop | | |

Also record: subject code (no names in anything committed), device placement/attachment, posture, any movement/cough/talking events.

## Output needed
1. The **.txt** SensorTile recording (to `data/recordings/`, never committed).
2. The **timestamp log table** above. The sync taps align the clock log with the device's microsecond timestamp.

## Scope
**One clean subject validates the protocol; expand to 3–5 subjects the same week for a reportable result.**
