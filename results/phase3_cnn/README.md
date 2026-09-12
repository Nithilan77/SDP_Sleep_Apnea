# Phase 3 — ECG-branch CNN: final result

**Status: closed.** Adopted config: **2-channel input (RR-interval + R-peak
amplitude), default class-balanced loss weighting.** No further tuning planned
for this branch — next step is Phase 4 (effort/IMU branch).

## Progression

All numbers are per-minute, subject-independent (35-fold leave-one-subject-out,
i.e. each fold trains on 34 records and tests on the 1 held-out record), on the
exact same 16,949-minute set across all 35 labelled Apnea-ECG records
(a01-a20, b01-b05, c01-c10).

| Step | Input | Loss weighting | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Phase 2 classical baseline | hand-crafted CVHR/HRV features (logistic regression) | balanced | 0.711 | 0.637 | 0.758 |
| Phase 3 v1 CNN | RR-interval sequence only (1 channel) | balanced | 0.775 | 0.653 | 0.851 |
| Phase 3 v2 CNN | RR-interval + R-amplitude (2 channels) | balanced × 1.5 (sensitivity-boosted) | 0.798 | 0.754 | 0.825 |
| **Phase 3 FINAL — Variant A** | **RR-interval + R-amplitude (2 channels)** | **balanced (default, ×1.0)** | **0.812** | **0.756** | **0.846** |

Record-level (apnea group a/b vs control group c, naive >5%-of-minutes-flagged
rule): 28/35 = 0.800 for the final model — same as v1 and v2. This per-recording
metric doesn't move much across variants; see the Phase 2 report for why (heavy
overlap between apnea/control recordings under this crude threshold, not a
per-variant weakness).

## The ablation finding (why v2's extra weighting was dropped)

v2 combined two changes at once (2-channel input + 1.5x sensitivity-boosted
loss) and beat v1, so the natural question was which change actually did the
work. An ablation isolated them:

| Variant | Accuracy | Sensitivity | Specificity |
|---|---|---|---|
| v1 (1ch, default weight) | 0.775 | 0.653 | 0.851 |
| **A: 2ch, default weight** | **0.812** | **0.756** | **0.846** |
| B: 1ch, 1.5x weight | 0.758 | 0.715 | 0.786 |
| v2: 2ch, 1.5x weight | 0.798 | 0.754 | 0.825 |

**The R-amplitude channel (variant A) drives essentially the entire
sensitivity gain by itself, and does so with a smaller specificity cost than
v2.** The 1.5x loss weighting on its own (variant B) is a net loss versus v1
(accuracy -0.017) — it raises sensitivity a little but at a much larger
specificity cost. Layering the weighting on top of the amplitude channel (v2)
doesn't compound the gains: v2 is worse than variant A on *every* metric.
Conclusion: keep the amplitude channel, drop the extra loss weighting.

Full ablation predictions: `results/phase3_cnn_ablation/`.

## What's in this folder

- `final_per_minute_predictions.csv` — the adopted result (variant A), one row
  per (record, minute): `y_true`, `y_pred`, `y_prob`.
- `final_per_recording_predictions.csv` — record-level rollup of the above.
- `per_minute_predictions.csv`, `per_recording_predictions.csv`,
  `sequences.npz` — **historical v1** (1-channel RRI-only) run; kept for the
  progression table above, not the final result.

Earlier development trail (not final, kept for provenance): `results/phase3_cnn_v2/`
(the 2ch + 1.5x-weight run) and `results/phase3_cnn_ablation/` (variants A and B).

## Reproducing the final result

`src/ecg/cnn_apnea_final.py` — 2-channel input via `src/ecg/rr_amp_sequence.py`,
`pos_weight_factor=1.0` in `src/ecg/cnn_model.py`'s `train_cnn`, 30 epochs/fold,
same LOSO split as Phase 2 (`src/eval/splits.py`). Not re-run after adoption
(the numbers above are the already-validated ablation run, "variant A") — the
script exists so the result can be regenerated if the pipeline code changes
later.

Note: this is a cross-validation evaluation, not a deployable checkpoint —
each of the 35 LOSO folds trains its own model from scratch and none of the
35 sets of weights is saved. A single deployable model (trained on all 35
records with no held-out subject) would need a separate training run and is
out of scope for Phase 3, whose job was to establish the method and the
honest accuracy number, not ship a model.

## Non-negotiables observed

- Subject-independent (LOSO) throughout — no subject ever in both train and test.
- Trained only on PhysioNet Apnea-ECG; no data from our own recordings.
- 0.812 accuracy checked against the >0.95 leakage red flag — well clear of it.
