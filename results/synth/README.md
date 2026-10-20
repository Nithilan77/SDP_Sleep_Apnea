# Synthetic data, augmentation and expanded-data experiments (2026-10-20)

Method and parameters: `docs/SYNTHETIC_DATA.md`. Everything here is **simulator output or augmentation of public PhysioNet minutes**, evaluated on **real held-out PhysioNet records** (Track A protocol, leave-one-record-out, 35 folds). Synthetic-only test scores are never reported as clinical performance.

## 1. Datasets generated (`python3 -m src.synth.gen`, `data/synthetic/`, gitignored)
| set | recordings | duration | format | size | valid (required checks) | events |
|---|---|---|---|---|---|---|
| pilot | 20 | 10 min | csv | 281 MB | 20/20 | — |
| medium | 100 | 30 min (50 h) | csv | 4.2 GB | 93/100 | 925 |
| large | 400 | 30 min (200 h) | npz | 2.3 GB | 377/400 | — |
Medium QC (`qc_medium/`): scenarios holds 40 / apnea-like 29 / baseline 31; tiles 67 ECG / 16 EMG / 17 EOG; 100 unique head hashes; nearest-neighbour distance between recordings 0.39–0.72 (standardised feature space; **0 near-duplicates < 0.3**). Failed required checks: beat-count within 8% (5), implausible-RR fraction (2) — noisy profiles degrade the R-peak detector; flagged invalid, excluded from use. Optional checks failing: hold effort drop (7; breathing amplitude too weak vs sensor noise), breathing frequency (4).
**Real-vs-synthetic:** summary statistics are inside the real p5–p95 range for 36–100% of recordings per feature (QVAR SD 70%, wander 91%, breathing-band RMS 100%, accel noise 90%, R amplitude 91%); QRS width is too wide (median 28.8 vs 16.6 ms, KS p < 0.01). **A domain classifier separates real from synthetic 60 s windows with AUROC 0.95** (0.9997 before the 4-LSB QVAR quantisation was added — a measured property the first version missed). The remaining cue is mainly the accelerometer breathing-band amplitude (real windows are burstier). **The synthetic recordings are therefore NOT claimed to be physiologically indistinguishable from real ones**; they are valid for pipeline/format testing, and the experiments below measure whether they help anyway. Figures: `qc_medium/fig_real_vs_synth.png`.

## 2. ECG apnea model with expanded data (`exp_summary.csv`, `exp_paired_vs_baseline.csv`, `fig_ratio_curve.png`)
Same CNN, folds, inner-validation protocol and thresholds as the Experiment-A CNN; only the training set changes. 3 seeds (except 1:10 — seed 0 only). Simulator priors are re-estimated per fold from the fit records only. Pooled per-minute metrics at threshold 0.5; Δ vs `baseline` (this code path: acc 0.753 ± 0.006, AUROC 0.809, sens 0.670, spec 0.805; consistent with Exp. A's CNN 0.753 / 0.807); 95% CI from record-clustered bootstrap of the seed-averaged probabilities.
| config | acc | AUROC | Δacc [CI] | ΔAUROC [CI] |
|---|---|---|---|---|
| baseline (real only) | 0.753 | 0.809 | — | — |
| Aug A (conservative) | 0.758 | 0.799 | +0.003 [−0.016, 0.020] | −0.010 [−0.051, 0.017] |
| Aug B (channel-specific) | 0.762 | 0.814 | +0.003 [−0.018, 0.022] | +0.012 [−0.018, 0.051] |
| best single component (gain) | 0.769 | 0.830 | +0.010 [−0.003, 0.023] | +0.016 [−0.006, 0.034] |
| Synthetic 1:1 | 0.746 | 0.811 | −0.011 [−0.066, 0.064] | +0.004 [−0.044, 0.062] |
| Synthetic 1:2 | 0.715 | 0.794 | −0.045 [−0.106, 0.022] | −0.009 [−0.062, 0.037] |
| Synthetic 1:5 | 0.693 | 0.790 | −0.064 [−0.152, 0.035] | −0.019 [−0.105, 0.062] |
| Synthetic 1:10 (1 seed) | 0.607 | 0.747 | −0.178 [−0.288, −0.074] | −0.087 [−0.184, 0.002] |
| Simulator pretrain → fine-tune on real | 0.753 | 0.790 | −0.004 [−0.020, 0.014] | −0.017 [−0.073, 0.030] |
| Simulator only (train), real test | 0.556 | 0.742 | −0.229 [−0.349, −0.101] | −0.073 [−0.160, 0.009] |
| Combined: Aug B + synthetic 1:1 (chosen by inner-validation AUROC only) | 0.756 | 0.817 | −0.006 [−0.065, 0.057] | +0.006 [−0.046, 0.063] |
| Ensemble of the 3 best-by-inner-validation expanded models (soft average) | 0.762 | 0.827 | — | — |
**Findings.** (1) No expansion strategy improves on real-only training in a statistically supportable way; the best single augmentation (amplitude gain) is +0.010 accuracy, CI includes 0. (2) More simulator data monotonically **degrades** specificity and accuracy (1:1 → 1:10: 0.746 → 0.607); the best useful ratio is ≤ 1:1 and not distinguishable from none. Simulator-only training transfers weakly (AUROC 0.74). (3) Fine-tuning after simulator pretraining does not help. (4) The expanded-data ensemble (0.762 / 0.827) is *lower* than a plain average of three baseline seeds (0.785 / 0.834): its apparent gain over a single model is seed averaging, not extra data. (5) Inner-validation AUROC (real validation records) barely moves (0.870–0.878), so ratio/augmentation selection on training-side data cannot separate these options either. **Recommendation: do not use simulator data for the apnea classifier; keep amplitude-gain augmentation only as an optional robustness measure; the best ECG-only model remains HGB on HRV/CVHR features (0.802 / 0.839, Phase 3).**
**Limits:** the generator's CVHR/amplitude structure is assumption-driven (class-independent amplitude, per-fold priors only for RR/HRV/amplitude scale/gap rate); the generator was adjusted once after viewing aggregate PhysioNet statistics of all 35 records (a small developer-level leak, disclosed; later results do not use it for selection). TCN/GRU/LSTM were not re-run on expanded data because the CNN showed no benefit.

## 3. Other uses tested / not tested
Pipeline testing (done): `tests/test_inhouse_policy_flow.py` validates a generated recording and its determinism; the sync-tap and effort-detector code in `src/breath_hold/` can be driven with synthetic taps/holds (`gen.py --scenario holds` with taps) — a quantitative synthetic stress test of the effort detector was **not** run. Signal-quality classifier and event detector trained on synthetic data were **not** built.
