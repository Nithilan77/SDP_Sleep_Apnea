# Phase 3 — ensembles over the saved leave-one-record-out OOF predictions (2026-10-20)

Code: `src/ensemble/run_ensemble.py` (built by a parallel session that was stopped; reviewed, re-run-verified and adopted) and `src/ensemble/report.py`.
Base predictions are reused from `results/expA_multiseed/` (CNN, TCN, GRU, LSTM, HGB, logreg; 3 seeds; 35 records, 16,949 minutes; identical population
asserted in code). Second level = **LOSO over OOF**: weights, Platt calibrators, stacking coefficients and the (optional) threshold are fitted only on the other 34
records' OOF predictions and applied frozen to the held-out record. **Caveat (named, not removable without 35×34 nested retraining):** a training record's base prediction came
from a model that had seen the held-out record, so there is a weak 1-step indirect influence; ≤ 7 second-level parameters keep it small. Results are "OOF-stacked LOSO", not fully nested.
Re-verification: 42 stored per-seed rows recomputed from the saved predictions, 0 mismatches.

| model (thr 0.5, mean of 3 seeds) | acc | AUROC | Δacc vs HGB [95% CI, record-bootstrap] | ΔAUROC vs HGB [95% CI] |
|---|---|---|---|---|
| HGB (strongest single) | 0.802 | 0.839 | — | — |
| CNN | 0.753 | 0.807 | −0.049 [−0.075, −0.025] | −0.031 [−0.067, 0.008] |
| **primary (pre-designated) cnn+tcn+hgb soft** | 0.802 | 0.844 | 0.000 [−0.016, 0.015] | +0.005 [−0.012, 0.022] |
| cnn+tcn+hgb weighted | 0.807 | 0.841 | +0.005 [−0.006, 0.015] | +0.002 [−0.010, 0.014] |
| cnn+tcn+hgb+logreg soft (best AUROC; selection-optimistic) | 0.807 | 0.850 | +0.005 [−0.008, 0.019] | +0.011 [−0.009, 0.032] |
| cnn+tcn+hgb stacking | 0.802 | 0.827 | 0.000 [−0.016, 0.013] | −0.012 [−0.034, 0.008] |
(all 40 combination×strategy rows: `summary_mean_sd.csv`; per-seed: `per_seed_metrics.csv`; `paired_vs_best_single.csv`; `error_overlap_seed0.csv`; `per_record_primary_vs_hgb.csv`.)

**Findings.** (1) No ensemble is distinguishable from the HGB alone: every paired CI contains 0; the best point gains (+0.005 acc, +0.011 AUROC) are inside the noise
and come from the best-of-40 selection. (2) Per record the primary ensemble beats HGB by >1 point on 13 and loses on 8 of 35 records — not consistent. (3) Adding GRU/LSTM
lowers every metric (individually the weakest, errors ≥ 60% shared with other models); CNN/TCN add no information HGB lacks beyond ~0.005 AUROC. (4) Calibrated averaging buys specificity
(0.90) at the cost of sensitivity (0.64): a threshold trade, not better discrimination. (5) Cost: ensemble latency = sum of members (CNN 2.7 + TCN 23.5 + HGB 9 ms per 1000 windows, 1 CPU thread, `latency_cpu.csv`).
**Recommendation: keep the single HGB as the best ECG-only model; an ensemble is not justified by this evidence.** Calibrated CNN+HGB is the only variant worth keeping as an option if lower false-positive rate is wanted.
**MESA (`src/ensemble/mesa_ensemble.py`, `mesa_ensemble_metrics.csv`, `mesa_ensemble_paired.csv`).** Saved subject-grouped OOF predictions of all models exist on the same 220 subjects / 156,826 epochs (identity asserted). Second level = 10-fold grouped CV over subjects. Wearable-deployable (no SpO2): the best single is the ECG+effort CANet (AUROC 0.780); ECG+CANet+resp soft average 0.773 (ΔAUROC −0.007, CI [−0.013, −0.002] — *worse*), stacking 0.783 (Δ ≈ 0, CI [−0.001, 0.000]) with sensitivity 0.85 / specificity 0.59 (a threshold shift). With SpO2 (not on our hardware): ceiling+ecg_spo2+headline stacking 0.819 vs ceiling 0.813 (Δ +0.006, CI [0.003, 0.009]; subject-bootstrap, 300 reps) — statistically positive but clinically negligible. **No MESA ensemble is recommended; the earlier statement that none could be built was wrong and is superseded.**
