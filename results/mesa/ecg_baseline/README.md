# MESA ECG-only baseline (Track B, step 1)

RR + R-peak amplitude (2 Hz grid, 150 s window = target 30 s epoch +/-2) -> 1D-CNN -> P(apnea|hypopnea) per
30 s sleep epoch. **Subject-independent**: grouped 5-fold CV over 220 subjects (150/26/44 train/inner-val/test per
fold; early stopping + Youden threshold from inner-val subjects only). Weighted BCE. Sleep epochs only,
156,826 epochs (4.6% of sleep epochs dropped for <10 valid beats), prevalence 25.4%.

| | sens | spec | AUROC | AUPRC (chance 0.254) | F1 |
|---|---|---|---|---|---|
| pooled test epochs | 0.622 | 0.535 | 0.610 | 0.335 | 0.416 |
| per-fold mean +/- sd | 0.62+/-0.08 | 0.53+/-0.04 | 0.610+/-0.030 | 0.339+/-0.055 | 0.41+/-0.04 |

Subject-level (n=220): predicted event-epochs/sleep-h vs NSRR a0h3 r=0.31, a0h4 r=0.30 (Spearman 0.31/0.31).
Reference: the true event-epoch rate vs a0h3/a0h4 gives r=0.69/0.67, so this is the ceiling of the epoch-rate proxy.
No SpO2 is used, so desaturation-based AHI definitions cannot be matched.

Files: `fold_metrics.csv`, `metrics.json`, `confusion_matrix.csv`, `ahi_correlation.json`, `ahi_points_anon.csv`
(no ids), `config.json`, `run.log`. Per-epoch predictions (carry mesaids) are in gitignored `data/mesa/cache/`.
