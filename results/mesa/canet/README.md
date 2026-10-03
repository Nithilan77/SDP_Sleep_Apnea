# CANet on MESA (Track B, steps 2-3)

Same 220 subjects, same grouped 5-fold split, same inner-val/threshold discipline and same 156,826 sleep epochs as
`results/mesa/ecg_baseline/` (asserted in code). Binary target: apnea|hypopnea vs normal. Pooled test epochs.

| model | params | AUROC | AUPRC (chance 0.254) | sens | spec | F1 | AHI r vs a0h3 / a0h4 |
|---|---|---|---|---|---|---|---|
| ECG-only baseline | ~0.1M | 0.609 | 0.335 | 0.622 | 0.535 | 0.416 | 0.31 / 0.30 |
| **CANet (ECG + effort) -- headline** | 169k | **0.780** | **0.490** | 0.822 | 0.613 | 0.555 | 0.74 / 0.70 |
| ablation: effort only | 54k | 0.747 | 0.422 | 0.837 | 0.570 | 0.540 | 0.71 / 0.68 |
| ablation: ECG + effort, concat (no attention) | 102k | 0.744 | 0.424 | 0.819 | 0.582 | 0.538 | 0.72 / 0.69 |
| CEILING: + SpO2 (NOT on our wearable) | 250k | 0.813 | 0.541 | 0.853 | 0.641 | 0.587 | 0.82 / 0.77 |

Headline vs baseline: dAUROC +0.171 [95% CI +0.151, +0.191], dAUPRC +0.155 [+0.132, +0.184] (paired subject bootstrap).
Headline vs concat fusion: dAUROC +0.036 [+0.029, +0.045]; vs effort-only: +0.033 [+0.023, +0.042]
(`headline_vs_ablations.txt`). Full per-class table (obstructive / central / hypopnea sensitivity) in `head_to_head.md`.
Caveats: no SpO2/arousal in the headline model; AHI correlation is an epoch-rate proxy; the cross-attention model has more
parameters than the concat ablation (169k vs 102k), so the attention gain is not parameter-matched.
Each variant subdir has per-fold metrics, confusion matrix, config. Per-epoch predictions (mesaids) are gitignored.
