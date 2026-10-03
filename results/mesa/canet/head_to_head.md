# CANet vs ECG-only baseline -- identical epochs, subject-independent grouped 5-fold CV

n_epochs=156826, n_subjects=220, event prevalence=0.254. Each model uses its own inner-val Youden threshold per fold. Deltas are vs the baseline; CI = paired bootstrap over subjects (200 reps).

| model                                         |   auroc |   auprc |   sens_event |   spec_normal |    f1 |   sens_obstructive |   sens_central |   sens_hypopnea |   d_auroc | d_auroc_ci95     |   d_auprc | d_auprc_ci95     |
|:----------------------------------------------|--------:|--------:|-------------:|--------------:|------:|-------------------:|---------------:|----------------:|----------:|:-----------------|----------:|:-----------------|
| ECG-only baseline (CardiacCNN)                |   0.609 |   0.335 |        0.622 |         0.535 | 0.416 |              0.829 |          0.840 |           0.564 |   nan     |                  |   nan     |                  |
| CANet (ECG + effort)  [HEADLINE]              |   0.780 |   0.490 |        0.822 |         0.613 | 0.555 |              0.969 |          0.981 |           0.780 |     0.171 | [+0.151, +0.191] |     0.155 | [+0.132, +0.184] |
| CEILING: CANet + SpO2 (not on our wearable)   |   0.813 |   0.541 |        0.853 |         0.641 | 0.587 |              0.914 |          0.945 |           0.835 |     0.204 | [+0.183, +0.224] |     0.206 | [+0.181, +0.227] |
| ablation: effort only                         |   0.747 |   0.422 |        0.837 |         0.570 | 0.540 |              0.987 |          1.000 |           0.795 |     0.138 | [+0.116, +0.158] |     0.087 | [+0.063, +0.116] |
| ablation: ECG + effort, concat (no attention) |   0.744 |   0.424 |        0.819 |         0.582 | 0.538 |              0.982 |          0.994 |           0.773 |     0.136 | [+0.113, +0.154] |     0.089 | [+0.066, +0.114] |
