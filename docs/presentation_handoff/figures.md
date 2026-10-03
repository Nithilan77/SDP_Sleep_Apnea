# Figures

All PNGs are in `figures/`. Contain aggregate numbers only: no subject identifiers (in-house nights are anonymised S03…S15, MESA subjects are never labelled), no raw data. Figures marked **generated** are produced by `make_figures.py` from the results CSVs (run from project root); **copied** are existing project figures, unmodified.

| File | Source | Caption |
|---|---|---|
| `phase3_progression.png` | generated (prediction CSVs, recomputed) | ECG branch, per-minute LOSO on PhysioNet: classical 0.711 → v1 CNN 0.775 → v2 0.798 → final 0.812 accuracy (sensitivity 0.637 → 0.756). |
| `feature_study_comparison.png` | generated (`results/feature_study/master_results.csv`) | Eight ECG feature representations under identical LOSO protocol; RR + R-amplitude (0.812) is best, scalar HRV banks sit near the 0.711 floor. |
| `canet_vs_baseline.png` | generated (`results/mesa/canet/head_to_head.csv`) | MESA head-to-head: ECG-only AUROC 0.609 / AUPRC 0.335 vs CANet 0.780 / 0.490, with effort-only, concat and SpO2-ceiling ablations. |
| `per_class_gain.png` | generated (`head_to_head.csv`) | Per-class sensitivity, ECG-only → CANet: obstructive apnea 0.83 → 0.97, central apnea 0.84 → 0.98, hypopnea 0.56 → 0.78; specificity 0.54 → 0.61. |
| `mesa_audit_overview.png` | copied (`results/mesa/audit/`) | MESA audit: AHI distribution, label-definition check against NSRR official AHI, and per-subject belt quality. |
| `mesa_usable_funnel.png` | generated (`results/mesa/audit/usable_funnel.csv`) | MESA usable-subject funnel: 250 downloaded → 220 usable after quality criteria. |
| `mesa_ecg_baseline_overview.png` | copied (`results/mesa/ecg_baseline/`) | ECG-only baseline on MESA: per-fold ROC curves (pooled AUROC 0.609) and weak subject-level AHI correlation (r ≈ 0.31 / 0.30). |
| `inhouse_domain_gap_separability.png` | generated (`results/inhouse_canet/domain_gap/domain_classifier_auroc.csv`) | A classifier separates MESA belts from our accelerometer at AUROC 0.99 (control MESA vs MESA: 0.51). |
| `inhouse_domain_gap_spectrum.png` | copied (`results/inhouse_canet/domain_gap/`) | Effort-stream spectrum and breathing features: accelerometer has a breathing peak near 0.2–0.25 Hz but is broadband and irregular vs belts. |
| `inhouse_domain_gap_amplitude_shape.png` | copied (`results/inhouse_canet/domain_gap/`) | After normalisation amplitude matches belts (RMS ≈ 1); waveform shape (kurtosis) differs. |
| `inhouse_negative_control_summary.png` | generated (`results/inhouse_canet/negative_control/`) | Healthy-night flag rate: ECG-only 0.58, effort-only accelerometer 0.84, CANet 0.94, white-noise effort 0.33 vs MESA low-AHI median 0.28; transfer failure isolated to the effort branch, not an apnea result. |
| `inhouse_negative_control_per_night.png` | copied (`results/inhouse_canet/negative_control/`) | Per-night flag fractions for all 10 usable healthy nights (S03–S15) and all models; every night is above the MESA reference line. |

Not included on purpose: the per-recording in-house waveform/diagnostic plots (their filenames contain volunteer names).
