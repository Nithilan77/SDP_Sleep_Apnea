# Slide outline — 15 slides, ~15–18 min

One idea per slide. Figures refer to files in `figures/`. Numbers are from `content.md` (section in brackets). Keep "healthy volunteers / validation only" visible wherever our own hardware appears.

## 1. Title
- Wearable sleep-apnea screening from ECG + breathing effort (one chest sensor). Names, guides, ST Microelectronics collaboration, "Pinnacle" programme.
- Figure: none.

## 2. Why this matters [§2]
- ~1 in 7 adults have sleep apnea; ~80% of moderate–severe cases undiagnosed; PSG is expensive, lab-bound, long waitlists.
- Idea: a cheap wearable screen closes the gap. Screening, not diagnosis.
- Figure: none (big-number layout: "1 in 7", "~80%").

## 3. The idea and the system [§1]
- One SensorTile on the chest: ECG (QVAR + electrodes) + breathing effort (IMU). Two branches, fused at decision level. Claim: heart + breathing beats either alone.
- Figure: architecture diagram (draw from the §1 pipeline; no file).

## 4. The core constraint [§3]
- Our volunteers are healthy 19–20-year-olds → train on public clinical data; our own recordings are validation only, never trained on. Safe validation = voluntary breath-holds + healthy negative controls.
- Rule on every result: subject-independent evaluation.
- Figure: none (simple two-column graphic: "train: public clinical" / "validate: our hardware").

## 5. Data [§4]
- PhysioNet Apnea-ECG: 35 labelled records, 100 Hz, 16,949 labelled minutes. MESA: 250 downloaded → 220 usable; ECG + chest/abdomen belts + SpO2, expert-scored events. Ours: 15 SensorTile recordings (10 usable for the night-level study).
- Figure: `mesa_usable_funnel.png` (optionally with `mesa_audit_overview.png` as backup/appendix).

## 6. ECG branch step 1: classical baseline [§5]
- 7 hand-crafted features → logistic regression: 0.711 accuracy / 0.637 sensitivity / 0.758 specificity (35-fold LOSO). This is the honest floor.
- Per-recording 26/35 = 0.743, noted as a crude metric.
- Figure: first bar group of `phase3_progression.png` (or full figure shown on next slide).

## 7. ECG branch step 2: 1D-CNN [§6]
- RR-interval + R-peak amplitude CNN: 0.812 / 0.756 / 0.846. Ablation: the amplitude channel does the work; extra loss weighting hurts.
- Figure: `phase3_progression.png`.

## 8. Which ECG representation? Eight-method study [§7]
- Real time series beat summary statistics; adopted method stays best; QRS-area close behind; frequency-domain HRV weakest; image methods trade sensitivity for specificity.
- Figure: `feature_study_comparison.png` (master table as appendix).

## 9. Why add breathing effort: MESA setup [§4.2, §8]
- ECG-only on harder, larger MESA data: AUROC 0.610 / AUPRC 0.335. Introduce CANet: ECG stream + effort stream + cross-attention (169k parameters), 30-s epochs, grouped 5-fold, no SpO2.
- Figure: `mesa_ecg_baseline_overview.png`.

## 10. Headline: CANet vs ECG-only [§8.2–8.3]
- AUROC 0.610 → 0.780 (+0.171, CI +0.151 to +0.191); AUPRC 0.335 → 0.490; AHI-proxy r 0.31 → 0.74. Every fold improves. Ablations: effort-only 0.747, concat 0.744, SpO2 ceiling 0.813 (not on our wearable).
- Figure: `canet_vs_baseline.png`.

## 11. Where the gain comes from [§8.4]
- Obstructive apnea 0.83 → 0.97; central 0.84 → 0.98; hypopnea 0.56 → 0.78. Most gain is the effort signal; attention adds +0.033 to +0.036 AUROC.
- Figure: `per_class_gain.png`.

## 12. Honest caveats [§8.5]
- Gain mostly from effort; concat not parameter-matched (102k vs 169k); single seed; AHI is an epoch-rate proxy; specificity only 0.61; elderly cohort; belts, not accelerometers.
- Figure: none (text slide).

## 13. Our hardware: it runs, but effort does not transfer [§9]
- Pipeline runs end to end on 10 healthy nights (deployable). But belt vs accelerometer separable at AUROC 0.99 (control 0.51); healthy flag rate 94% vs MESA low-AHI 28%; white-noise ablation (33%) isolates the effort branch. Transfer finding, not an apnea result.
- Figures: `inhouse_domain_gap_separability.png` and `inhouse_negative_control_summary.png` (side by side). Backup/appendix: `inhouse_domain_gap_spectrum.png`, `inhouse_domain_gap_amplitude_shape.png`, `inhouse_negative_control_per_night.png`.

## 14. What's next [§10–11]
- Monday: breath-hold session (sternum-mounted, 5–6 holds of 15–25 s, one 5–7 s decoy, sync taps, timestamped log). Question: can the accelerometer see cessation at all? Then adapt the effort branch; then fusion on our hardware and per-night report.
- Figure: none (timeline graphic).

## 15. Summary [§12]
- 0.812 per-minute from ECG alone; AUROC 0.610 → 0.780 with effort on MESA; our hardware runs but effort needs adaptation; Monday's breath-holds decide next steps.
- Figure: none.

## Appendix (optional)
- A1 Eight-method master table (§7). A2 Phase 3 ablation table (§6). A3 Full ablation + per-class table (§8.3–8.4). A4 MESA audit details: `mesa_audit_overview.png`. A5 PhysioNet-vs-QVAR domain-gap table (§9.1). A6 Glossary (§13).
