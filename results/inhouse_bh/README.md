# Track B — in-house voluntary breath-hold feasibility (2026-10-20)

**Task:** window-level detection of `voluntary_breath_hold` (positive) vs `baseline_reference` + `recovery` (negative) from the chest accelerometer; uncertain intervals excluded. **This is not sleep-apnea detection** and never uses clinical apnea labels. Reported separately from the public PhysioNet/MESA results (Track A).
Policy: `CLAUDE.md` §4 (revised 2026-10-20). Code: `src/inhouse_bh/`. Annotation file: `annotations.csv`.

## Labels (`annotations.csv`)
| recording | usable for supervised work | intervals (s) |
|---|---|---|
| S1_ECGTILE (session 1, 2026-10-07, one volunteer) | **yes, with ±5 s boundary uncertainty** | 7 `voluntary_breath_hold` cores (142 s; 3 events visually checked = `high`, 4 = `medium`), 14 `uncertain` boundary bands (140 s), 7 `recovery` (140 s), 7 `baseline_reference` (375 s) |
| S2 (Mithin, 2026-10-11) | **no** — 17 stopwatch intervals, all `uncertain`, `time_base=stopwatch`, `unaligned_to_any_tile` | see `results/breath_hold_s2/mithin/REPORT.md` (shuffle-null p = 0.49–0.95: no tile aligns) |
Windows: 10 s, stride 2 s, 3-axis 0.08–0.6 Hz accelerometer at 32 Hz, scaled by the median RMS of its ±80 s context (no labels used). Windows touching an uncertain interval are never created (**0 rejected-by-label windows by construction**; quality rejection at inference: flat sensor, non-finite, > 10× local RMS). Counts: **237 original windows (40 hold / 197 non-hold)**; + 6 augmented copies each in `aug` configs (1,422); + simulator windows after the separability QC gate (75 of 100 simulated recordings rejected; 5,282 windows kept in the LOEO run, 3,764 in an earlier smaller run). Generated windows are not independent observations.

## Leave-one-event-out (7 folds, 3 seeds; `loeo_summary.csv`, `loeo_predictions.csv.gz`, `loeo_unlabelled_flag_rates.csv`)
**Not an independent-subject evaluation**: one volunteer, one session, 7 events (blocks ≥ 5 s apart). Threshold fixed 0.5. Pooled over the held-out blocks:
| config (ensemble = mean of LR, HGB, CNN where available) | AUROC | AP | sens | spec | events with ≥ 50% of core flagged | flag rate in unlabelled time |
|---|---|---|---|---|---|---|
| real, ENS | **0.826** | 0.598 | 0.60 | 0.81 | 4/7 | 0.13 |
| real, HGB / LR / CNN | 0.769 / 0.805 / 0.777 | 0.48 / 0.46 / 0.34 | 0.55 / 0.73 / 0.68 | 0.92 / 0.67 / 0.71 | 4 / 5 / 5 | 0.08 / 0.32 / 0.21 |
| + augmentation, ENS | 0.812 | 0.556 | 0.54 | 0.80 | 4/7 | 0.14 |
| + simulator windows, ENS | 0.792 | 0.411 | 0.78 | 0.66 | 5/7 | 0.36 |
| + augmentation + simulator, ENS | 0.773 | 0.450 | 0.65 | 0.68 | 5/7 | 0.31 |
| CNN pretrained on simulator → fine-tuned | 0.807 | 0.512 | 0.65 | 0.72 | 4/7 | 0.24 |
| CNN, SSL encoder (masked reconstruction on 14 other in-house recordings, never session 1), frozen + head | **0.828** | **0.648** | 0.69 | 0.73 | 5/7 | 0.29 |
| CNN, SSL init, fine-tuned | 0.811 | 0.586 | 0.60 | 0.76 | 4/7 | 0.24 |
**Findings.** (1) Window-level discrimination is moderate (AUROC ≈ 0.77–0.83) and the differences between configurations are of the size of the seed/fold noise with only 7 events — no configuration is demonstrably better. (2) Augmentation: no gain. (3) Simulator data: higher sensitivity but clearly more false flags (unlabelled-time flag rate 0.31–0.36 vs 0.13) — not recommended. (4) SSL pretraining + linear head gave the best AUROC/AP (0.828 / 0.648 vs 0.777 / 0.344 for the same CNN trained from scratch) — a hypothesis worth testing on more sessions, not a finding at n = 7. (5) Events 3 and 5 are almost never detected in any configuration (mid-hold burst; hold not cleanly quiet — the same two events whose boundaries failed the earlier visual check). (6) The unweighted LR+HGB+CNN mean ranks windows best among the real-only variants (0.826 vs 0.77–0.81 for its members) with equal weights chosen in advance; the difference is within noise at n = 7.

## Whole-recording behaviour (`operating_points.csv`, seed 0; optimistic because the sweep re-uses the 7 events)
Scoring every window of session 1 (each point from the fold that excluded its event): real-only ensemble, thr 0.5, segment ≥ 5 windows (≥ 18 s): **4/7 holds detected, 6 additional segments (≈ 12 per hour)**; with augmentation + simulator data 5/7 detected but 22 additional segments (44/h). **Deployed model = real-only ensemble** (`deployed/`, `train_final.py`), thr 0.5, min run 5 — fixed beforehand from the core duration, not tuned. A prior-corrected threshold (0.88) detects 0–2/7, i.e. the models are not calibrated for the ~12% hold time of a protocol recording.
**In-sample on session 1** the deployed ensemble flags the holds but also quiet-breathing valleys; the dashboard says so.

## Healthy-volunteer recordings as unlabelled controls (`healthy_controls.csv`; models not trained on them)
10 usable recordings (0.8–7.5 h). For five of them (H04, H06, H07, H12, H15) most windows (62–99%) are rejected as "flat sensor" — the band-passed accelerometer is at the quantisation floor, i.e. the box was not on a breathing chest — so their flag rates are not meaningful. Among scored recordings: 0–0.9% of windows flagged, 0–1.2 segments/hour. Reading: a low flag rate on sleep-like recordings, **not** a specificity estimate (the true state is unknown) and nothing about apnea.

## Reproduce
`python3 -m src.inhouse_bh.annotate` · `python3 -m src.inhouse_bh.run_experiments --seeds 0 1 2` (~90 min on CPU) · `python3 -m src.inhouse_bh.operating_point` · `python3 -m src.inhouse_bh.train_final --config real` · `python3 -m src.inhouse_bh.healthy_controls`.
