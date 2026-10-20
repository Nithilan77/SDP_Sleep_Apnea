# Experiment manifest and baseline audit — 2026-10-20

Distinguishes **original** results (before this session, untouched), **new** experiments (this session), **corrected** statements and **superseded** findings.
Nothing in `results/` that existed before was overwritten (new outputs go to new directories: `results/ensemble/`, `results/synth/`, `results/inhouse_bh/`, `results/breath_hold_s2/mithin/`).

## 0. State at start (verified, not copied from summaries)
* Git: branch `main`, HEAD `a184d56`; untracked at start: `project_structure.txt`, `results/mesa/canet_lstm/` (+log), `src/mesa/canet_lstm.py`, `results/webapp_e2e/demo_result.html|png`. Dashboard process (`python3 -m src.webapp.app`) was running. Python 3.12.3, torch 2.13.0+cu130, RTX 5000 Ada, 20 cores. No other training job was running (four other `claude` sessions found in the checkout were stopped at the user's instruction; their partial outputs — `src/ensemble/run_ensemble.py`, `results/ensemble/`, `src/mithin/`, `results/mithin_s2/` — were reviewed; the ensemble code was re-verified and adopted, `results/mithin_s2/` is kept as a secondary sensitivity result).
* Recomputed from saved per-minute predictions: Phase-2 CVHR logistic regression **acc 0.7111 / sens 0.6365 / spec 0.7577** (16,949 minutes, 35-fold LOSO); Phase-3 CNN "final" **acc 0.8116 / sens 0.7564 / spec 0.8459** — matches the documented numbers.

## 1. Public-dataset baselines (Track A) — original results
| Model | Protocol | Result (per-minute, thr 0.5, mean of 3 seeds) | Source |
|---|---|---|---|
| CVHR logistic regression | LOSO(35), 7 features | acc 0.711 | `results/phase2_cvhr_baseline/` |
| 1D-CNN, historical headline | LOSO(35); variant chosen on test folds (selection optimism) | acc 0.812 (single saved run; 3-seed mean 0.778 ± 0.030) | `results/phase3_cnn/`, PROGRESS §3, §7h.3 |
| Controlled comparison (clean protocol: 5 inner-validation records, val-AUROC early stopping, no test-fold choices) | LOSO(35), 3 seeds | **HGB 0.802 / AUROC 0.839**; TCN 0.760 / 0.797; CNN 0.753 / 0.807; GRU 0.717 / 0.739; LSTM 0.692 / 0.710; logreg 0.737 / 0.783 | `results/expA_multiseed/` |
| MESA CANet (ECG + effort belt) | subject-grouped 5-fold, 220 subjects | AUROC 0.780 (ECG-only 0.609; +SpO2 0.813) | `results/mesa/canet/` |
**Best model under one consistent protocol: gradient boosting on HRV/CVHR features** (paired CI vs CNN includes 0 for AUROC). 0.812 is a historical single-run number and is superseded as a headline by the controlled table.

## 2. Inventory
* **Checkpoints** (pinned with SHA-256 in `docs/checkpoints.json`): `data/mesa/cache/final_models/{headline,ecg_only,resp_only,ecg_spo2_lstm,ecg_resp_spo2_lstm}.pt` (headline tracked in git since a184d56), `results/phase3_cnn/deployable/`, `results/phase3_rnn_lstm{,_v2}/deployable_*/`, `results/transfer/deployed_heads.joblib`.
* **In-house recordings** (never committed): `data/recordings/inhouse_ecg/*.txt` (15 files, 10 usable ≥ 600 s), `data/recordings/breath_hold/` (session 1, three tiles), `data/recordings/drive_2026-10-19/` (session 1 copies, session-2 tiles, Muse exports, timestamps video), `~/Desktop/Mithin-BreathHold/` (session-2 complete EOG tile), `~/Desktop/Mithin/` (a separate 36-min "Mithin10" recording; its EMG file is 0 bytes).
* **Timestamp "screenshots":** no images exist; the material is `MithinBreathHold-TimeStamps.mp4` (phone stopwatch screen-recording). Decoded this session; 17 laps transcribed.
* **Known issues found:** (1) dashboard checkpoint loaded by a CWD-relative path (fixed, §4); (2) `results/mithin_s2` and my scan disagree weakly (resolved in `results/breath_hold_s2/mithin/REPORT.md`); (3) earlier statement "no MESA OOF predictions saved" was wrong (corrected in `results/ensemble/README.md`); (4) no `pytest` in the environment — tests use `unittest`.

## 3. New this session
| Work | Where |
|---|---|
| Mithin session-2 timeline + sync audit (negative: no alignment) | `results/breath_hold_s2/mithin/` |
| Ensembles (PhysioNet OOF + MESA OOF): none beats best single | `results/ensemble/` |
| Synthetic SensorTile generator, validator, datasets, QC | `src/synth/`, `results/synth/` (data under gitignored `data/synthetic/`) |
| ECG expanded-data matrix (augmentation / simulator ratios / fine-tune) | `results/synth/exp/` |
| In-house breath-hold track (annotations, windows, augmentation, LR/HGB/CNN, fine-tune, SSL, ensemble; leave-one-event-out) | `src/inhouse_bh/`, `results/inhouse_bh/` |
| Policy revision, pinned checkpoints, dashboard pages | `CLAUDE.md` §4, `src/deploy/checkpoints.py`, `src/webapp/` |
(final numbers: see the sections of `docs/PROGRESS.md` §7i and `results/*/README.md`)
