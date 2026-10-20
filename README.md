# SDP Sleep Apnea — Cardiorespiratory Sleep Apnea Detection

A wearable sleep apnea *screening* system (not a diagnostic device) that fuses ECG and respiratory-effort (IMU) signals from a single chest-worn sensor (ST SensorTile.box PRO) to flag people who should get a proper clinical sleep study.

- **[`CLAUDE.md`](CLAUDE.md)** — full project brief: motivation, hardware, data sources, architecture, methodology rules, phase plan. Start here.
- **[`docs/PROGRESS.md`](docs/PROGRESS.md)** — detailed, numbers-backed record of everything built and measured so far, how to reproduce each result, and what's still open.

## Status

*Last updated 2026-10-19 (earlier bullets from 2026-10-09 kept; later work is in the section below). Numbers and sources: `docs/PROGRESS.md` (§7h for 2026-10-19); full record and resume instructions: `docs/HANDOFF_2026-10-09.md` (§11).*

- **ECG branch (PhysioNet Apnea-ECG, leave-one-subject-out):** classical CVHR baseline 71.1% per-minute accuracy; 1D-CNN (RR + R-peak amplitude) **81.2% accuracy / 75.6% sensitivity / 84.6% specificity**; a frozen deployable model exists but does not transfer to our QVAR sensor as-is.
- **Fusion model (MESA, 220 subjects, subject-independent):** CANet (ECG + respiratory effort, cross-attention) **AUROC 0.780 vs 0.609 ECG-only**. It does *not* transfer to our accelerometer as-is (belt-vs-accelerometer separability AUROC 0.997; ~94% healthy-night flag rate); four label-free adaptation attempts failed (§7d).
- **Our own hardware (healthy volunteers; in-house breath-hold feasibility track, reported separately from the public results):** a first breath-hold session (7 holds, one subject) shows accelerometer effort visibly collapsing during holds (effort ratio < 0.8 in 7/7 events under the conservative window); automatic boundary detection and HR direction are unresolved (§7e). A supervised transfer experiment (§7f) found **no demonstrated benefit of frozen MESA features over simple hand-crafted effort features** at this data scale (pilot, n = 7 events). An end-to-end wrapper (`python3 -m src.deploy.predict_session <recording>`) runs on our recordings.
- Not done: decision-level fusion and per-night report on our hardware (Phases 6-7), a validated cessation/boundary detector, subject-independent validation on our hardware.

**Update 2026-10-19** (uncommitted work; details in `docs/PROGRESS.md` §7h):
- **Read the 81.2% above as the historical Phase 3 number** (its CNN variant was chosen on the same test folds; single seed). A controlled multi-seed re-evaluation (LOSO, inner-validation early stopping/thresholds, 3 seeds) gives the CNN **75.3% accuracy at threshold 0.5** (AUROC 0.807); TCN 76.0%, GRU 71.7%, LSTM 69.2%. No recurrent/temporal model beats the CNN. The gap to 81.2% is explained: the historical recipe averages 77.8% ± 3.0 over three seeds (81.2% was the best seed) and drops to ≈ 75.4% when trained on 29 instead of 34 records; a gradient-boosting model on hand-crafted HRV/CVHR features reaches 80.2% accuracy @0.5 (AUROC 0.84) under the same protocol — competitive with, not demonstrably better than, the CNN (paired CI includes 0).
- **MESA SpO2 ablations** (same 220 subjects/folds, AUROC): ECG 0.609, SpO2 0.775, ECG+SpO2 0.769, effort 0.747, ECG+effort 0.780, ECG+effort+SpO2 0.813. Our wearable has no SpO2.
- **In-house recordings:** the Google Drive folder (ID ends in `_`) was downloaded to `data/recordings/drive_2026-10-19/` and verified; the second breath-hold session could not be aligned to an event timeline and its tiles are not synchronised. See §7h.5.
- **Dashboard:** `python3 -m src.webapp.app` → http://127.0.0.1:5000 (overview, in-house recordings, upload & inference, model comparison, MESA multimodal, methodology, report download). End-to-end tests (`python3 tests/e2e_webapp.py`, real server over HTTP): 25/25 pass as of 2026-10-19.

**Update 2026-10-20** (details: `docs/PROGRESS.md` §7i, `docs/EXPERIMENT_MANIFEST_2026-10-20.md`):
- **Policy:** in-house recordings may now be used for training/fine-tuning/augmentation (`CLAUDE.md` §4) with strict separation; voluntary breath-holds are `voluntary_breath_hold`, never clinical apnea; public (Track A) and in-house (Track B) results are reported separately.
- **Best ECG-only model under one protocol: gradient boosting on HRV/CVHR features (acc 0.802 / AUROC 0.839, 35-fold LOSO).** Ensembles (PhysioNet and MESA) and simulator/augmentation expansion do **not** improve on it with supportable CIs (`results/ensemble/`, `results/synth/`).
- **Breath-hold track (one volunteer, feasibility):** leave-one-event-out AUROC 0.77–0.83; deployed real-only ensemble detects 4/7 holds with ≈ 12 extra segments/hour in a whole-recording scan (`results/inhouse_bh/`). Session 2 (Mithin) cannot be aligned to its sensor logs and gives no labels (`results/breath_hold_s2/mithin/`).
- **Dashboard** (`python3 -m src.webapp.app`): new *Breath-hold demo*, *Ensembles & expanded data*, *System status* pages; checkpoint loading fixed (project-root paths + SHA-256 pinning, `docs/checkpoints.json`). Tests: `python3 tests/e2e_webapp.py` and `python3 -m unittest tests.test_inhouse_policy_flow -v`.

## Setup

```
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Download the [PhysioNet Apnea-ECG database](https://physionet.org/content/apnea-ecg/1.0.0/) into `data/physionet/` (not committed here — public, ~580MB; e.g. via `wget -r -N -c -np https://physionet.org/files/apnea-ecg/1.0.0/` or the `wfdb` package's `dl_database`). Expected files: `a01.dat`/`a01.hea`/`a01.apn`/`a01.qrs`, etc. for `a01`–`a20`, `b01`–`b05`, `c01`–`c10`, `x01`–`x35`.

Run any phase's script from the project root (use the system `python3` for anything that imports torch; the project `venv/` on the current Linux machine has no torch -- see `docs/HANDOFF_2026-10-09.md` §8), e.g.:
```
python src/ingest/physionet.py
python src/ecg/cvhr_baseline.py
python src/ecg/cnn_apnea_final.py
```
See `docs/PROGRESS.md` §3–4 for the full list and expected results.

## Running the dashboard (also on a fresh clone)

```
git clone https://github.com/Nithilan77/SDP_Sleep_Apnea.git && cd SDP_Sleep_Apnea
pip install -r requirements.txt          # includes Flask; use a Python with torch (system python3 on the lab machine)
python3 -m src.webapp.app                # then open http://127.0.0.1:5000
```

Everything the dashboard needs is in git, **including** the MESA-trained CANet weights `data/mesa/cache/final_models/headline.pt` (~0.7 MB; force-added although `data/mesa/` is gitignored — it is the only file from that folder that is tracked) used by the effort branch. Three things are deliberately *not* in git (human-subject / gated data):
- **Demo button** needs `data/recordings/breath_hold/user_MukuBreathHold_ECG_07_10_2026.csv`; without it the button reports "Demo recording missing". Uploading your own SensorTile ECG `.csv`/`.txt` (>= 10 min) works.
- The MESA dataset itself and the other model caches under `data/mesa/` (only needed to retrain, not to run the dashboard).
- Page content (Overview, In-house, Model comparison, MESA, Methodology) is read from committed files under `results/`; the in-house figures are included.

Tests: `python3 tests/e2e_webapp.py` (needs the recordings above; 25/25 pass on the original machine).

Our own SensorTile recordings (`data/recordings/`) are never committed to this repo (human-subject data). Under the in-house data-use policy revised 2026-10-20 (`CLAUDE.md` §4) they may be used for training, fine-tuning, augmentation and synthetic-data generation, with strict train/validation/test separation; voluntary breath-holds are labelled `voluntary_breath_hold`, never clinical apnea; public (PhysioNet/MESA) and in-house results are reported separately.
