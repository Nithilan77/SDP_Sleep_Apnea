# SDP Sleep Apnea — Cardiorespiratory Sleep Apnea Detection

A wearable sleep apnea *screening* system (not a diagnostic device) that fuses ECG and respiratory-effort (IMU) signals from a single chest-worn sensor (ST SensorTile.box PRO) to flag people who should get a proper clinical sleep study.

- **[`CLAUDE.md`](CLAUDE.md)** — full project brief: motivation, hardware, data sources, architecture, methodology rules, phase plan. Start here.
- **[`docs/PROGRESS.md`](docs/PROGRESS.md)** — detailed, numbers-backed record of everything built and measured so far, how to reproduce each result, and what's still open.

## Status

*Last updated 2026-10-19 (earlier bullets from 2026-10-09 kept; later work is in the section below). Numbers and sources: `docs/PROGRESS.md` (§7h for 2026-10-19); full record and resume instructions: `docs/HANDOFF_2026-10-09.md` (§11).*

- **ECG branch (PhysioNet Apnea-ECG, leave-one-subject-out):** classical CVHR baseline 71.1% per-minute accuracy; 1D-CNN (RR + R-peak amplitude) **81.2% accuracy / 75.6% sensitivity / 84.6% specificity**; a frozen deployable model exists but does not transfer to our QVAR sensor as-is.
- **Fusion model (MESA, 220 subjects, subject-independent):** CANet (ECG + respiratory effort, cross-attention) **AUROC 0.780 vs 0.609 ECG-only**. It does *not* transfer to our accelerometer as-is (belt-vs-accelerometer separability AUROC 0.997; ~94% healthy-night flag rate); four label-free adaptation attempts failed (§7d).
- **Our own hardware (healthy volunteers, validation only):** a first breath-hold session (7 holds, one subject) shows accelerometer effort visibly collapsing during holds (effort ratio < 0.8 in 7/7 events under the conservative window); automatic boundary detection and HR direction are unresolved (§7e). A supervised transfer experiment (§7f) found **no demonstrated benefit of frozen MESA features over simple hand-crafted effort features** at this data scale (pilot, n = 7 events). An end-to-end wrapper (`python3 -m src.deploy.predict_session <recording>`) runs on our recordings.
- Not done: decision-level fusion and per-night report on our hardware (Phases 6-7), a validated cessation/boundary detector, subject-independent validation on our hardware.

**Update 2026-10-19** (uncommitted work; details in `docs/PROGRESS.md` §7h):
- **Read the 81.2% above as the historical Phase 3 number** (its CNN variant was chosen on the same test folds; single seed). A controlled multi-seed re-evaluation (LOSO, inner-validation early stopping/thresholds, 3 seeds) gives the CNN **75.3% accuracy at threshold 0.5** (AUROC 0.807); TCN 76.0%, GRU 71.7%, LSTM 69.2%. No recurrent/temporal model beats the CNN. The gap to 81.2% is not yet explained — a diagnostic is still running (its seed 0 reproduces 0.81 with the historical recipe, and gives 0.74 when 5 records are withheld from training); a tabular HRV model is under evaluation (preliminary, one seed).
- **MESA SpO2 ablations** (same 220 subjects/folds, AUROC): ECG 0.609, SpO2 0.775, ECG+SpO2 0.769, effort 0.747, ECG+effort 0.780, ECG+effort+SpO2 0.813. Our wearable has no SpO2.
- **In-house recordings:** the Google Drive folder (ID ends in `_`) was downloaded to `data/recordings/drive_2026-10-19/` and verified; the second breath-hold session could not be aligned to an event timeline and its tiles are not synchronised. See §7h.5.
- **Dashboard:** `python3 -m src.webapp.app` → http://127.0.0.1:5000 (overview, in-house recordings, upload & inference, model comparison, MESA multimodal, methodology, report download). End-to-end tests (`python3 tests/e2e_webapp.py`, real server over HTTP): 25/25 pass as of 2026-10-19.

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

Our own SensorTile recordings (`data/recordings/`) are validation-only and never committed to this repo (human-subject data).
