# SDP Sleep Apnea — Cardiorespiratory Sleep Apnea Detection

A wearable sleep apnea *screening* system (not a diagnostic device) that fuses ECG and respiratory-effort (IMU) signals from a single chest-worn sensor (ST SensorTile.box PRO) to flag people who should get a proper clinical sleep study.

- **[`CLAUDE.md`](CLAUDE.md)** — full project brief: motivation, hardware, data sources, architecture, methodology rules, phase plan. Start here.
- **[`docs/PROGRESS.md`](docs/PROGRESS.md)** — detailed, numbers-backed record of everything built and measured so far, how to reproduce each result, and what's still open.

## Status

ECG branch: classical CVHR baseline (71.1% per-minute accuracy) and a 1D-CNN (81.2% accuracy / 75.6% sensitivity / 84.6% specificity, subject-independent LOSO on PhysioNet Apnea-ECG) are done, plus a frozen deployable model. Effort branch (IMU → breathing envelope → cessation detection) is built and validated against a synthetic signal. Real-hardware validation and sensor fusion are next, pending SensorTile hardware availability. See `docs/PROGRESS.md` for the full picture.

## Setup

```
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Download the [PhysioNet Apnea-ECG database](https://physionet.org/content/apnea-ecg/1.0.0/) into `data/physionet/` (not committed here — public, ~580MB; e.g. via `wget -r -N -c -np https://physionet.org/files/apnea-ecg/1.0.0/` or the `wfdb` package's `dl_database`). Expected files: `a01.dat`/`a01.hea`/`a01.apn`/`a01.qrs`, etc. for `a01`–`a20`, `b01`–`b05`, `c01`–`c10`, `x01`–`x35`.

Run any phase's script from the project root, e.g.:
```
python src/ingest/physionet.py
python src/ecg/cvhr_baseline.py
python src/ecg/cnn_apnea_final.py
```
See `docs/PROGRESS.md` §3–4 for the full list and expected results.

Our own SensorTile recordings (`data/recordings/`) are validation-only and never committed to this repo (human-subject data).
