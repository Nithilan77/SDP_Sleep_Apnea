# 48-hour completion work log (started 2026-10-19)

Everything below was produced and verified in this session; numbers are copied from the saved artifacts.

## 1. Audit (what already worked)
- Deployed effort pipeline `python3 -m src.deploy.predict_session <file>` ran on session 1 (3.7 s). CNN checkpoint `results/phase3_cnn/deployable/` loads. MESA CANet checkpoints exist.
- No web framework was installed; Flask 3.0.3 was installed (`pip install --user --break-system-packages flask`).
- NOT available: the report templates (not found on this machine) and the Google Drive folder (no authenticated Drive access).

## 2. RNN / LSTM (`src/ecg/rnn_lstm.py`, `rnn_lstm_select.py`, `rnn_lstm_v2.py`)
Task and protocol identical to the adopted CNN: per-minute apnea from the 2-channel (RR, R-amplitude) 60-step sequence, 16,949 minutes, 35 Apnea-ECG records, leave-one-subject-out, per-fold normalisation, class-balanced BCE, seed 0, single run.

| Model | Params | Acc | Sens | Spec | Per-recording |
|---|---|---|---|---|---|
| CVHR logistic baseline (existing) | - | 0.711 | 0.637 | 0.758 | - |
| 1D-CNN (existing, adopted) | ~3.4k | **0.812** | 0.756 | 0.846 | 28/35 |
| RNN primary (hidden 32, 30 ep, last state; pre-specified) | 1,697 | 0.581 | 0.586 | 0.578 | 25/35 |
| LSTM primary (same) | 5,153 | 0.640 | 0.517 | 0.717 | 24/35 |
| RNN v2 (hidden 64, 80 ep, mean-pool) | see summary.json | 0.737 | 0.669 | 0.779 | 25/35 |
| LSTM v2 (hidden 64, 80 ep, last state) | see summary.json | 0.746 | 0.634 | 0.816 | 29/35 |

- The primary configuration underfit (train accuracy 0.72-0.75 on a diagnostic fold), so v2 changed epochs/width/pooling.
- v2 configs were chosen on a fixed 7-record validation split (`results/phase3_rnn_lstm/selection.json`). Those 7 records are also LOSO test folds, so v2 numbers are mildly optimistic; the primary run is the unbiased one.
- Conclusion: neither recurrent model beats the CNN (0.812). LSTM v2 and RNN v2 beat the classical baseline (0.711) only by ~3 points and that margin is within the optimism caveat. LSTM > RNN in both settings. Single seed, no CIs.
- Checkpoints: `results/phase3_rnn_lstm/deployable_*`, `results/phase3_rnn_lstm_v2/deployable_*` (trained on all 35 records, no held-out set).

## 3. Web app (`src/webapp/`)
Run: `python3 -m src.webapp.app` then open http://127.0.0.1:5000. Upload .csv/.txt SensorTile recording (or the demo button). Shows validation/metadata, signals, ECG-branch per-minute probabilities from CNN/RNN/LSTM checkpoints, effort-branch hold-likelihood from the two heads, limitations. Every output is computed from the uploaded file.
Tests: `python3 tests/e2e_webapp.py` -> `results/webapp_e2e/test_table.md` (9/9 PASS, real server subprocess + HTTP). Screenshot: `results/webapp_e2e/screenshot_demo_result.png`.
Known caveat: the ECG-branch RR+amplitude column flags ~100% of minutes on healthy recordings (QVAR amplitude domain gap, already documented in PROGRESS §7); the RR-only column is the more meaningful one. Large overnight files are processed synchronously (slow).

## 4. Second breath-hold recording (`results/breath_hold_s2/`)
Used the only candidate found locally: `~/Desktop/Mithin-BreathHold/user_MithinBreathHold_ECG_000.csv` (the Drive link could not be accessed; it is unconfirmed that this is the Drive file).
- 1574.5 s, 242.7 Hz, 382,157 samples, no sync taps found (start/end), 0% implausible RR, 1572 beats, 59.9 bpm.
- The reconstructed lap sequence (991 s baseline + 30/90 s alternation + 57 s) equals session 1's laps (total 1800.46 s). It cannot fit this recording (1574 s): holds 6 and 7 fall beyond the end.
- Accelerometer: breathing-band std 4.9e-5 g vs 1.36e-3 g in session 1 (~28x lower); per-axis std 2-3 mg vs 58-180 mg; no breathing spectral peak (peak/median 1.6 vs 3.6). The sensor was essentially motionless / not coupled to chest motion, so the effort branch has no usable breathing signal here.
- At nominal session-1 windows the effort ratios in S2 are 7.38, 0.14, 1.48, 1.39, 0.34 (n=5), consistent with noise, i.e. session-1 findings are NOT reproduced; session-1 ratios by the same code: 0.22, 0.39, 0.79, 0.11, 0.38, 0.17, 0.14. Alignment is unverified, so this is not evidence that holds have no effect.
- ECG quality is good; HR delta in holds is not consistent (S1 mostly negative, S2 mostly positive).

## 5. Corrections and continuation (added 2026-10-19, later the same day; the sections above are left as originally written)
- **Drive access (supersedes §1 and §4).** The Drive folder was *not* blocked by permissions: the folder ID is `1wS9USehoXPs37GDOZKEbiNTCKUq29yo_` and the trailing underscore had been dropped, which gives a genuine 404. With the correct ID it is listable and downloadable. Nine files were downloaded to `data/recordings/drive_2026-10-19/` and verified (sizes, zip integrity, SHA-256). The Desktop `Mithin-BreathHold` ECG and EMG files are byte-identical to the Drive files, so the "unconfirmed" caveat in §4 is resolved for them. The Drive EOG file is a truncated byte-prefix of the Desktop EOG file. Details: `docs/PROGRESS.md` §7h.5, `docs/HANDOFF_2026-10-09.md` §11.3.
- **§4 timeline statement.** "The reconstructed lap sequence equals session 1's laps" was an assumption applied to session 2; no evidence supports it for that recording (no sync taps, no laps file, 1574.5 s log). Treat the session-2 effort ratios quoted in §4 as arbitrary-window values, not as a failed replication of a protocol.
- **§3 (web app).** The page described there was replaced by a multi-page dashboard (`docs/PROGRESS.md` §7h.6). The 9/9 table `results/webapp_e2e/test_table.md` belongs to the old app; the new suite `tests/e2e_webapp.py` was subsequently run: 25/25 PASS (`results/webapp_e2e/test_table.v2.md`; `docs/PROGRESS.md` §7h.6).
- **§2 (RNN/LSTM).** A controlled multi-seed comparison of CNN/LSTM/GRU/TCN replaced the single-seed numbers as the reference (`docs/PROGRESS.md` §7h.1): CNN 0.753, TCN 0.760, GRU 0.717, LSTM 0.692 accuracy at threshold 0.5; the CNN remains primary. Keep §2's table as history.

