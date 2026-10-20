# Mithin session 2 — timestamp reconstruction and synchronisation audit (2026-10-20)

Code: `src/breath_hold/s2_timeline.py`, `s2_report.py` · Raw files untouched (read-only). Earlier results not overwritten.

## 1. What the "timestamp screenshots" are
No image screenshots exist on this machine (searched home, Drive copy, Pictures). The only timestamp material is
`data/recordings/drive_2026-10-19/MithinBreathHold-TimeStamps.mp4` (13.6 s, 735 frames, 384x832): an **iPhone Stopwatch
screen-recording, stopwatch stopped at 33:34.04, lap list scrolled from Lap 17 to Lap 1**. It was decoded with
`opencv-python-headless` (installed with `pip --user`; it was the one thing missing last session) and all laps were read at full
resolution from three scroll positions. If separate screenshots were also meant to be supplied, they are **not** present — please add them.

## 2. Directly observed (`laps_extracted.csv`, `timestamps_extracted.json`)
Laps (s): 1080.84 | 89.18 30.13 89.86 30.06 89.87 30.11 89.86 30.14 89.99 29.98 89.91 30.06 90.01 29.94 90.06 | 3.96.
Total shown 33:34.04 = 2014.04 s; sum of laps = 2013.96 s (0.08 s difference = display rounding / running lap). Phone status-bar clock 12:55 is
the time the video was filmed, **not** a session clock.
**Calculated:** cumulative lap start/end times (stopwatch time). **Inferred (not in the video):** the seven ~30 s laps (3,5,…,15) are the holds,
lap 1 (18 min) is set-up/baseline. Seven holds, 7×~30 s, 90 s rests = session-1 (v1) schedule; **protocol v2 (9 holds, 5–60 s, sync taps) was not followed.**

## 3. Files and provenance
| file | duration | fs | notes |
|---|---|---|---|
| Mithin-BreathHold ECG tile | 1574.5 s | 242.72 Hz | clean ECG (60 bpm); Drive = Desktop copy |
| Mithin-BreathHold EMG tile | 1045.6 s | 242.98 Hz | QVAR clipped flat; IMU only |
| Mithin-BreathHold EOG tile (Desktop, complete) | 2131.9 s | 243.16 Hz | Drive copy is a truncated byte-prefix; QVAR near-flat |
| Mithin10 ECG / EOG / EMG (`~/Desktop/Mithin/`) | 2194.3 s / 2864.2 s / **0-byte** | 243.10 / 242.70 Hz | **separate recording** (36 min, ECG 2058 beats, 54 bpm); no timestamps, no annotations; not a breath-hold session as far as the data show (`mithin10_audit.csv`) |
Every session-2 log ends in one truncated row (dropped by the loader). No tile shares a clock with another tile, the stopwatch or the Muse log.

## 4. Synchronisation test (reproducible; `sync_scan_summary.json`, `fig_sync_scans.png`)
The stopwatch has no event shared with any tile, so the offset τ (recording time of stopwatch zero) is unknown. For every τ that keeps ≥ 5 of 7 inferred holds
inside a tile, the score = mean over holds of [median log breathing-band accel envelope inside the hold (3 s inset) − same in ±60 s flanks]. The best τ is compared with the
best score from 300 block-shuffled (30 s blocks) copies of the same envelope with the whole scan repeated (max-statistic null).

| tile | best τ (s) | best score | p (shuffle null) |
|---|---|---|---|
| ECG | −1324 | −0.22 | 0.90 |
| EMG | −725 | −0.36 | 0.95 |
| EOG | +386 | −0.26 | 0.49 |

**No tile shows hold-quiet structure at any offset beyond what shuffled data produce → no alignment is established, no event boundary in recording time can be assigned, and
no effort ratio can be computed for session 2.** A second, independent scan (`results/mithin_s2/`, written by a parallel session that was stopped; different statistic,
60-s minimum hold overlap, null scanned at 3× coarser offset spacing than the observed scan) reported p = 0.04 for EMG and EOG and 0.76 for ECG. Those two values are
uncorrected over three tiles (Bonferroni ≈ 0.12), based on only two holds overlapping, and the null/observed offset grids differ (anti-conservative). They are kept as a
sensitivity result and are **not** accepted as evidence. The 4-sustained-dip pattern found earlier in the EMG tile's breathing line is likewise not 7 holds.

## 5. Session 1 vs session 2
| | S1 (Muku, 2026-10-07) | S2 (Mithin, 2026-10-11) |
|---|---|---|
| sync | stopwatch total vs recording → 4.8 s lag; events analysable | none; not alignable |
| ECG tile | 1795.6 s, 72 bpm, accel breathing in 31% of 60 s windows | 1574.5 s, 59.9 bpm, 4% (≈28× weaker) |
| events | 7 holds, inner-window effort ratio <0.8 in 7/7 | 7 holds on the stopwatch only; **no verified signal evidence** |
Replicated: nothing yet (S2 cannot test S1's finding). Different: protocol adherence, tile durations, breathing visibility.
## 6. What can / cannot be concluded
Can: exact stopwatch schedule (observed), tile integrity and rates, that session 2 did not follow v2 and has no usable sync. Cannot: hold timing in any tile, effort ratios,
detector validation, HR response. **Most valuable fix:** re-record with the v2 sync taps, or film the stopwatch next to the box once (a shared clock event).
