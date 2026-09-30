# In-house SensorTile QVAR-ECG quality report

**Scope: signal quality / plausibility only. No apnea labels exist for this data
(healthy 19-20yo subjects, validation-only per CLAUDE.md Sec.4). This report
cannot and does not produce a sensitivity/accuracy number.**

Files scanned: 15
Files with a plausible QVAR-derived HR (HR 40-120 bpm, implausible-RR <= 1.0%): 13
Files with a usable (non-flat) accelerometer channel: 15 / 15

## Per-file summary

| file                                    | load_error   |   fs_measured_hz |   fs_nominal_hz |   duration_s |   n_samples |   n_gaps | accel_usable   |   n_beats |   mean_hr_bpm |   pct_implausible_rr | plausibility_flag   |
|:----------------------------------------|:-------------|-----------------:|----------------:|-------------:|------------:|---------:|:---------------|----------:|--------------:|---------------------:|:--------------------|
| user_0502-bala_ecg_000.txt              |              |          243.139 |             240 |         20.7 |        5032 |        0 | True           |        30 |         89.96 |                0     | PLAUSIBLE           |
| user_0502-bala_ecg_001.txt              |              |          243.14  |             240 |         18.8 |        4573 |        0 | True           |        26 |         84.46 |                0     | PLAUSIBLE           |
| user_0502-bala_ecg_002.txt              |              |          243.135 |             240 |       5613.8 |     1364920 |        0 | True           |      7330 |         78.34 |                0     | PLAUSIBLE           |
| user_0802-anagesh_ecg_000.txt           |              |          238.079 |             240 |      27127.8 |     6458546 |        0 | True           |     32960 |         72.9  |                1.092 | NOISY_ECG           |
| user_1002-jayendran_ecg_000.txt         |              |          238.088 |             240 |          6.7 |        1585 |        0 | True           |         8 |         77.82 |                0     | PLAUSIBLE           |
| user_1002-jayendran_ecg_001.txt         |              |          238.074 |             240 |      14090.6 |     3354598 |        0 | True           |     14565 |         62.02 |                0.076 | PLAUSIBLE           |
| user_1101hudson_ecg_000.txt             |              |          242.688 |             240 |      20802   |     5048391 |        0 | True           |     15484 |         44.66 |                0.019 | PLAUSIBLE           |
| user_1510hudson_ecg_000.txt             |              |          242.682 |             240 |       2816.8 |      683592 |        0 | True           |      2150 |         45.81 |                0     | PLAUSIBLE           |
| user_1612bhavi_ecg_000.txt              |              |          243.163 |             240 |          1.3 |         320 |        0 | True           |         1 |        nan    |              nan     | NO_BEATS_DETECTED   |
| user_1612bhavi_ecg_001.txt              |              |          243.149 |             240 |      10106.6 |     2457404 |        0 | True           |     11410 |         67.74 |                0.009 | PLAUSIBLE           |
| user_2406Hari(chest_chest & ecg_000.txt |              |          243.085 |             240 |       3959.7 |      962548 |        0 | True           |      3796 |         57.52 |                0.896 | PLAUSIBLE           |
| user_aadhithiyaa_ecg_000.txt            |              |          242.644 |             240 |      18847.7 |     4573299 |        0 | True           |     17765 |         56.55 |                0.152 | PLAUSIBLE           |
| user_aniruth3103_ecg1_000.txt           |              |          242.677 |             240 |      20990.2 |     5093844 |        0 | True           |     18383 |         52.55 |                0.092 | PLAUSIBLE           |
| user_aniruth3103_ecg_000.txt            |              |          242.682 |             240 |         31.7 |        7688 |        0 | True           |        30 |         57.16 |                0     | PLAUSIBLE           |
| user_kishor0404_ecg_000.txt             |              |          243.126 |             240 |      24761.4 |     6020141 |        0 | True           |     23522 |         57    |                0.225 | PLAUSIBLE           |

## Flag legend
- `PLAUSIBLE`: mean HR in a sane resting/light-activity range and RR-interval
  noise comparable to PhysioNet's clean recordings.
- `NOISY_ECG`: HR plausible but too many physiologically-impossible RR intervals
  -- suggests the QVAR ECG lead is too noisy for reliable R-peak detection here.
- `IMPLAUSIBLE_HR`: RR intervals are internally consistent but the resulting mean
  HR is outside a sane range -- likely a lead/contact issue, not true bradycardia
  or tachycardia.
- `IMPLAUSIBLE_HR_AND_NOISY`: both problems -- QVAR signal not usable as-is.
- `TOO_SHORT`: fewer than 2 samples, an aborted capture.
- `NO_BEATS_DETECTED` / `RPEAK_ERROR`: R-peak detector failed outright.

## What this does and does not tell us
- It tells us whether the QVAR channel, wired as ECG on our own hardware, yields
  clean enough R-peaks for the existing PhysioNet-trained pipeline's R-peak stage
  to work at all, and whether the accelerometer (effort branch) is alive in the
  same files.
- It does NOT tell us anything about apnea detection accuracy/sensitivity on our
  hardware -- these are healthy subjects with no per-minute apnea labels, and per
  CLAUDE.md this data is validation-only and is never used for training.
