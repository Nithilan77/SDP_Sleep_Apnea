# Frozen-model plausibility test (negative control) on in-house QVAR data

**Not an accuracy measurement.** These are healthy 19-20yo subjects with no
apnea labels (validation-only per CLAUDE.md Sec.4). The frozen deployable model
(`results/phase3_cnn/deployable/`, trained ONLY on PhysioNet) is run here purely
as a negative control: a model that transfers should report a LOW apnea-minute
rate on healthy subjects. A high rate is a red flag to investigate -- most likely
the R-peak-amplitude channel's domain gap documented in
`results/inhouse_validation/domain_gap/` -- not a finding about real apnea.

Overall predicted apnea-minute rate across 2027 minutes (9 subjects): **2-channel (RR+amplitude) = 52.8%**, **RR-only (amplitude neutralized) = 21.3%**.

## Per-subject predictions

| file                                    |   n_minutes |   apnea_pct_2ch |   mean_prob_2ch |   apnea_pct_rronly |   mean_prob_rronly |
|:----------------------------------------|------------:|----------------:|----------------:|-------------------:|-------------------:|
| user_0502-bala_ecg_000.txt              |           0 |          nan    |        nan      |             nan    |           nan      |
| user_0502-bala_ecg_001.txt              |           0 |          nan    |        nan      |             nan    |           nan      |
| user_0502-bala_ecg_002.txt              |          93 |            6.45 |          0.0671 |               4.3  |             0.1212 |
| user_1002-jayendran_ecg_000.txt         |           0 |          nan    |        nan      |             nan    |           nan      |
| user_1002-jayendran_ecg_001.txt         |         234 |           58.97 |          0.5957 |              11.11 |             0.1991 |
| user_1101hudson_ecg_000.txt             |         346 |           63.87 |          0.6382 |              65.61 |             0.6024 |
| user_1510hudson_ecg_000.txt             |          46 |           80.43 |          0.7996 |              82.61 |             0.7289 |
| user_1612bhavi_ecg_001.txt              |         168 |           34.52 |          0.3517 |               8.33 |             0.1538 |
| user_2406Hari(chest_chest & ecg_000.txt |          65 |           10.77 |          0.1125 |              30.77 |             0.3482 |
| user_aadhithiyaa_ecg_000.txt            |         314 |           51.59 |          0.5164 |               6.05 |             0.0896 |
| user_aniruth3103_ecg1_000.txt           |         349 |           42.12 |          0.4249 |              10.89 |             0.1398 |
| user_aniruth3103_ecg_000.txt            |           0 |          nan    |        nan      |             nan    |           nan      |
| user_kishor0404_ecg_000.txt             |         412 |           71.36 |          0.7113 |              10.92 |             0.1754 |

## How to read this
- `apnea_pct_2ch` / `mean_prob_2ch`: the full frozen model (RR + R-amplitude), exactly as trained.
- `apnea_pct_rronly` / `mean_prob_rronly`: same model and same RR channel, but the amplitude channel's raw value is replaced with the model's own training-mean amplitude before normalization -- so that channel carries exactly zero information (it normalizes to 0). This isolates what the RR channel alone is driving vs what the (likely out-of-distribution) amplitude channel is driving.
- If `apnea_pct_2ch` is much higher than `apnea_pct_rronly`, the amplitude channel is the dominant cause of any implausible (high) apnea rate -- consistent with the domain-gap analysis if that showed a large PhysioNet-vs-QVAR amplitude scale ratio.
- If both are similarly high, the RR channel itself (or the model's threshold/calibration in general) is also part of the story, not just amplitude scale.

## Specific observations (this run)

- **For 5 of 7 usable subjects** (bala_002, jayendran_001, bhavi_001, aadhithiyaa,
  aniruth1, kishor -- 6 of 7), `apnea_pct_rronly` drops sharply below `apnea_pct_2ch`
  (e.g. kishor 71.4% -> 10.9%, aadhithiyaa 51.6% -> 6.1%, jayendran 59.0% -> 11.1%).
  This is the expected signature of the amplitude channel being out-of-distribution:
  once neutralized, the model's apnea rate falls to a much more plausible range for
  healthy subjects. Consistent with the domain-gap analysis's ~108x amplitude and
  ~268x QRS-area scale mismatch.
- **2 exceptions -- both `hudson` recordings -- do NOT fit this pattern**:
  `user_1101hudson_ecg_000` (63.9% -> 65.6%) and `user_1510hudson_ecg_000`
  (80.4% -> 82.6%) stay high or get slightly worse with the amplitude channel
  neutralized. For these two, the RR channel ALONE is driving an implausibly high
  apnea rate -- not just the amplitude channel. This is a separate finding from the
  amplitude domain gap and needs its own investigation (e.g. check this subject's
  measured HR against quality_report.csv: both hudson files show unusually low mean
  HR, 44.7 and 45.8 bpm -- close to the low end of the "PLAUSIBLE" band -- worth
  checking whether R-peak detection is under-detecting beats or missing peaks in a
  way that distorts the RR sequence itself, independent of amplitude scale).
- **1 inversion**: `user_2406Hari(chest_chest & ecg_000` goes the OTHER way
  (10.8% 2-channel -> 30.8% RR-only) -- here the amplitude channel was pulling the
  prediction down, not up. This subject should not be read as "the amplitude channel
  is fine for them"; it just means the failure mode isn't uniform across subjects,
  which is itself evidence that the amplitude channel's behavior on QVAR data is
  essentially unpredictable/out-of-distribution noise rather than a consistent bias
  that could be corrected with a single rescaling factor.
- **4 of 13 plausible files produced 0 usable minutes** (bala_000, bala_001,
  jayendran_000, aniruth_000) -- all under ~32s duration, too short to form even one
  60-second window at the MIN_BEATS_PER_WINDOW threshold. Expected, not a model
  problem.

## Cross-reference
See `results/inhouse_validation/domain_gap/README.md` for the measured PhysioNet-vs-QVAR amplitude and QRS-area scale ratios that this test's RR-only vs 2-channel comparison is meant to explain.
