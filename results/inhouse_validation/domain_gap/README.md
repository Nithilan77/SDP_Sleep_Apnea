# Domain-gap analysis: PhysioNet vs in-house QVAR

**Distribution comparison only -- no accuracy/sensitivity claims.** Groups:
`physionet` = all 35 labelled Apnea-ECG records (the deployable model's training
distribution). `inhouse_plausible` = the 13 in-house files flagged PLAUSIBLE in
the quality report. `inhouse_noisy` = user_0802-anagesh_ecg_000.txt (flagged
NOISY_ECG), reported separately. user_1612bhavi_ecg_000.txt (too short, 1 beat)
is excluded entirely.

## Summary table

| group             | metric        |       n |        mean |         std |     median |        q25 |         q75 |        iqr |
|:------------------|:--------------|--------:|------------:|------------:|-----------:|-----------:|------------:|-----------:|
| physionet         | rr_interval_s | 1125797 |   0.9081    |   0.596735  |   0.9      |  0.8       |   0.99      |  0.19      |
| inhouse_plausible | rr_interval_s |  114485 |   1.06608   |   0.230782  |   1.0669   |  0.904794  |   1.18457   |  0.279778  |
| inhouse_noisy     | rr_interval_s |   32959 |   0.823041  |   0.276615  |   0.756053 |  0.70985   |   0.814857  |  0.105007  |
| physionet         | hr_inst_bpm   | 1125797 |  68.1352    |  11.7978    |  66.6667   | 60.6061    |  75         | 14.3939    |
| inhouse_plausible | hr_inst_bpm   |  114485 |  59.0534    |  13.8383    |  56.2378   | 50.6512    |  66.3135    | 15.6622    |
| inhouse_noisy     | hr_inst_bpm   |   32959 |  77.4953    |  15.0706    |  79.3595   | 73.6326    |  84.5249    | 10.8924    |
| physionet         | r_amplitude   | 1125797 |   1.29548   |   0.893458  |   1.01318  |  0.643058  |   1.94419   |  1.30113   |
| inhouse_plausible | r_amplitude   |  114485 | 140.535     | 279.877     | 116.649    | 94.5508    | 166.237     | 71.6866    |
| inhouse_noisy     | r_amplitude   |   32959 |  40.852     | 222.042     |  32.3454   |  9.91109   |  57.5171    | 47.606     |
| physionet         | qrs_area      | 1125797 |   0.0470207 |   0.0290861 |   0.040382 |  0.0260986 |   0.0703768 |  0.0442782 |
| inhouse_plausible | qrs_area      |  114485 |  12.603     |  18.7551    |  10.5277   |  7.90108   |  13.954     |  6.05288   |
| inhouse_noisy     | qrs_area      |   32959 |   9.97968   |  15.1448    |   9.73024  |  6.39268   |  11.7267    |  5.33402   |

## Reading the gap

- **RR-interval (s)**: PhysioNet mean=0.9081 (std=0.5967) vs in-house-plausible mean=1.066 (std=0.2308) -> mean ratio 1.17x, std ratio 0.39x.
- **Instantaneous HR (bpm)**: PhysioNet mean=68.14 (std=11.8) vs in-house-plausible mean=59.05 (std=13.84) -> mean ratio 0.87x, std ratio 1.17x.
- **R-peak amplitude (cleaned-signal units)**: PhysioNet mean=1.295 (std=0.8935) vs in-house-plausible mean=140.5 (std=279.9) -> mean ratio 108.48x, std ratio 313.25x.
- **QRS-area (cleaned-signal-units * s)**: PhysioNet mean=0.04702 (std=0.02909) vs in-house-plausible mean=12.6 (std=18.76) -> mean ratio 268.03x, std ratio 644.81x.

## Plain-language verdict

- **RR-interval**: mean ratio 1.17x (PhysioNet subjects are older with clinical apnea/borderline/control mix; our subjects are healthy 19-20yo, so some HR difference is expected and NOT itself a hardware artifact). Distribution shape (std/IQR relative to mean) is the more informative check -- see the histogram (`hist_rr_interval_s.png`) and table above for whether the spread is comparable.
- **R-peak amplitude**: mean ratio 108.48x. This is the flagged transfer risk from the deployable model's README -- PhysioNet amplitude is on a calibrated-ECG scale, QVAR is an uncalibrated electrostatic-sensor scale after the same `neurokit2` cleaning. A large ratio here (order-of-magnitude or more) means the frozen model's `norm_stats.json` (fit on PhysioNet's amplitude scale) will NOT put QVAR amplitudes in the range the model was trained on -- expect the amplitude channel to look like an extreme, out-of-distribution outlier to the model regardless of any true respiration modulation in it.
- **QRS-area**: same PhysioNet-vs-QVAR scale question as amplitude, since it's built from the same cleaned-signal units. If this ratio is similarly large, QRS-area is NOT an automatic fix for the amplitude problem -- it would need its own re-normalization against QVAR's own scale, not a drop-in replacement using PhysioNet's stats.
- **Bottom line**: RR-interval, being time-based (seconds), is the one channel this project's rule 6 expects to transfer without rescaling. Amplitude and QRS-area are both raw-signal-unit metrics and inherit whatever scale mismatch exists between a calibrated ECG amplifier and QVAR's electrostatic front-end -- treat any large ratio above as confirming, not just flagging, that risk.
- **Noisy-file effect**: compare `inhouse_noisy` (anagesh) against `inhouse_plausible` in the table -- if noisy-file std/IQR is visibly wider for the same metric, that's the 1.09% implausible-RR translating into distribution spread, isolated from the plausible group's own numbers.
