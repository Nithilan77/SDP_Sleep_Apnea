# MESA subset audit

Subjects audited (EDF+XML complete): **250** (errors: 0). Usable under the strict definition: **220**.

## Usable-subject funnel

| criterion (cumulative)                                    |   subjects |
|:----------------------------------------------------------|-----------:|
| complete subjects (EDF+XML)                               |        250 |
| ECG + SpO2 + Thor + Abdo all present                      |        250 |
| XML has sleep-stage events                                |        250 |
| sleep time >= 4 h                                         |        224 |
| both belts flat_frac < 0.5                                |        222 |
| both belts breathing visible in >= 50% epochs             |        220 |
| scored respiratory events present (>=1 apnea or hypopnea) |        220 |

## Signal presence and native sampling rate

- `ecg`: present in 250/250; fs values [256.0]
- `thor`: present in 250/250; fs values [32.0]
- `abdo`: present in 250/250; fs values [32.0]
- `spo2`: present in 250/250; fs values [1.0]
- recording duration h: median 10.50 (min 7.86, max 30.70); scored sleep h: median 6.09 (min 1.85)

## Scored events

Totals across audited subjects: obstructive 6071, central 849, mixed 4, hypopnea 27746.
Subjects with >=1 central apnea: 106; with >=5 central: 30; with >=1 obstructive: 208; with any apnea (O/C/M): 215.

All respiratory event concepts seen in the XMLs (anything unexpected here is NOT parsed into a subtype):

| concept                                   |   count |
|:------------------------------------------|--------:|
| SpO2 desaturation|SpO2 desaturation       |   82278 |
| Hypopnea|Hypopnea                         |   27746 |
| Unsure|Unsure                             |   20063 |
| Obstructive apnea|Obstructive Apnea       |    6071 |
| SpO2 artifact|SpO2 artifact               |    3176 |
| Central apnea|Central Apnea               |     849 |
| Mixed apnea|Mixed Apnea                   |       4 |
| Respiratory effort related arousal|RERA   |       2 |
| Respiratory artifact|Respiratory artifact |       1 |

- Epoch class balance, all epochs (30 s, n=325669): apnea 3.3%, hypopnea 11.0%, resp_event 14.3%
- Epoch class balance, sleep epochs only (30 s, n=177745): apnea 5.7%, hypopnea 19.8%, resp_event 25.4%

Note: 'apnea' (obstructive/central/mixed) is rare next to hypopnea in MESA; an apnea-only label is a highly imbalanced target.

## AHI distribution

- XML-derived, all scored events / scored sleep h: n=250, median 21.6, IQR 13.5-32.2, max 81.4; <5 normal 13, 5-15 mild 59, 15-30 moderate 111, >=30 severe 67
- XML-derived, desat(>=3%)-supported: n=250, median 7.2, IQR 2.8-16.3, max 80.6; <5 normal 93, 5-15 mild 88, 15-30 moderate 53, >=30 severe 16
- NSRR official ahi_a0h3a (3% or arousal): n=250, median 20.2, IQR 10.8-37.3, max 111.3; <5 normal 24, 5-15 mild 67, 15-30 moderate 74, >=30 severe 85
- NSRR official ahi_a0h4 (4%): n=250, median 10.8, IQR 3.6-23.5, max 101.3; <5 normal 80, 5-15 mild 76, 15-30 moderate 49, >=30 severe 45

Cross-check vs NSRR official AHI (n=250): Pearson r(xml_all, a0h3)=0.714, r(xml_all, a0h3a)=0.748, r(xml_desat3, a0h3)=0.848. Median ratio xml_all/a0h3a = 1.03; xml_desat3/a0h3 = 0.50.
Sleep-time check (stage-derived vs NSRR slpprdp5): median |diff| 0.0 min, max 1.0 min (n=250).

## Thoracic vs abdominal belt quality

|                                |   thor |   abdo |
|:-------------------------------|-------:|-------:|
| flat_frac median               |  0     |  0.002 |
| good_epoch_frac median         |  0.857 |  0.887 |
| good_epoch_frac<0.5 (subjects) |  3     |  4     |
| clip_frac median               |  0     |  0     |
| breath rate median (bpm)       | 14     | 14     |

Per subject, cleaner belt by breathing-visible epoch fraction: Thor 65, Abdo 183, tie 2. Median |r| between belts (0.1-0.6 Hz, per epoch): 0.82.

## Apnea prevalence sanity check (elderly cohort)

Subset age at sleep exam: median 70 (range 55-92); male fraction 0.50.
| group                         | hypopnea rule   |    n | AHI>=5   | AHI>=15   | AHI>=30   |   median AHI |
|:------------------------------|:----------------|-----:|:---------|:----------|:----------|-------------:|
| audited subset                | 3% or arousal   |  250 | 90%      | 64%       | 34%       |         20.2 |
| audited subset                | 4% desat        |  250 | 68%      | 38%       | 18%       |         10.8 |
| full MESA sleep cohort (NSRR) | 3% or arousal   | 2057 | 90%      | 59%       | 30%       |         18.4 |
| full MESA sleep cohort (NSRR) | 4% desat        | 2057 | 66%      | 34%       | 15%       |          9.1 |

Read-out: prevalence depends heavily on the hypopnea rule (3%-or-arousal is much more liberal than 4%), so quote the rule with any number. The subset row vs the full-cohort row (same rule) shows whether the sampled subset is representative of MESA; the full-cohort row is NSRR's own reported AHI, i.e. the reference this audit is checked against.

## Usable subset

n=220. Epoch balance (sleep epochs): apnea 5.5%, hypopnea 19.6%, resp_event 25.1%.
Usable subjects with >=5 central apneas: 29; with >=5 obstructive apneas: 128.
