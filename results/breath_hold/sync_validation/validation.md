# Sync-tap detector validation (final logic of `src/breath_hold/sync.py`)

Session-1 noise: robust sigma 0.0023 g (z = 25 -> 0.058 g; z = 60 -> 0.140 g; z = 100 -> 0.233 g); fs 243.10 Hz; 1796 s.

Shortcut check: baseline z via high-pass of the stored 3-axis array vs the production path `sync.analyse`: max |dz| = 0.00e+00.

## False alarms on the real recording (no deliberate taps)

Candidate pairs (two sharp peaks, z>=25, 0.25-1.2 s apart): 13. **Accepted markers anywhere: 0**; start window (first 60 s): none; end window (last 120 s): none.

| t1 (s) | z1 / z2 | gap (s) | dominance | accepted | reason(s) rejected |
|---|---|---|---|---|---|
| 45.7 | 25 / 64 | 1.19 | 1.10x | False | weak (<60 sigma); not dominant (<2x) |
| 291.4 | 25 / 28 | 1.20 | 0.13x | False | not isolated; weak (<60 sigma); not dominant (<2x) |
| 292.6 | 28 / 38 | 0.26 | 0.15x | False | not isolated; weak (<60 sigma); not dominant (<2x) |
| 292.9 | 38 / 187 | 0.48 | 1.38x | False | amplitude ratio; not isolated; weak (<60 sigma); not dominant (<2x) |
| 293.4 | 187 / 26 | 0.79 | 0.68x | False | amplitude ratio; not isolated; weak (<60 sigma); not dominant (<2x) |
| 1074.6 | 26 / 27 | 0.44 | 0.80x | False | weak (<60 sigma); not dominant (<2x) |
| 1075.0 | 27 / 31 | 1.07 | 0.84x | False | not isolated; weak (<60 sigma); not dominant (<2x) |
| 1664.0 | 46 / 61 | 0.40 | 0.46x | False | weak (<60 sigma); not dominant (<2x) |
| 1665.8 | 50 / 73 | 0.62 | 0.49x | False | not isolated; weak (<60 sigma); not dominant (<2x) |
| 1666.4 | 73 / 28 | 0.45 | 0.28x | False | not isolated; weak (<60 sigma); not dominant (<2x) |
| 1666.9 | 37 / 62 | 0.42 | 0.36x | False | not isolated; weak (<60 sigma); not dominant (<2x) |
| 1667.3 | 62 / 50 | 0.64 | 0.49x | False | not isolated; weak (<60 sigma); not dominant (<2x) |
| 1668.3 | 102 / 95 | 0.79 | 1.29x | False | not isolated; not dominant (<2x) |

## Detection of injected double-taps (final logic; 200 trials per cell; quiet sites in the search windows)

accepted = passes ALL rules incl. >=60 sigma and >=2x dominance; candidate = a sharp pair was seen but may have been rejected (e.g. too weak). Timing error = detected second-tap time minus the TRUE peak sample of the injected tap (largest-magnitude sample of the unfiltered injected waveform; independent of the detector's filtered peak; one sample = 4.1 ms): median and 95th percentile of |error|, and mean signed bias. The last column compares to the injection START (onset), a looser reference.

| window | nominal peak (g) | candidate pair seen | **accepted marker** | median weaker-tap z (accepted) | |timing err| median (ms) | p95 (ms) | mean bias (ms) | median vs inject-onset (ms) |
|---|---|---|---|---|---|---|---|---|
| start | 0.02 | 0% | **0%** | nan | nan | nan | +nan | nan |
| start | 0.04 | 0% | **0%** | nan | nan | nan | +nan | nan |
| start | 0.08 | 0% | **0%** | nan | nan | nan | +nan | nan |
| start | 0.12 | 9% | **0%** | nan | nan | nan | +nan | nan |
| start | 0.20 | 88% | **0%** | nan | nan | nan | +nan | nan |
| start | 0.30 | 100% | **4%** | 70 | 0.0 | 0.0 | +0.0 | 0.0 |
| start | 0.40 | 100% | **46%** | 76 | 0.0 | 0.0 | +0.0 | 0.0 |
| start | 0.50 | 100% | **90%** | 85 | 0.0 | 0.0 | +0.1 | 0.0 |
| start | 0.60 | 100% | **97%** | 97 | 0.0 | 0.0 | +0.1 | 0.0 |
| start | 0.80 | 100% | **100%** | 130 | 0.0 | 0.0 | +0.1 | 0.0 |
| start | 1.20 | 100% | **100%** | 191 | 0.0 | 0.0 | +0.0 | 0.0 |
| end | 0.02 | 0% | **0%** | nan | nan | nan | +nan | nan |
| end | 0.04 | 0% | **0%** | nan | nan | nan | +nan | nan |
| end | 0.08 | 0% | **0%** | nan | nan | nan | +nan | nan |
| end | 0.12 | 8% | **0%** | nan | nan | nan | +nan | nan |
| end | 0.20 | 80% | **0%** | nan | nan | nan | +nan | nan |
| end | 0.30 | 100% | **12%** | 65 | 0.0 | 0.0 | +0.0 | 0.0 |
| end | 0.40 | 100% | **63%** | 71 | 0.0 | 0.0 | +0.0 | 0.0 |
| end | 0.50 | 100% | **90%** | 80 | 0.0 | 0.0 | +0.0 | 0.0 |
| end | 0.60 | 100% | **99%** | 99 | 0.0 | 0.0 | +0.0 | 0.0 |
| end | 0.80 | 100% | **100%** | 127 | 0.0 | 0.0 | +0.0 | 0.0 |
| end | 1.20 | 100% | **100%** | 199 | 0.0 | 0.0 | +0.0 | 0.0 |

## Pre-session tap check (3 trial double-taps >= 20 s apart; PASS needs 3 accepted)

| nominal peak (g) | check PASS rate (100 trials) |
|---|---|
| 0.02 | 0% |
| 0.04 | 0% |
| 0.08 | 0% |
| 0.12 | 0% |
| 0.20 | 0% |
| 0.30 | 0% |
| 0.40 | 22% |
| 0.50 | 76% |
| 0.60 | 94% |
| 0.80 | 97% |
| 1.20 | 93% |

## Stress: taps injected 6 s before the real transient at ~46.8 s (z = 64) inside the start window

| nominal peak (g) | accepted marker (200 trials) | note |
|---|---|---|
| 0.20 | 0% | accepted only if both taps are >= 60 sigma and >= 2x the real transient |
| 0.40 | 0% | accepted only if both taps are >= 60 sigma and >= 2x the real transient |
| 0.80 | 56% | accepted only if both taps are >= 60 sigma and >= 2x the real transient |

## Lap-alignment self-test (synthetic laps, known truth)

| case | start tap (s) | drift added (s over 25 min) | recovered max abs error of all 20 lap times (ms) | anchor rule applied |
|---|---|---|---|---|
| offset 17.3, device clock x1.00000 | 17.3 | +0.00 | 0.00 | start only (|drift| < 0.5 s) |
| offset 17.3, device clock x1.00014 | 17.3 | +0.20 | 200.00 | start only (|drift| < 0.5 s) |
| offset 17.3, device clock x1.00136 | 17.3 | +2.00 | 0.00 | start+end (interpolated) |
| offset 8.9, device clock x0.99796 | 8.9 | -3.00 | 0.00 | start+end (interpolated) |
