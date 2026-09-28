# ECG feature-extraction comparison study -- master results

**Date:** 2026-09-28
**Protocol:** all 8 methods evaluated on the identical 35-fold leave-one-subject-out
(LOSO) split used throughout this project (see `docs/PROGRESS.md` section 5) --
one Apnea-ECG record held out per fold, never seen in training within that fold.
Methods 3/4/5/6/7/8 are further restricted to the exact same (record, minute)
set Phase 2/3 used (16,949 minutes; a handful fewer for methods 4/5 where a
few extra minutes lack usable frequency-domain or entropy features -- see each
method's own note). All 8 per-minute accuracies are well below the project's
0.95 leakage-suspicion threshold; none triggered it.

## Master table

| # | Method | Family | Model | Acc | Sens | Spec | Prec | F1 | Rec-Acc |
|---|---|---|---|---|---|---|---|---|---|
| 1 | RR-intervals | time | 1D-CNN | 0.775 | 0.653 | 0.851 | 0.732 | 0.691 | 0.800 |
| 2 | RR + R-peak amplitude (**adopted 81.2% model**) | time | 1D-CNN | **0.812** | 0.756 | 0.846 | 0.754 | 0.755 | 0.800 |
| 3 | Time-domain HRV bank | time | logistic regression | 0.702 | 0.620 | 0.754 | 0.611 | 0.615 | 0.743 |
| 4 | Frequency-domain HRV (VLF/LF/HF/LF:HF) | frequency | logistic regression | 0.672 | 0.513 | 0.772 | 0.584 | 0.546 | 0.743 |
| 5 | Nonlinear/Poincare + sample entropy | nonlinear | logistic regression | 0.707 | 0.673 | 0.728 | 0.617 | 0.644 | 0.743 |
| 6 | RR + QRS-area (EDR proxy) | time | 1D-CNN | 0.798 | 0.687 | 0.868 | 0.764 | 0.724 | **0.829** |
| 7 | CWT scalogram of RR series | image | 2D-CNN | 0.749 | 0.491 | 0.911 | 0.774 | 0.601 | 0.771 |
| 8 | Spectrogram of raw ECG | image | 2D-CNN | 0.534 | 0.673 | 0.448 | 0.432 | 0.526 | 0.800 |

Methods 3/4/5 (scalar HRV banks) each show the **best of three classifiers**
run under the identical LOSO discipline -- 1D-CNN, logistic regression, and a
small MLP (`src/ecg/cnn_model.py::FeatureMLP`). Logistic regression won all
three; full three-way comparisons are in each method's
`classifier_comparison.md` (`results/feature_study/method{3,4,5}_*/`).

## Key findings

1. **Real within-minute time series beats aggregated scalars.** The two
   methods that feed genuine per-beat sequences to the CNN (1, 2, 6) are the
   only ones to clear 0.79 accuracy. Every method that collapses a minute
   into a handful of numbers first (3, 4, 5) -- regardless of classifier --
   tops out in the 0.67-0.71 range, close to Phase 2's original classical
   floor (0.711).

2. **R-peak amplitude isn't uniquely special as an EDR proxy.** Method 6
   (QRS-area, a whole-complex integral instead of a single peak sample)
   comes within 1.4 points of method 2's accuracy and actually posts the
   **best per-recording accuracy in the study** (29/35 = 0.829). Different
   EDR extraction, same underlying respiration-modulation signal.

3. **Classifier choice matters a lot for weak/noisy features, less for
   strong ones.** Forcing scalar HRV banks through the 1D-CNN cost 7-22
   accuracy points depending on the bank (worst for method 4, where the CNN's
   specificity collapsed to 0.293 and logistic regression recovered it to
   0.772). For the RR/QRS-area sequence methods (1, 2, 6), the CNN is the
   right tool and no fairness question arises.

4. **Frequency-domain HRV is the weakest bank, as flagged in advance.** VLF
   needs ~5.5 minutes of data to resolve and LF is borderline at a 60s
   window (`src/ecg/hrv_bank_features.py` documents this before the numbers
   came in) -- method 4 is the worst performer of the non-image methods
   under every classifier.

5. **Image representations trade sensitivity for specificity, hard.**
   Both 2D-CNN methods (7, 8) show the sharpest sensitivity/specificity
   splits in the study (7: 0.491/0.911; 8: 0.673/0.448 the other way).
   Method 7 (CWT of RR) is a reprojection of the same information as method 1
   and lands in a similar overall ballpark; method 8 (raw ECG spectrogram)
   is the only method not derived from R-peaks at all, and is the weakest
   per-minute performer in the whole study despite a respectable
   per-recording accuracy (28/35 = 0.800).

## Caveats / honest limitations

- **35 subjects is a small dataset** for training 2D-CNNs on 3,000-6,000-pixel
  images per method (7, 8) as much as it was for the original 1D-CNN --
  taken more seriously here since the image methods have more parameters in
  their first conv layer's receptive field per input pixel.
- **Frequency-domain features (method 4) are flagged as unreliable, not just
  weak** -- the window-length mismatch (needs 5+ minutes, has 1) means
  vlf_power in particular is closer to noise than a real physiological
  estimate. Reported for completeness because it was requested, not because
  it should be trusted.
- **The scalar-HRV-bank-as-1D-CNN-input design (methods 3/4/5's CNN column)
  was a deliberate but debatable adaptation** -- treating an unordered
  feature vector as a "sequence" for architecture reuse. The three-classifier
  comparison exists specifically because that adaptation was flagged as
  potentially unfair; logistic regression winning all three banks supports
  the concern, and the master table reports the winner per method_bank, not
  the CNN's number, to avoid understating what these features can do.
- **No new leakage checks were needed** -- every method's per-minute accuracy
  sits well under the project's 0.95 suspicion threshold, and the two
  highest-sensitivity/lowest-specificity and vice-versa splits (methods 4,
  7, 8) are explained by identifiable, documented causes (window-length
  limits, image/classifier mismatch), not silently accepted.

## Where each result lives

```
results/feature_study/
  master_results.csv / master_results.md      This table.
  hrv_bank_features.csv                        Cached per-minute time/freq/nonlinear features (methods 3-5).
  method3_time_domain/{cnn,logreg,mlp}/        Method 3 predictions per classifier + classifier_comparison.md.
  method4_freq_domain/{cnn,logreg,mlp}/        Method 4, same structure.
  method5_nonlinear/{cnn,logreg,mlp}/          Method 5, same structure.
  method6_qrsarea_edr/                         Method 6 predictions + cached sequences.
  method7_cwt_scalogram/                       Method 7 predictions + cached scalograms.
  method8_ecg_spectrogram/                     Method 8 predictions + cached spectrograms.

results/phase3_cnn/{per_minute,final_per_minute}_predictions.csv   Methods 1 and 2 (pre-existing Phase 3 results, reused unmodified).
```
