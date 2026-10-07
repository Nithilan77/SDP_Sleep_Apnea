# Nonlinear adversarial adaptor — attempt 1, negative result

**Date:** 2026-10-06
**Status:** Negative result — overfit to training nights, does not generalize.

## Motivation

CORAL (linear covariance alignment) failed to close the belt-vs-accelerometer
gap (see `results/adaptation/coral/FINDINGS.md`). This motivated a nonlinear,
trainable adaptor: a small residual MLP applied per-token, trained
adversarially against a domain discriminator, with a breathing-consistency
loss (autocorrelation at the measured respiratory lag) to prevent the
adversarial loss from collapsing tokens into unstructured noise, and an L2
residual penalty to anchor the adaptor near identity.

Train nights (8): S03, S06, S07, S08, S11, S12, S13, S15 (2,411 epochs total)
Val nights (2, held out completely from training): S04 (noisy), S10 (clean)

## Result

**Training was unstable, not convergent.** The residual-magnitude loss grew
from 0.02 to a peak of 13.2 and oscillated between ~1.6 and ~7.4 for the
remainder of training without settling. Discriminator confidence on real
belt tokens oscillated between 0.46 and 0.78 rather than stabilising near
the adversarial equilibrium of 0.5.

**Held-out domain AUROC: 1.0000** (belt vs adapted-accelerometer, on the two
validation nights never seen during training). Before any adaptation this
was ~0.93-0.95 (measured in the CORAL experiment, token space). The adaptor
made the held-out nights' tokens *more* separable from belt tokens, not
less — the opposite of the intended effect.

Flag rate on held-out nights did drop (S04: 97%->84%, S10: ~96%->91.7%),
but given the AUROC finding this is not credited as genuine improvement —
most likely the adaptor nudged these tokens in some direction that happens
to lower the classification head's output, without learning any
transferable belt-like structure.

## Diagnosis

This is textbook adversarial-training overfitting: with only 2,411 training
epochs across 8 nights, the adaptor and discriminator reached a training-set-
specific equilibrium — the adaptor learned to fool the discriminator using
quirks specific to those 8 nights' accelerometer traces, not a domain-general
transform. The held-out AUROC of 1.0 is strong, unambiguous evidence of this:
a genuinely successful adaptation would show *reduced* separability on
unseen nights, not increased.

The breathing-consistency loss (autocorrelation) did rise from ~0 to
~0.3-0.5 over training, but without independent ground truth this cannot be
credited as evidence of preserving real respiratory structure — it may
simply be tracking whatever periodicity was easiest to fit in the training
nights.

## Conclusion

**2,411 epochs across 8 short/medium nights is very likely insufficient
data for adversarial domain adaptation to generalise.** Rather than tune
hyperparameters on this same data (which would risk fitting the validation
nights too, defeating the purpose of holding them out), the honest next
question is whether this approach is viable at all with the amount of
in-house data currently available, or whether it specifically needs either:
(a) substantially more in-house accelerometer recording (more nights/subjects),
(b) a much more constrained adaptor (fewer parameters / stronger
regularisation) that cannot memorise 8 nights' idiosyncrasies, or
(c) a fundamentally different strategy not dependent on adversarial training
at this data scale (e.g. a purely consistency/reconstruction-based approach
with no discriminator, or revisiting whether the breath-hold protocol
should come first to get labelled events before any further adaptation work).

## Files

- `train_adaptor.py` — training script (ResidualAdaptor, Discriminator,
  ConsistencyReadout, autocorrelation-based consistency loss).
- `training_history.csv` / `training_curves.png` — full loss curves, all steps.
- `validation_flag_rate.csv` — per-held-out-night flag rate.
- `adaptor_checkpoint.pt` — trained weights (kept for reference; not
  recommended for use given the finding above).
- `summary.json` — numeric summary.
