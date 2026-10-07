# Nonlinear adversarial adaptor — attempt 2 (constrained), corrected measurement

**Date:** 2026-10-06
**Status:** Negative result — no effect (not overfitting this time, just no signal).

## Change from attempt 1

Attempt 1 (`results/adaptation/adaptor/FINDINGS.md`) showed active overfitting:
held-out domain AUROC got *worse* than pre-adaptation (1.0 vs baseline
~0.93-0.95), with unstable, divergent training curves. Attempt 2 used a much
smaller adaptor (hidden dim 32 vs 128), a 6x stronger residual anchor penalty
(0.3 vs 0.05), a slower discriminator learning rate relative to the adaptor
(previously equal), and explicit early-stopping on held-out AUROC tracked
during training (not just checked at the end).

**Bug found and fixed mid-experiment:** the held-out AUROC measurement
originally fit a logistic regression and scored it on the *same* data
(resubstitution) rather than a disjoint split. This gave a spuriously
perfect AUROC=1.0 even on the *identity* (no-op) adaptor as a sanity check
— which caught the bug before any further numbers were trusted. Fixed to
use a proper random train/test split inside the measurement function
before rerunning. All numbers below are post-fix and trustworthy.

## Result

Pre-adaptation (identity) held-out domain AUROC: **0.9984**
Best held-out domain AUROC achieved during training: **0.9981**
Final (at early stop, step 240): **0.9997**

Training auto-stopped at step 240 per the early-stop guardrail (held-out
AUROC > 0.94), but the AUROC never meaningfully moved from baseline in
either direction — the difference (0.9984 -> 0.9981) is within noise.
Downstream flag rate on held-out nights: S04 85.8%, S10 92.3% — essentially
unchanged from both the pre-adaptation baseline (~93-97%) and attempt 1's
numbers, despite attempt 1's AUROC being actively worse.

**Note on cross-experiment comparison:** this run's pre-adaptation baseline
(0.9984) differs from the CORAL experiment's token-space baseline (0.9542,
see `results/adaptation/coral/FINDINGS.md`). These are not directly
comparable — this measurement uses only the 2 held-out nights (S04, S10)
with a single random 50/50 split, while CORAL's used `GroupKFold` across
tokens from all nights. Different night subsets and different CV protocol.
Flagged here rather than left unexplained.

## Interpretation

Unlike attempt 1 (active overfitting — the adaptor learned a transform that
actively increased separability on unseen nights), attempt 2's constrained
adaptor essentially **did nothing** — training stayed stable, the residual
penalty kept it near-identity (alpha only grew 0.050 -> 0.057 over 240
steps before early stop), but it never found a direction that reduced
belt-vs-accelerometer separability either.

Taken together with attempt 1, this traces out a real trade-off: enough
adaptor capacity to change the representation (attempt 1) overfits on 8
nights of data; little enough capacity to stay stable (attempt 2) does not
have enough expressive power to close a gap this large with gradients from
this much data.

## Across all adaptation attempts so far

| Attempt | Mechanism | Train-domain effect | Held-out generalization |
|---|---|---|---|
| CORAL, pooled embedding | linear | AUROC 0.994->0.218 (strong) | Broke the model's forward pass (cross-attention bypassed); flag rate worse |
| CORAL, token level | linear | AUROC 0.954->0.934 (weak) | Consistent with weak train effect — no real improvement |
| Adversarial adaptor v1 | nonlinear, large | AUROC pattern suggests train-set-specific fit | AUROC 1.0 (worse than baseline) — overfit |
| Adversarial adaptor v2 | nonlinear, constrained | ~no change (stable but inert) | ~no change (0.9984->0.9981) |

Four distinct mechanisms, four negative or null results. This is now a
consistent pattern, not a single failed attempt.

## Conclusion

**Representation-space manipulation of the effort encoder's tokens — linear
or nonlinear, loosely or tightly constrained — has not closed the
belt-vs-accelerometer gap with the data currently available (2,411 training
epochs across 8 nights, evaluated on 2 held-out nights).** This is not
strong evidence the gap is permanently unclosable, but it is evidence that
it will not yield to these representation-adaptation approaches at this
data scale without either substantially more in-house recording or a
different strategy entirely (e.g., labelled breath-hold events as a direct
supervisory signal rather than unsupervised domain confusion, or
retraining/fine-tuning the effort encoder itself rather than adapting its
output).

## Files

- `train_adaptor_v2.py` — corrected measurement, constrained adaptor,
  early-stopping guardrail.
- `training_history.csv` / `training_curves.png` — loss and held-out AUROC
  curves (stable, bounded, stopped at step 240).
- `validation_flag_rate.csv` — per-night flag rate with best checkpoint.
- `adaptor_checkpoint_best.pt` — trained weights (reference only; not
  recommended for use, no demonstrated benefit).
- `summary.json` — numeric summary, verdict field.
