# CORAL domain adaptation — attempt and negative result

**Date:** 2026-10-06
**Status:** Negative result, informative — motivates a nonlinear adaptor next.

## Motivation

Step 1 (`embed_gap.py`) found that the MESA-belt-trained effort encoder's
embedding space shows near-perfect belt-vs-accelerometer separability
(AUROC 0.997 in GAP+GMP pooled space), but the *mean* shift was small
(0.16 SD across 50 PCA dims). PC1 (82.4% of variance) showed accelerometer
embeddings collapsed into a narrow sub-region of the space belts occupy.
This pattern — large separability, small mean shift — suggested a
**covariance misalignment** rather than a simple offset, which is exactly
what CORAL (Sun et al. 2016, "Return of Frustratingly Easy Domain
Adaptation") is designed to fix: a closed-form, label-free linear transform
that matches second-order statistics (covariance) between domains.

## What was tried

**Attempt 1 — CORAL at pooled (GAP+GMP, 128-dim) embedding level.**
Aligned covariance of accelerometer embeddings to belt embeddings,
bypassing the model's cross-attention and feeding the adapted embedding
directly into a concat-style head.
- Domain AUROC: 0.9938 -> 0.2177 (near-perfect alignment)
- BUT in-house flag rate: ~93% -> 100% (got worse)
- **Diagnosis:** the classification head was trained on cross-attended
  tokens, not raw pooled embeddings. Bypassing cross-attention fed the
  head a representation it never saw during training — the alignment
  succeeded but broke the model's actual computation path.

**Attempt 2 (corrected) — CORAL at raw token level (64-dim, pre-attention),
  preserving the full cross-attention + head pipeline.**
Aligned covariance of each of the 75 per-epoch accelerometer tokens to
belt tokens, then ran the model exactly as trained (cross-attention,
GAP+GMP, head) on the adapted tokens.
- Domain AUROC: 0.9542 -> 0.9340 (regularisation sweep 0.01-10.0, best at reg=1.0)
- Flag rate: ~93% -> 96.4% median (no real improvement, within noise)

## Why it failed

Pooling (mean+max over 75 tokens) collapses a sequence into a single
summary statistic — a near-Gaussian blob that a linear covariance
transform can align well. Raw tokens carry **temporal/positional
structure** (each token encodes a different part of the 150 s window).
The belt-vs-accelerometer gap at the token level isn't just differently-
scaled — it's a different *nonlinear relationship* between raw signal
shape and what the encoder reads into each token position. This is
consistent with the epoch-level spectral characterisation from
`run_canet.py` (`domain_gap/feature_gap_summary.csv`): spectral entropy,
zero-crossing rate, and breathing-rate IQR all differ in ways that aren't
simple linear/scale differences between belt and accelerometer effort
signals.

**Conclusion: a linear transform (at either pooled or token level) cannot
close this gap.** The encoder's belt-trained token representations encode
sensor-specific morphology that CORAL's linear assumption doesn't capture.

## What this motivates

A **nonlinear, trainable adaptor** is justified — not as a first resort,
but because the simpler linear approach was tried first and concretely
failed. Candidate: a small MLP on top of the resp encoder's tokens,
trained with:
1. an adversarial domain-confusion loss (discriminator: belt vs
   accelerometer tokens) to close the measured 0.93-0.95 AUROC gap,
2. a breathing-consistency loss anchored to the confirmed real spectral
   peak near 0.2-0.25 Hz in the accelerometer signal (§7c of
   `docs/PROGRESS.md`), to prevent the adversarial loss from collapsing
   the representation to noise,
3. an anchor loss keeping belt-domain token outputs close to the
   original (frozen) encoder's belt outputs, to preserve what CANet
   already learned on MESA.

No labels required for any of this — same label-free constraint as
everything else in the in-house validation track.

## Files

- `coral_adaptor.py` — the (superseded) GAP+GMP-level attempt's code is
  retained in git history; current file implements the corrected
  token-level attempt.
- `coral_params_token.npz` — fitted W, src_mean, tgt_mean (reg=1.0);
  kept for reference, not used going forward.
- `reg_sweep_token.png` — regularisation sweep, token-space AUROC.
- `flag_rate_token_coral.csv` — per-night flag rate after token CORAL.
- `summary.json` — numeric summary of this experiment.
