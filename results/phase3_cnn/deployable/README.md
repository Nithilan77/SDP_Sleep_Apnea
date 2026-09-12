# Deployable ECG-branch model

**This is a deployment artifact, not an evaluation.** It's the model to run on our own QVAR recordings once the SensorTile hardware is back — it is **not** how we know how well the ECG branch performs.

## What this is

Phase 3's adopted final config (2-channel RR-interval + R-peak amplitude input, default class-balanced loss weighting — see `results/phase3_cnn/README.md`), trained **once on all 35 labelled PhysioNet records with no held-out test set.** Every LOSO run in this project trains a fresh model per fold and throws the weights away; this is the first (and only) time we've kept a set of weights.

**Trained only on PhysioNet Apnea-ECG — never on our own recordings**, per the project's core constraint.

## Files

- `model_state_dict.pt` — PyTorch `state_dict` for `RRCNN(seq_len=60, n_channels=2)` (see `src/ecg/cnn_model.py`).
- `norm_stats.json` — per-channel mean/std used to normalize inputs before feeding the model (computed over all 16,949 training minutes). **Must be applied to any new input before inference** — the model was never shown unnormalized data.
- `config.json` — architecture/training config, plus the LOSO numbers from Phase 3 for reference (see caveat below).

## How to use it

```python
import json, torch
from src.ecg.cnn_model import RRCNN, predict_cnn

model = RRCNN(seq_len=60, n_channels=2)
model.load_state_dict(torch.load("results/phase3_cnn/deployable/model_state_dict.pt"))

norm = json.load(open("results/phase3_cnn/deployable/norm_stats.json"))
mu = np.array(norm["mean"]).reshape(1, 2, 1)
sd = np.array(norm["std"]).reshape(1, 2, 1)

# X: (n_minutes, 2, 60) -- channel 0 = RR-interval sequence (seconds),
# channel 1 = R-peak amplitude sequence, each resampled onto a 60-point grid
# exactly as src/ecg/rr_amp_sequence.py does it.
X_norm = (X - mu) / sd
probs = predict_cnn(model, X_norm)  # P(apnea) per minute
```

## What "performance" means for this model

We did **not** measure accuracy/sensitivity/specificity for this exact model — there's no held-out data left to measure it on, by design (this run used every labelled minute to make the deployed model as good as possible). The expected-performance numbers in `config.json` are Phase 3's LOSO cross-validation result (**accuracy 0.812, sensitivity 0.756, specificity 0.846**) for the same architecture and training recipe, which is the honest estimate of how a model built this way generalizes to an unseen subject. This model's own training-set fit was 0.834 accuracy — expected to look a bit better than LOSO since it has seen every minute it was checked against, and not a claim about how it'll do on new data.

## Known transfer risk to our own QVAR hardware (not yet resolved)

The R-peak amplitude channel's scale comes from PhysioNet's calibrated single-lead ECG (physical units via the recording's header gain, cleaned with `neurokit2`). Our SensorTile's QVAR channel is an **electrostatic sensor**, not a calibrated ECG amplifier — its raw amplitude units, morphology, and noise floor will very likely differ substantially from PhysioNet's, even after the same cleaning pipeline. The RR-interval channel (time-based, seconds) should transfer more safely since it's sample-rate-independent by construction. Before trusting this model's output on our own recordings, the amplitude channel's scale should be checked (e.g. does normalizing with *this* `norm_stats.json` put QVAR R-peak amplitudes in a sane range, or do they need their own rescaling) — this has not been tested, since we have no QVAR data yet.

## Reproducing

`python src/ecg/train_deployable_model.py` — rebuilds this exact artifact (same 30 epochs, seed 0, same 16,949-minute set as every other Phase 2/3 result). Reuses the cached 2-channel sequences at `results/phase3_cnn_v2/sequences.npz` if present.
