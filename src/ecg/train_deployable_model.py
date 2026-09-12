"""
Freeze a DEPLOYABLE ECG-branch model: Phase 3's adopted final config (2-channel
RR-interval + R-peak amplitude, default class-balanced weighting), trained on
ALL 35 labelled PhysioNet records with NO held-out test set.

This is NOT an evaluation run. Phase 3's LOSO cross-validation
(accuracy=0.812, sensitivity=0.756, specificity=0.846 -- see
results/phase3_cnn/README.md) is the honest estimate of how this
architecture+config generalizes to an unseen subject; that estimate stays
valid for THIS model too, since it's the same recipe trained on more data of
the same kind. But this script's own training-set fit number (printed below)
is not a generalization estimate -- it's just a sanity check that training
converged, and is expected to look better than LOSO because the model has
seen every minute it's being checked against.

Purpose: this is the actual weight file to run on our own QVAR recordings
once the SensorTile hardware is back. Per the project's core constraint, it
is trained ONLY on public PhysioNet data -- never on our own recordings.

Usage:
    python src/ecg/train_deployable_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ecg.cnn_model import predict_cnn, train_cnn  # noqa: E402
from src.ecg.features import FEATURE_COLUMNS  # noqa: E402
from src.ecg.rr_amp_sequence import N_CHANNELS, SEQ_LEN, build_sequence_dataset_2ch  # noqa: E402
from src.eval.splits import list_labelled_records  # noqa: E402

OUT_DIR = Path("results/phase3_cnn/deployable")
SEQ_CACHE = Path("results/phase3_cnn_v2/sequences.npz")  # reuse the already-built 2ch cache
PHASE2_FEATURES = Path("results/phase2_cvhr_baseline/features.csv")

EPOCHS = 30
SEED = 0
POS_WEIGHT_FACTOR = 1.0  # adopted Phase 3 final config (see results/phase3_cnn/README.md)


def load_or_build_sequences() -> dict:
    if SEQ_CACHE.exists():
        print(f"Loading cached 2-channel sequences from {SEQ_CACHE}")
        d = np.load(SEQ_CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}
    print("Building sequences for all 35 records...")
    records = list_labelled_records()
    ds = build_sequence_dataset_2ch(records, seq_len=SEQ_LEN)
    SEQ_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SEQ_CACHE, **ds)
    return ds


def restrict_to_phase2_minutes(ds: dict) -> np.ndarray:
    p2 = pd.read_csv(PHASE2_FEATURES).dropna(subset=FEATURE_COLUMNS)
    keys = set(zip(p2["record"], p2["minute"].astype(int)))
    mask = np.array([(r, int(m)) in keys for r, m in zip(ds["record"], ds["minute"])])
    assert mask.sum() == len(p2), "minute set does not match Phase 2/3's evaluated minutes!"
    return mask


def main() -> None:
    ds = load_or_build_sequences()
    mask = restrict_to_phase2_minutes(ds)

    X = ds["sequence"][mask]  # (n, 2, seq_len)
    y = (ds["label"][mask] == "A").astype(np.float32)
    n_records = len(np.unique(ds["record"][mask]))
    n = len(y)

    print(f"\nTraining on ALL {n} minutes across {n_records} records "
          f"(no held-out test -- deployment build, not an evaluation).")
    print(f"Label balance: {int(y.sum())} apnea / {int((1 - y).sum())} normal "
          f"({100 * y.mean():.1f}% apnea)")

    mu = X.mean(axis=(0, 2), keepdims=True)
    sd = X.std(axis=(0, 2), keepdims=True)
    X_n = (X - mu) / sd

    model = train_cnn(X_n, y, seq_len=SEQ_LEN, epochs=EPOCHS, seed=SEED,
                       pos_weight_factor=POS_WEIGHT_FACTOR)

    probs = predict_cnn(model, X_n)
    pred = (probs > 0.5).astype(int)
    train_fit_acc = float((pred == y).mean())
    print(f"\nTraining-set fit (NOT a generalization estimate -- see docstring): "
          f"acc={train_fit_acc:.3f}, pred apnea frac={pred.mean():.3f} vs true {y.mean():.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_DIR / "model_state_dict.pt")

    norm_stats = {
        "channel_names": ["rr_interval_seconds", "r_peak_amplitude"],
        "mean": mu.ravel().tolist(),
        "std": sd.ravel().tolist(),
    }
    with open(OUT_DIR / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    config = {
        "architecture": "RRCNN",
        "seq_len": SEQ_LEN,
        "n_channels": N_CHANNELS,
        "pos_weight_factor": POS_WEIGHT_FACTOR,
        "epochs": EPOCHS,
        "seed": SEED,
        "trained_on": "all 35 labelled PhysioNet Apnea-ECG records (a01-a20, b01-b05, c01-c10)",
        "n_minutes_trained_on": int(n),
        "training_set_fit_accuracy": train_fit_acc,
        "expected_generalization_performance": {
            "source": "Phase 3 LOSO evaluation (results/phase3_cnn/README.md), "
                       "NOT measured on this exact model",
            "accuracy": 0.812, "sensitivity": 0.756, "specificity": 0.846,
        },
    }
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved to {OUT_DIR}/: model_state_dict.pt, norm_stats.json, config.json")


if __name__ == "__main__":
    main()
