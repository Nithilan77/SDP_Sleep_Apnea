"""
Phase 3 iteration (v2) -- targets the sensitivity weakness found in the v1
CNN (sens 0.653, barely above the classical baseline's 0.637) with two
specific changes, per the brief:

  1. Class-weighted loss with an explicit sensitivity boost: pos_weight is
     the natural class-balance ratio (n_neg/n_pos) times POS_WEIGHT_FACTOR
     (>1), deliberately biasing the loss toward catching apnea minutes at
     some cost to specificity. Fixed before looking at any results -- not
     tuned against test-fold performance.
  2. A second input channel: R-peak amplitude (ECG-derived respiration
     proxy) alongside the RR-interval sequence, per the literature's
     higher-accuracy single-lead models.

Same LOSO split, same exact (record, minute) set as Phase 2 and Phase 3 v1
-- all three numbers are directly comparable.

Usage:
    python src/ecg/cnn_apnea_v2.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ecg.cnn_model import train_cnn, predict_cnn  # noqa: E402
from src.ecg.features import FEATURE_COLUMNS  # noqa: E402
from src.ecg.rr_amp_sequence import SEQ_LEN, build_sequence_dataset_2ch  # noqa: E402
from src.eval.metrics import binary_metrics, format_metrics  # noqa: E402
from src.eval.splits import leave_one_subject_out, list_labelled_records, record_group  # noqa: E402

RESULTS_DIR = Path("results/phase3_cnn_v2")
SEQ_CACHE = RESULTS_DIR / "sequences.npz"
PHASE2_FEATURES = Path("results/phase2_cvhr_baseline/features.csv")
PHASE2_PER_MINUTE = Path("results/phase2_cvhr_baseline/per_minute_predictions.csv")
V1_PER_MINUTE = Path("results/phase3_cnn/per_minute_predictions.csv")

EPOCHS = 30
SEED = 0
POS_WEIGHT_FACTOR = 1.5  # fixed, documented sensitivity-boost multiplier -- not tuned on test folds
RECORDING_APNEA_FRACTION_THRESHOLD = 0.05


def load_or_build_sequences() -> dict:
    if SEQ_CACHE.exists():
        print(f"Loading cached 2-channel RR+amplitude sequences from {SEQ_CACHE}")
        d = np.load(SEQ_CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}

    print("Building per-minute (RR, R-amplitude) sequences for all 35 labelled records...")
    records = list_labelled_records()
    print(f"  {len(records)} labelled records found")
    ds = build_sequence_dataset_2ch(records, seq_len=SEQ_LEN)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(SEQ_CACHE, **ds)
    print(f"  cached to {SEQ_CACHE}")
    return ds


def restrict_to_phase2_minutes(ds: dict) -> np.ndarray:
    if not PHASE2_FEATURES.exists():
        raise FileNotFoundError(
            f"{PHASE2_FEATURES} not found -- run src/ecg/cvhr_baseline.py first."
        )
    p2 = pd.read_csv(PHASE2_FEATURES).dropna(subset=FEATURE_COLUMNS)
    keys = set(zip(p2["record"], p2["minute"].astype(int)))
    mask = np.array([(r, int(m)) in keys for r, m in zip(ds["record"], ds["minute"])])
    print(f"Restricting to Phase 2's evaluated minutes: {mask.sum()} of {len(mask)} "
          f"(Phase 2 had {len(p2)})")
    assert mask.sum() == len(p2), "minute sets do not match across phases!"
    return mask


def run_loso(ds: dict, mask: np.ndarray) -> pd.DataFrame:
    records = sorted(np.unique(ds["record"][mask]))
    sequences = ds["sequence"][mask]  # (n, 2, seq_len)
    labels = (ds["label"][mask] == "A").astype(np.float32)
    rec_col = ds["record"][mask]
    min_col = ds["minute"][mask]

    all_preds = []
    t_start_all = time.time()
    for fold_i, (train_records, test_record) in enumerate(leave_one_subject_out(records), 1):
        t0 = time.time()
        train_idx = np.isin(rec_col, train_records)
        test_idx = rec_col == test_record

        X_train, y_train = sequences[train_idx], labels[train_idx]
        X_test, y_test = sequences[test_idx], labels[test_idx]

        # Per-channel normalization from TRAIN-FOLD stats only (RR in seconds
        # and R-amplitude are on very different scales).
        mu = X_train.mean(axis=(0, 2), keepdims=True)
        sd = X_train.std(axis=(0, 2), keepdims=True)
        X_train_n = (X_train - mu) / sd
        X_test_n = (X_test - mu) / sd

        model = train_cnn(X_train_n, y_train, seq_len=SEQ_LEN, epochs=EPOCHS, seed=SEED,
                           pos_weight_factor=POS_WEIGHT_FACTOR)
        y_prob = predict_cnn(model, X_test_n)
        y_pred = (y_prob > 0.5).astype(int)

        fold_result = pd.DataFrame({
            "record": rec_col[test_idx],
            "minute": min_col[test_idx],
            "y_true": y_test.astype(int),
            "y_pred": y_pred,
            "y_prob": y_prob,
        })
        all_preds.append(fold_result)
        print(f"  fold {fold_i:2d}/{len(records)}: held out {test_record} "
              f"({test_idx.sum()} minutes, {time.time()-t0:.1f}s)")

    print(f"  total LOSO time: {(time.time()-t_start_all)/60:.1f} min")
    return pd.concat(all_preds, ignore_index=True)


def per_recording_evaluation(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record, g in preds.groupby("record"):
        true_frac = g["y_true"].mean()
        pred_frac = g["y_pred"].mean()
        true_group = record_group(record)
        pred_group = "apnea" if pred_frac > RECORDING_APNEA_FRACTION_THRESHOLD else "control"
        rows.append({
            "record": record, "n_minutes": len(g),
            "true_apnea_fraction": true_frac, "pred_apnea_fraction": pred_frac,
            "true_group": true_group, "pred_group": pred_group,
            "correct": true_group == pred_group,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ds = load_or_build_sequences()
    mask = restrict_to_phase2_minutes(ds)

    print(f"\nRunning LOSO CNN-v2 training: {EPOCHS} epochs/fold, "
          f"pos_weight_factor={POS_WEIGHT_FACTOR}, 2 input channels (RR, R-amplitude), "
          f"{len(np.unique(ds['record'][mask]))} folds")
    print("=== Running leave-one-subject-out cross-validation ===")
    preds = run_loso(ds, mask)
    preds.to_csv(RESULTS_DIR / "per_minute_predictions.csv", index=False)

    print("\n=== Per-minute results (subject-independent, LOSO, all 35 records) ===")
    v2_m = binary_metrics(preds["y_true"], preds["y_pred"])
    print("CNN v2 (2ch + weighted loss): " + format_metrics(v2_m))
    if v2_m["accuracy"] > 0.95:
        print("  ** SUSPICIOUSLY HIGH accuracy. Check for subject leakage before "
              "trusting this number. **")

    if PHASE2_PER_MINUTE.exists():
        base = pd.read_csv(PHASE2_PER_MINUTE)
        base_m = binary_metrics(base["y_true"], base["y_pred"])
        print("Phase 2 classical baseline:    " + format_metrics(base_m))
    else:
        base_m = None

    if V1_PER_MINUTE.exists():
        v1 = pd.read_csv(V1_PER_MINUTE)
        v1_m = binary_metrics(v1["y_true"], v1["y_pred"])
        print("Phase 3 v1 CNN (RRI only):      " + format_metrics(v1_m))
    else:
        v1_m = None

    def delta_block(label, ref_m):
        if ref_m is None:
            return
        print(f"\nDelta vs {label}:")
        print(f"  accuracy:    {v2_m['accuracy'] - ref_m['accuracy']:+.3f}  "
              f"({ref_m['accuracy']:.3f} -> {v2_m['accuracy']:.3f})")
        print(f"  sensitivity: {v2_m['sensitivity'] - ref_m['sensitivity']:+.3f}  "
              f"({ref_m['sensitivity']:.3f} -> {v2_m['sensitivity']:.3f})")
        print(f"  specificity: {v2_m['specificity'] - ref_m['specificity']:+.3f}  "
              f"({ref_m['specificity']:.3f} -> {v2_m['specificity']:.3f})")

    delta_block("Phase 2 classical baseline (71.1% acc)", base_m)
    delta_block("Phase 3 v1 CNN (77.5% acc)", v1_m)

    if v1_m is not None:
        if v2_m["sensitivity"] > v1_m["sensitivity"]:
            print(f"\nSensitivity IMPROVED as targeted: {v1_m['sensitivity']:.3f} -> "
                  f"{v2_m['sensitivity']:.3f}.")
        else:
            print(f"\nSensitivity did NOT improve: {v1_m['sensitivity']:.3f} -> "
                  f"{v2_m['sensitivity']:.3f}. Reporting this as-is, not tuning further "
                  "to hide it.")

    print("\n=== Per-recording results (apnea group a/b vs control group c) ===")
    rec_eval = per_recording_evaluation(preds)
    rec_eval.to_csv(RESULTS_DIR / "per_recording_predictions.csv", index=False)
    print(f"record-level accuracy: {int(rec_eval['correct'].sum())}/{len(rec_eval)} "
          f"= {rec_eval['correct'].mean():.3f}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
