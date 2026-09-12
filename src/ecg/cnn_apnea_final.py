"""
Phase 3 -- FINAL ECG-branch model (adopted after the ablation).

Config: 2-channel input (RR-interval + R-peak amplitude, see
rr_amp_sequence.py), DEFAULT class-balanced loss weighting (pos_weight =
n_neg/n_pos, no extra sensitivity-boost factor). Same LOSO split, same exact
16,949-minute set as Phase 2 and every earlier Phase 3 variant.

Why this config and not v1 or v2: the ablation (see results/phase3_cnn/README.md)
showed the R-amplitude channel drives the entire sensitivity gain by itself,
and the extra 1.5x loss-weighting tried in v2 only costs specificity on top
of that -- it does not help. This script reproduces that result, called
"Variant A" during the ablation:

    accuracy=0.812  sensitivity=0.756  specificity=0.846

This is the closed, adopted Phase 3 result. Per the project plan, Phase 3 is
now done -- this script exists for reproducibility (e.g. after a code
change), not for further tuning.

Usage:
    python src/ecg/cnn_apnea_final.py
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

RESULTS_DIR = Path("results/phase3_cnn")               # the canonical Phase 3 results home
SEQ_CACHE = Path("results/phase3_cnn_v2/sequences.npz")  # reuse the already-built 2ch cache
PHASE2_FEATURES = Path("results/phase2_cvhr_baseline/features.csv")

EPOCHS = 30
SEED = 0
POS_WEIGHT_FACTOR = 1.0  # adopted: the ablation showed >1.0 only costs specificity here
RECORDING_APNEA_FRACTION_THRESHOLD = 0.05


def load_or_build_sequences() -> dict:
    if SEQ_CACHE.exists():
        print(f"Loading cached 2-channel RR+amplitude sequences from {SEQ_CACHE}")
        d = np.load(SEQ_CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}

    print("Building per-minute (RR, R-amplitude) sequences for all 35 labelled records...")
    records = list_labelled_records()
    ds = build_sequence_dataset_2ch(records, seq_len=SEQ_LEN)
    SEQ_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SEQ_CACHE, **ds)
    print(f"  cached to {SEQ_CACHE}")
    return ds


def restrict_to_phase2_minutes(ds: dict) -> np.ndarray:
    p2 = pd.read_csv(PHASE2_FEATURES).dropna(subset=FEATURE_COLUMNS)
    keys = set(zip(p2["record"], p2["minute"].astype(int)))
    mask = np.array([(r, int(m)) in keys for r, m in zip(ds["record"], ds["minute"])])
    assert mask.sum() == len(p2), "minute sets do not match across phases!"
    return mask


def run_loso(ds: dict, mask: np.ndarray) -> pd.DataFrame:
    records = sorted(np.unique(ds["record"][mask]))
    sequences = ds["sequence"][mask]
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

        mu = X_train.mean(axis=(0, 2), keepdims=True)
        sd = X_train.std(axis=(0, 2), keepdims=True)
        X_train_n = (X_train - mu) / sd
        X_test_n = (X_test - mu) / sd

        model = train_cnn(X_train_n, y_train, seq_len=SEQ_LEN, epochs=EPOCHS, seed=SEED,
                           pos_weight_factor=POS_WEIGHT_FACTOR)
        y_prob = predict_cnn(model, X_test_n)
        y_pred = (y_prob > 0.5).astype(int)

        all_preds.append(pd.DataFrame({
            "record": rec_col[test_idx], "minute": min_col[test_idx],
            "y_true": y_test.astype(int), "y_pred": y_pred, "y_prob": y_prob,
        }))
        print(f"  fold {fold_i:2d}/{len(records)}: held out {test_record} "
              f"({test_idx.sum()} minutes, {time.time()-t0:.1f}s)")

    print(f"  total LOSO time: {(time.time()-t_start_all)/60:.1f} min")
    return pd.concat(all_preds, ignore_index=True)


def per_recording_evaluation(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record, g in preds.groupby("record"):
        true_group = record_group(record)
        pred_frac = g["y_pred"].mean()
        pred_group = "apnea" if pred_frac > RECORDING_APNEA_FRACTION_THRESHOLD else "control"
        rows.append({
            "record": record, "n_minutes": len(g),
            "true_apnea_fraction": g["y_true"].mean(), "pred_apnea_fraction": pred_frac,
            "true_group": true_group, "pred_group": pred_group,
            "correct": true_group == pred_group,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ds = load_or_build_sequences()
    mask = restrict_to_phase2_minutes(ds)

    print(f"\nRunning FINAL Phase 3 model: 2ch (RR+amplitude), "
          f"pos_weight_factor={POS_WEIGHT_FACTOR}, {EPOCHS} epochs/fold, "
          f"{len(np.unique(ds['record'][mask]))} LOSO folds")
    preds = run_loso(ds, mask)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(RESULTS_DIR / "final_per_minute_predictions.csv", index=False)

    m = binary_metrics(preds["y_true"], preds["y_pred"])
    print("\n=== FINAL Phase 3 result (subject-independent, LOSO, all 35 records) ===")
    print(format_metrics(m))
    if m["accuracy"] > 0.95:
        print("  ** SUSPICIOUSLY HIGH accuracy. Check for subject leakage. **")

    rec_eval = per_recording_evaluation(preds)
    rec_eval.to_csv(RESULTS_DIR / "final_per_recording_predictions.csv", index=False)
    print(f"record-level accuracy: {int(rec_eval['correct'].sum())}/{len(rec_eval)} "
          f"= {rec_eval['correct'].mean():.3f}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
