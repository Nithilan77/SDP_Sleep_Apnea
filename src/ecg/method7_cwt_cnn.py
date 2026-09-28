"""
Feature study -- method 7: CWT scalogram of the RR series, 2D-CNN.

Image sub-comparison (per the study's split: 1D-feature methods share
RRCNN, image methods share ScalogramCNN, src/ecg/cnn2d_model.py). Same
35-fold LOSO discipline, same eval metrics, same minute set as every other
method.

Normalization: single global mean/std over the whole scalogram (all
scales/timepoints/minutes in the training fold) -- the scalogram is a
homogeneous 2D image (one physical quantity, |CWT|, varying over scale and
time), not a set of heterogeneous scalar columns, so this matches how
Phase 3 normalized its homogeneous RR/amplitude channels rather than the
per-column standardization used for the scalar HRV banks (methods 3-5).

Usage:
    python src/ecg/method7_cwt_cnn.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ecg.cnn2d_model import predict_cnn2d, train_cnn2d  # noqa: E402
from src.ecg.cwt_scalogram import build_scalogram_dataset  # noqa: E402
from src.ecg.features import FEATURE_COLUMNS  # noqa: E402
from src.eval.metrics import binary_metrics, format_metrics  # noqa: E402
from src.eval.splits import leave_one_subject_out, list_labelled_records, record_group  # noqa: E402

RESULTS_DIR = Path("results/feature_study/method7_cwt_scalogram")
SCALOGRAM_CACHE = RESULTS_DIR / "scalograms.npz"
PHASE2_FEATURES = Path("results/phase2_cvhr_baseline/features.csv")

EPOCHS = 30
SEED = 0
RECORDING_APNEA_FRACTION_THRESHOLD = 0.05


def load_or_build_scalograms() -> dict:
    if SCALOGRAM_CACHE.exists():
        print(f"Loading cached CWT scalograms from {SCALOGRAM_CACHE}")
        d = np.load(SCALOGRAM_CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}

    print("Building per-minute CWT scalograms for all 35 labelled records...")
    records = list_labelled_records()
    ds = build_scalogram_dataset(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(SCALOGRAM_CACHE, **ds)
    print(f"  cached to {SCALOGRAM_CACHE}")
    return ds


def restrict_to_phase2_minutes(ds: dict) -> np.ndarray:
    p2 = pd.read_csv(PHASE2_FEATURES).dropna(subset=FEATURE_COLUMNS)
    keys = set(zip(p2["record"], p2["minute"].astype(int)))
    mask = np.array([(r, int(m)) in keys for r, m in zip(ds["record"], ds["minute"])])
    assert mask.sum() == len(p2), "minute sets do not match across phases!"
    return mask


def run_loso(ds: dict, mask: np.ndarray) -> pd.DataFrame:
    records = sorted(np.unique(ds["record"][mask]))
    scalograms = ds["scalogram"][mask]
    labels = (ds["label"][mask] == "A").astype(np.float32)
    rec_col = ds["record"][mask]
    min_col = ds["minute"][mask]

    all_preds = []
    t_start_all = time.time()
    for fold_i, (train_records, test_record) in enumerate(leave_one_subject_out(records), 1):
        t0 = time.time()
        train_idx = np.isin(rec_col, train_records)
        test_idx = rec_col == test_record

        X_train, y_train = scalograms[train_idx], labels[train_idx]
        X_test, y_test = scalograms[test_idx], labels[test_idx]

        mu = X_train.mean()
        sd = X_train.std()
        X_train_n = (X_train - mu) / sd
        X_test_n = (X_test - mu) / sd

        model = train_cnn2d(X_train_n, y_train, epochs=EPOCHS, seed=SEED)
        y_prob = predict_cnn2d(model, X_test_n)
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
    ds = load_or_build_scalograms()
    mask = restrict_to_phase2_minutes(ds)

    print(f"\nRunning method 7 (CWT scalogram, 2D-CNN), {EPOCHS} epochs/fold, "
          f"{len(np.unique(ds['record'][mask]))} LOSO folds")
    preds = run_loso(ds, mask)
    preds.to_csv(RESULTS_DIR / "per_minute_predictions.csv", index=False)

    m = binary_metrics(preds["y_true"], preds["y_pred"])
    print("\n=== Method 7 result (subject-independent, LOSO, all 35 records) ===")
    print(format_metrics(m))
    if m["accuracy"] > 0.95:
        print("  ** SUSPICIOUSLY HIGH accuracy. Check for subject leakage. **")

    rec_eval = per_recording_evaluation(preds)
    rec_eval.to_csv(RESULTS_DIR / "per_recording_predictions.csv", index=False)
    print(f"record-level accuracy: {int(rec_eval['correct'].sum())}/{len(rec_eval)} "
          f"= {rec_eval['correct'].mean():.3f}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
