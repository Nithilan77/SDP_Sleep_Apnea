"""
Feature study -- method 8: spectrogram of the raw per-minute ECG segment,
2D-CNN. Same image-methods sub-comparison as method 7: shares ScalogramCNN
(src/ecg/cnn2d_model.py), same 35-fold LOSO discipline, same eval metrics,
same minute set, same single-global-mean/std normalization rationale (one
homogeneous image quantity, log-magnitude STFT power, not heterogeneous
scalar columns).

Usage:
    python src/ecg/method8_spectrogram_cnn.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ecg.cnn2d_model import predict_cnn2d, train_cnn2d  # noqa: E402
from src.ecg.ecg_spectrogram import build_spectrogram_dataset  # noqa: E402
from src.ecg.features import FEATURE_COLUMNS  # noqa: E402
from src.eval.metrics import binary_metrics, format_metrics  # noqa: E402
from src.eval.splits import leave_one_subject_out, list_labelled_records, record_group  # noqa: E402

RESULTS_DIR = Path("results/feature_study/method8_ecg_spectrogram")
SPEC_CACHE = RESULTS_DIR / "spectrograms.npz"
PHASE2_FEATURES = Path("results/phase2_cvhr_baseline/features.csv")

EPOCHS = 30
SEED = 0
RECORDING_APNEA_FRACTION_THRESHOLD = 0.05


def load_or_build_spectrograms() -> dict:
    if SPEC_CACHE.exists():
        print(f"Loading cached ECG spectrograms from {SPEC_CACHE}")
        d = np.load(SPEC_CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}

    print("Building per-minute ECG spectrograms for all 35 labelled records...")
    records = list_labelled_records()
    ds = build_spectrogram_dataset(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(SPEC_CACHE, **ds)
    print(f"  cached to {SPEC_CACHE}")
    return ds


def restrict_to_phase2_minutes(ds: dict) -> np.ndarray:
    p2 = pd.read_csv(PHASE2_FEATURES).dropna(subset=FEATURE_COLUMNS)
    keys = set(zip(p2["record"], p2["minute"].astype(int)))
    mask = np.array([(r, int(m)) in keys for r, m in zip(ds["record"], ds["minute"])])
    assert mask.sum() == len(p2), "minute sets do not match across phases!"
    return mask


def run_loso(ds: dict, mask: np.ndarray) -> pd.DataFrame:
    records = sorted(np.unique(ds["record"][mask]))
    spectrograms = ds["spectrogram"][mask]
    labels = (ds["label"][mask] == "A").astype(np.float32)
    rec_col = ds["record"][mask]
    min_col = ds["minute"][mask]

    all_preds = []
    t_start_all = time.time()
    for fold_i, (train_records, test_record) in enumerate(leave_one_subject_out(records), 1):
        t0 = time.time()
        train_idx = np.isin(rec_col, train_records)
        test_idx = rec_col == test_record

        X_train, y_train = spectrograms[train_idx], labels[train_idx]
        X_test, y_test = spectrograms[test_idx], labels[test_idx]

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
    ds = load_or_build_spectrograms()
    mask = restrict_to_phase2_minutes(ds)

    print(f"\nRunning method 8 (ECG spectrogram, 2D-CNN), {EPOCHS} epochs/fold, "
          f"{len(np.unique(ds['record'][mask]))} LOSO folds")
    preds = run_loso(ds, mask)
    preds.to_csv(RESULTS_DIR / "per_minute_predictions.csv", index=False)

    m = binary_metrics(preds["y_true"], preds["y_pred"])
    print("\n=== Method 8 result (subject-independent, LOSO, all 35 records) ===")
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
