"""
Feature study -- method 6: RR-interval + QRS-area, 2-channel 1D-CNN.

Directly analogous to the adopted Phase 3 final model (method 2: RR +
R-peak amplitude) -- same RRCNN architecture, same train_cnn/predict_cnn
(unmodified), same default class-balanced loss weighting, same per-channel
mu/sd normalization (fit per training fold), same 35-fold LOSO discipline,
same minute set (joined against Phase 2's cached feature table). The only
change from method 2 is swapping the second channel (R-peak amplitude ->
QRS-area, src/ecg/qrs_area.py) -- an isolated, controlled comparison of two
EDR proxies under an identical model.

Usage:
    python src/ecg/method6_qrsarea_cnn.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ecg.cnn_model import predict_cnn, train_cnn  # noqa: E402
from src.ecg.features import FEATURE_COLUMNS  # noqa: E402
from src.ecg.rr_qrsarea_sequence import SEQ_LEN, build_sequence_dataset_qrsarea  # noqa: E402
from src.eval.metrics import binary_metrics, format_metrics  # noqa: E402
from src.eval.splits import leave_one_subject_out, list_labelled_records, record_group  # noqa: E402

RESULTS_DIR = Path("results/feature_study/method6_qrsarea_edr")
SEQ_CACHE = RESULTS_DIR / "sequences.npz"
PHASE2_FEATURES = Path("results/phase2_cvhr_baseline/features.csv")

EPOCHS = 30
SEED = 0
POS_WEIGHT_FACTOR = 1.0  # matches the adopted Phase 3 config (method 2)
RECORDING_APNEA_FRACTION_THRESHOLD = 0.05


def load_or_build_sequences() -> dict:
    if SEQ_CACHE.exists():
        print(f"Loading cached RR+QRS-area sequences from {SEQ_CACHE}")
        d = np.load(SEQ_CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}

    print("Building per-minute (RR, QRS-area) sequences for all 35 labelled records...")
    records = list_labelled_records()
    ds = build_sequence_dataset_qrsarea(records, seq_len=SEQ_LEN)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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

    print(f"\nRunning method 6 (RR + QRS-area, 2ch), pos_weight_factor={POS_WEIGHT_FACTOR}, "
          f"{EPOCHS} epochs/fold, {len(np.unique(ds['record'][mask]))} LOSO folds")
    preds = run_loso(ds, mask)
    preds.to_csv(RESULTS_DIR / "per_minute_predictions.csv", index=False)

    m = binary_metrics(preds["y_true"], preds["y_pred"])
    print("\n=== Method 6 result (subject-independent, LOSO, all 35 records) ===")
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
