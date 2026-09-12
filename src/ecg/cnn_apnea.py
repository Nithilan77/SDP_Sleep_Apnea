"""
Phase 3 -- 1D-CNN for per-minute apnea detection, on RR-interval sequences.

Same leave-one-subject-out (LOSO) split as Phase 2's classical CVHR baseline,
restricted to the *exact same* (record, minute) set Phase 2 evaluated on
(Phase 2's hr_delta_prev feature drops each record's first minute via a
pandas .diff(), which the raw RR-sequence extraction does not do by itself --
so we explicitly join on Phase 2's cached feature table to guarantee the two
numbers are directly comparable). A fresh CNN is trained from scratch per
fold; normalization stats (mean/std) are computed from that fold's training
data only, never the held-out record, to avoid any test-fold leakage.

Usage:
    python src/ecg/cnn_apnea.py
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
from src.ecg.rr_sequence import SEQ_LEN, build_sequence_dataset  # noqa: E402
from src.eval.metrics import binary_metrics, format_metrics  # noqa: E402
from src.eval.splits import leave_one_subject_out, list_labelled_records, record_group  # noqa: E402

RESULTS_DIR = Path("results/phase3_cnn")
SEQ_CACHE = RESULTS_DIR / "sequences.npz"
PHASE2_FEATURES = Path("results/phase2_cvhr_baseline/features.csv")
PHASE2_PER_MINUTE = Path("results/phase2_cvhr_baseline/per_minute_predictions.csv")

EPOCHS = 30
SEED = 0
RECORDING_APNEA_FRACTION_THRESHOLD = 0.05  # same heuristic as Phase 2, for comparability


def load_or_build_sequences() -> dict:
    if SEQ_CACHE.exists():
        print(f"Loading cached RR sequences from {SEQ_CACHE}")
        d = np.load(SEQ_CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}

    print("Building per-minute RR-interval sequences for all 35 labelled records...")
    records = list_labelled_records()
    print(f"  {len(records)} labelled records found")
    ds = build_sequence_dataset(records, seq_len=SEQ_LEN)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(SEQ_CACHE, **ds)
    print(f"  cached to {SEQ_CACHE}")
    return ds


def restrict_to_phase2_minutes(ds: dict) -> np.ndarray:
    """Boolean mask selecting the exact (record, minute) set Phase 2 evaluated."""
    if not PHASE2_FEATURES.exists():
        raise FileNotFoundError(
            f"{PHASE2_FEATURES} not found -- run src/ecg/cvhr_baseline.py first "
            "so Phase 3 evaluates on the exact same minutes as Phase 2."
        )
    p2 = pd.read_csv(PHASE2_FEATURES).dropna(subset=FEATURE_COLUMNS)
    keys = set(zip(p2["record"], p2["minute"].astype(int)))
    mask = np.array([(r, int(m)) in keys for r, m in zip(ds["record"], ds["minute"])])
    print(f"Restricting to Phase 2's evaluated minutes: {mask.sum()} of {len(mask)} "
          f"(Phase 2 had {len(p2)})")
    assert mask.sum() == len(p2), "Phase 2/Phase 3 minute sets do not match!"
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

        # Normalize using TRAIN-FOLD statistics only -- never the held-out record.
        mu, sd = X_train.mean(), X_train.std()
        X_train_n = (X_train - mu) / sd
        X_test_n = (X_test - mu) / sd

        model = train_cnn(X_train_n, y_train, seq_len=SEQ_LEN, epochs=EPOCHS, seed=SEED)
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

    print(f"\nRunning LOSO CNN training: {EPOCHS} epochs/fold, "
          f"{len(np.unique(ds['record'][mask]))} folds")
    print("=== Running leave-one-subject-out cross-validation ===")
    preds = run_loso(ds, mask)
    preds.to_csv(RESULTS_DIR / "per_minute_predictions.csv", index=False)

    print("\n=== Per-minute results (subject-independent, LOSO, all 35 records) ===")
    cnn_m = binary_metrics(preds["y_true"], preds["y_pred"])
    print("CNN:      " + format_metrics(cnn_m))
    if cnn_m["accuracy"] > 0.95:
        print("  ** SUSPICIOUSLY HIGH accuracy. Check for subject leakage before "
              "trusting this number. **")

    if PHASE2_PER_MINUTE.exists():
        base = pd.read_csv(PHASE2_PER_MINUTE)
        base_m = binary_metrics(base["y_true"], base["y_pred"])
        print("Baseline: " + format_metrics(base_m))
        print(f"\nDelta over Phase 2 classical CVHR baseline:")
        print(f"  accuracy:    {cnn_m['accuracy'] - base_m['accuracy']:+.3f}  "
              f"({base_m['accuracy']:.3f} -> {cnn_m['accuracy']:.3f})")
        print(f"  sensitivity: {cnn_m['sensitivity'] - base_m['sensitivity']:+.3f}  "
              f"({base_m['sensitivity']:.3f} -> {cnn_m['sensitivity']:.3f})")
        print(f"  specificity: {cnn_m['specificity'] - base_m['specificity']:+.3f}  "
              f"({base_m['specificity']:.3f} -> {cnn_m['specificity']:.3f})")
        if cnn_m["accuracy"] <= base_m["accuracy"]:
            print("  ** CNN did NOT beat the classical baseline on accuracy. "
                  "Per rule 3, this needs to be reported honestly, not hidden. **")
    else:
        print(f"\n(No {PHASE2_PER_MINUTE} found -- run Phase 2 first for a delta.)")

    print("\n=== Per-recording results (apnea group a/b vs control group c) ===")
    rec_eval = per_recording_evaluation(preds)
    rec_eval.to_csv(RESULTS_DIR / "per_recording_predictions.csv", index=False)
    rec_acc = rec_eval["correct"].mean()
    print(f"record-level accuracy: {int(rec_eval['correct'].sum())}/{len(rec_eval)} "
          f"= {rec_acc:.3f}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
