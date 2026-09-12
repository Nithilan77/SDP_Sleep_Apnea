"""
Phase 2 -- classical CVHR baseline for per-minute apnea detection.

Logistic regression on hand-crafted CVHR/HRV features (mean HR, HR swing,
SDNN, RMSSD, pNN50, RR mean, minute-to-minute HR delta), evaluated with
leave-one-subject-out (LOSO) cross-validation across all 35 labelled
Apnea-ECG records (a01-a20, b01-b05, c01-c10). No subject's minutes are ever
in both train and test within a fold -- this is the "floor" classical
detector Phase 3's 1D-CNN must beat.

Reports per-minute AND per-recording metrics separately (different tasks,
per CLAUDE.md rule 2).

Usage:
    python src/ecg/cvhr_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ecg.features import FEATURE_COLUMNS, build_dataset  # noqa: E402
from src.eval.metrics import binary_metrics, format_metrics  # noqa: E402
from src.eval.splits import (  # noqa: E402
    leave_one_subject_out,
    list_labelled_records,
    record_group,
)

RESULTS_DIR = Path("results/phase2_cvhr_baseline")
FEATURES_CACHE = RESULTS_DIR / "features.csv"

# A record is flagged "apnea" at the recording level if more than this
# fraction of its minutes are predicted apnea. Control (group c) records
# should sit near 0; apnea/borderline records (group a/b) are, by the
# PhysioNet grouping definition, well above 5%. This threshold is a simple,
# documented heuristic -- not tuned on the test folds.
RECORDING_APNEA_FRACTION_THRESHOLD = 0.05


def load_or_build_features() -> pd.DataFrame:
    if FEATURES_CACHE.exists():
        print(f"Loading cached features from {FEATURES_CACHE}")
        return pd.read_csv(FEATURES_CACHE)

    print("Building per-minute features for all 35 labelled records...")
    records = list_labelled_records()
    print(f"  {len(records)} labelled records found")
    df = build_dataset(records)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_CACHE, index=False)
    print(f"  cached to {FEATURES_CACHE}")
    return df


def run_loso(df: pd.DataFrame) -> pd.DataFrame:
    """Run leave-one-subject-out CV, returning a table of per-minute predictions."""
    records = sorted(df["record"].unique())
    all_preds = []

    for fold_i, (train_records, test_record) in enumerate(leave_one_subject_out(records), 1):
        train_df = df[df["record"].isin(train_records)]
        test_df = df[df["record"] == test_record]

        X_train = train_df[FEATURE_COLUMNS].values
        y_train = (train_df["label"] == "A").astype(int).values
        X_test = test_df[FEATURE_COLUMNS].values
        y_test = (test_df["label"] == "A").astype(int).values

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        fold_result = test_df[["record", "minute"]].copy()
        fold_result["y_true"] = y_test
        fold_result["y_pred"] = y_pred
        fold_result["y_prob"] = y_prob
        all_preds.append(fold_result)

        print(f"  fold {fold_i:2d}/{len(records)}: held out {test_record} "
              f"({len(test_df)} minutes)")

    return pd.concat(all_preds, ignore_index=True)


def per_recording_evaluation(preds: pd.DataFrame) -> pd.DataFrame:
    """Roll per-minute predictions up to a per-recording apnea/control call."""
    rows = []
    for record, g in preds.groupby("record"):
        true_frac = g["y_true"].mean()
        pred_frac = g["y_pred"].mean()
        true_group = record_group(record)
        pred_group = "apnea" if pred_frac > RECORDING_APNEA_FRACTION_THRESHOLD else "control"
        rows.append({
            "record": record,
            "n_minutes": len(g),
            "true_apnea_fraction": true_frac,
            "pred_apnea_fraction": pred_frac,
            "true_group": true_group,
            "pred_group": pred_group,
            "correct": true_group == pred_group,
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = load_or_build_features()

    n_before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS)
    n_dropped = n_before - len(df)
    print(f"\n{len(df)} minutes usable across {df['record'].nunique()} records "
          f"({n_dropped} dropped for missing features, e.g. too few beats)")

    print(f"\nOverall label balance: "
          f"{int((df['label'] == 'A').sum())} apnea / "
          f"{int((df['label'] == 'N').sum())} normal minutes "
          f"({100 * (df['label'] == 'A').mean():.1f}% apnea)")

    print("\n=== Running leave-one-subject-out cross-validation ===")
    preds = run_loso(df)
    preds.to_csv(RESULTS_DIR / "per_minute_predictions.csv", index=False)

    print("\n=== Per-minute results (subject-independent, LOSO, all 35 records) ===")
    m = binary_metrics(preds["y_true"], preds["y_pred"])
    print(format_metrics(m))
    if m["accuracy"] > 0.95:
        print("  ** SUSPICIOUSLY HIGH accuracy for a classical per-minute baseline. "
              "Check for subject leakage before trusting this number. **")

    print("\n=== Per-recording results (apnea group a/b vs control group c) ===")
    rec_eval = per_recording_evaluation(preds)
    rec_eval.to_csv(RESULTS_DIR / "per_recording_predictions.csv", index=False)
    rec_acc = rec_eval["correct"].mean()
    n_correct = int(rec_eval["correct"].sum())
    print(f"record-level accuracy: {n_correct}/{len(rec_eval)} = {rec_acc:.3f}")
    misclassified = rec_eval[~rec_eval["correct"]]
    if len(misclassified):
        print("misclassified records:")
        for _, r in misclassified.iterrows():
            print(f"  {r['record']}: true={r['true_group']}, pred={r['pred_group']} "
                  f"(pred apnea-minute fraction={r['pred_apnea_fraction']:.3f})")
    else:
        print("  all records classified correctly at the recording level")

    print(f"\nResults saved to {RESULTS_DIR}/")
    print("\nLiterature floor for a from-scratch CVHR baseline: ~72-80% per-minute "
          "(untuned). Compare the per-minute accuracy above against that, not against "
          "the per-recording number -- they are different tasks.")


if __name__ == "__main__":
    main()
