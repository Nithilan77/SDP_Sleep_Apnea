"""
Feature study -- per-minute apnea classification for a scalar HRV feature
bank (methods 3/4/5: time-domain, frequency-domain, nonlinear), compared
across THREE classifiers per bank: the Phase 3 1D-CNN (RRCNN), logistic
regression, and a small MLP. Same 35-fold LOSO discipline, same minute set
(joined against the Phase 2 cached feature table), same eval metrics for all
three.

Why three classifiers (not just the CNN): an HRV feature bank is a handful
of already-aggregated scalars, not a real within-minute time series -- the
first version of this study forced these scalars through RRCNN (treating the
feature vector as a 1-channel length-K "sequence") to keep a strictly
controlled single-model comparison, but that handicaps feature types that
suit a simpler classifier (logistic regression, or an unstructured MLP) more
naturally than a convolutional architecture built for ordered sequences.
This version keeps the CNN result already produced (see cnn/ subdir, code
below reused verbatim -- src/ecg/cnn_model.py's RRCNN/train_cnn/predict_cnn
are UNCHANGED) and adds two more classifiers per bank:

  - Logistic regression: StandardScaler + sklearn LogisticRegression
    (class_weight="balanced"), same recipe as Phase 2's classical baseline.
  - Small MLP (src/ecg/cnn_model.py: FeatureMLP/train_mlp/predict_mlp):
    2 hidden layers (16, 8 units), no convolution/ordering assumption,
    trained the same way (Adam, BCEWithLogitsLoss with class-balanced
    pos_weight, 30 epochs) as the CNN for a fair training budget.

The BEST of the three per bank (by per-minute accuracy) is what goes into
the feature study's master results table -- the table records which
classifier won for each method, so the comparison stays honest about what
produced the reported number.

Usage:
    python src/ecg/feature_bank_cnn.py time        # bank 3, all 3 classifiers
    python src/ecg/feature_bank_cnn.py freq        # bank 4
    python src/ecg/feature_bank_cnn.py nonlinear   # bank 5
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ecg.cnn_model import (  # noqa: E402
    predict_cnn,
    predict_mlp,
    train_cnn,
    train_mlp,
)
from src.ecg.features import FEATURE_COLUMNS as PHASE2_FEATURE_COLUMNS  # noqa: E402
from src.ecg.hrv_bank_features import (  # noqa: E402
    FREQ_DOMAIN_COLUMNS,
    NONLINEAR_COLUMNS,
    TIME_DOMAIN_COLUMNS,
    build_bank_dataset,
)
from src.eval.metrics import binary_metrics, format_metrics  # noqa: E402
from src.eval.splits import leave_one_subject_out, list_labelled_records, record_group  # noqa: E402

RESULTS_ROOT = Path("results/feature_study")
BANK_CACHE = RESULTS_ROOT / "hrv_bank_features.csv"
PHASE2_FEATURES = Path("results/phase2_cvhr_baseline/features.csv")
RECORDING_APNEA_FRACTION_THRESHOLD = 0.05

EPOCHS = 30
SEED = 0

BANKS = {
    "time": {
        "columns": TIME_DOMAIN_COLUMNS,
        "label": "Time-domain HRV bank (method 3)",
        "slug": "method3_time_domain",
    },
    "freq": {
        "columns": FREQ_DOMAIN_COLUMNS,
        "label": "Frequency-domain HRV bank (method 4)",
        "slug": "method4_freq_domain",
    },
    "nonlinear": {
        "columns": NONLINEAR_COLUMNS,
        "label": "Nonlinear/Poincare HRV bank (method 5)",
        "slug": "method5_nonlinear",
    },
}

CLASSIFIERS = ("cnn", "logreg", "mlp")


def load_or_build_bank_features() -> pd.DataFrame:
    if BANK_CACHE.exists():
        print(f"Loading cached HRV bank features from {BANK_CACHE}")
        return pd.read_csv(BANK_CACHE)

    print("Building per-minute HRV bank features (time+freq+nonlinear) for all 35 records...")
    records = list_labelled_records()
    df = build_bank_dataset(records)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(BANK_CACHE, index=False)
    print(f"  cached to {BANK_CACHE}")
    return df


def restrict_to_phase2_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the exact (record, minute) set Phase 2/3 used, for comparability."""
    p2 = pd.read_csv(PHASE2_FEATURES).dropna(subset=PHASE2_FEATURE_COLUMNS)
    keys = p2[["record", "minute"]].drop_duplicates()
    keys["minute"] = keys["minute"].astype(int)
    df = df.copy()
    df["minute"] = df["minute"].astype(int)
    out = df.merge(keys, on=["record", "minute"], how="inner")
    assert len(out) == len(p2), (
        f"minute set mismatch: got {len(out)}, expected {len(p2)} (Phase 2/3 parity)"
    )
    return out


def run_loso_cnn(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Feature vector as a 1-channel length-K sequence through RRCNN (unmodified)."""
    records = sorted(df["record"].unique())
    seq_len = len(feature_columns)

    all_preds = []
    t_start_all = time.time()
    for fold_i, (train_records, test_record) in enumerate(leave_one_subject_out(records), 1):
        t0 = time.time()
        train_df = df[df["record"].isin(train_records)]
        test_df = df[df["record"] == test_record]

        X_train = train_df[feature_columns].values.astype(np.float32)
        X_test = test_df[feature_columns].values.astype(np.float32)
        y_train = (train_df["label"] == "A").astype(np.float32).values
        y_test = (test_df["label"] == "A").astype(int).values

        mu = X_train.mean(axis=0, keepdims=True)
        sd = X_train.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        X_train_n = (X_train - mu) / sd
        X_test_n = (X_test - mu) / sd

        X_train_n = X_train_n[:, None, :]
        X_test_n = X_test_n[:, None, :]

        model = train_cnn(X_train_n, y_train, seq_len=seq_len, epochs=EPOCHS, seed=SEED)
        y_prob = predict_cnn(model, X_test_n)
        y_pred = (y_prob > 0.5).astype(int)

        all_preds.append(pd.DataFrame({
            "record": test_df["record"].values, "minute": test_df["minute"].values,
            "y_true": y_test, "y_pred": y_pred, "y_prob": y_prob,
        }))
        print(f"  [cnn] fold {fold_i:2d}/{len(records)}: held out {test_record} "
              f"({len(test_df)} minutes, {time.time()-t0:.1f}s)")

    print(f"  [cnn] total LOSO time: {(time.time()-t_start_all)/60:.1f} min")
    return pd.concat(all_preds, ignore_index=True)


def run_loso_logreg(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """StandardScaler + class-balanced logistic regression, same recipe as Phase 2."""
    records = sorted(df["record"].unique())

    all_preds = []
    t_start_all = time.time()
    for fold_i, (train_records, test_record) in enumerate(leave_one_subject_out(records), 1):
        train_df = df[df["record"].isin(train_records)]
        test_df = df[df["record"] == test_record]

        X_train = train_df[feature_columns].values
        X_test = test_df[feature_columns].values
        y_train = (train_df["label"] == "A").astype(int).values
        y_test = (test_df["label"] == "A").astype(int).values

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        all_preds.append(pd.DataFrame({
            "record": test_df["record"].values, "minute": test_df["minute"].values,
            "y_true": y_test, "y_pred": y_pred, "y_prob": y_prob,
        }))

    print(f"  [logreg] {len(records)} folds, {(time.time()-t_start_all):.1f}s total")
    return pd.concat(all_preds, ignore_index=True)


def run_loso_mlp(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Small unordered-feature MLP (FeatureMLP), same training budget as the CNN."""
    records = sorted(df["record"].unique())

    all_preds = []
    t_start_all = time.time()
    for fold_i, (train_records, test_record) in enumerate(leave_one_subject_out(records), 1):
        t0 = time.time()
        train_df = df[df["record"].isin(train_records)]
        test_df = df[df["record"] == test_record]

        X_train = train_df[feature_columns].values.astype(np.float32)
        X_test = test_df[feature_columns].values.astype(np.float32)
        y_train = (train_df["label"] == "A").astype(np.float32).values
        y_test = (test_df["label"] == "A").astype(int).values

        mu = X_train.mean(axis=0, keepdims=True)
        sd = X_train.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        X_train_n = (X_train - mu) / sd
        X_test_n = (X_test - mu) / sd

        model = train_mlp(X_train_n, y_train, epochs=EPOCHS, seed=SEED)
        y_prob = predict_mlp(model, X_test_n)
        y_pred = (y_prob > 0.5).astype(int)

        all_preds.append(pd.DataFrame({
            "record": test_df["record"].values, "minute": test_df["minute"].values,
            "y_true": y_test, "y_pred": y_pred, "y_prob": y_prob,
        }))
        print(f"  [mlp] fold {fold_i:2d}/{len(records)}: held out {test_record} "
              f"({len(test_df)} minutes, {time.time()-t0:.1f}s)")

    print(f"  [mlp] total LOSO time: {(time.time()-t_start_all)/60:.1f} min")
    return pd.concat(all_preds, ignore_index=True)


RUNNERS = {"cnn": run_loso_cnn, "logreg": run_loso_logreg, "mlp": run_loso_mlp}


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


def run_bank(bank_key: str, classifiers: tuple[str, ...] = CLASSIFIERS,
             reuse_cached_cnn: bool = True) -> dict:
    spec = BANKS[bank_key]
    columns = spec["columns"]
    slug = spec["slug"]
    bank_dir = RESULTS_ROOT / slug

    print(f"\n=== {spec['label']}: {columns} ===")
    df = load_or_build_bank_features()
    df = restrict_to_phase2_minutes(df)

    n_before = len(df)
    df_usable = df.dropna(subset=columns).reset_index(drop=True)
    n_dropped = n_before - len(df_usable)
    print(f"{len(df_usable)} minutes usable ({n_dropped} dropped for NaN in this bank's features)")

    results = {}
    for clf_name in classifiers:
        clf_dir = bank_dir / clf_name
        cached = clf_dir / "per_minute_predictions.csv"
        if clf_name == "cnn" and reuse_cached_cnn and cached.exists():
            print(f"\n--- {clf_name}: reusing cached predictions from {cached} ---")
            preds = pd.read_csv(cached)
        else:
            print(f"\n--- running {clf_name} ---")
            clf_dir.mkdir(parents=True, exist_ok=True)
            preds = RUNNERS[clf_name](df_usable, columns)
            preds.to_csv(cached, index=False)

        m = binary_metrics(preds["y_true"], preds["y_pred"])
        print(f"[{clf_name}] {format_metrics(m)}")
        if m["accuracy"] > 0.95:
            print(f"  ** [{clf_name}] SUSPICIOUSLY HIGH accuracy. Check for subject leakage. **")

        clf_dir.mkdir(parents=True, exist_ok=True)
        rec_eval = per_recording_evaluation(preds)
        rec_eval.to_csv(clf_dir / "per_recording_predictions.csv", index=False)
        rec_acc = rec_eval["correct"].mean()
        print(f"[{clf_name}] record-level accuracy: {int(rec_eval['correct'].sum())}/{len(rec_eval)} "
              f"= {rec_acc:.3f}")

        results[clf_name] = {"metrics": m, "record_accuracy": rec_acc, "n_minutes": len(preds)}

    best_name = max(results, key=lambda k: results[k]["metrics"]["accuracy"])
    print(f"\n=== {spec['label']}: best classifier = {best_name} "
          f"(acc={results[best_name]['metrics']['accuracy']:.3f}) ===")
    for name in classifiers:
        m = results[name]["metrics"]
        marker = " <-- WINNER" if name == best_name else ""
        print(f"  {name:8s} {format_metrics(m)}{marker}")

    summary_lines = [
        f"# {spec['label']} -- classifier comparison",
        "",
        f"Feature columns: {columns}",
        f"Minutes evaluated: {len(df_usable)} (subject-independent, 35-fold LOSO)",
        "",
        "| classifier | accuracy | sensitivity | specificity | precision | f1 | record-acc |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in classifiers:
        m = results[name]["metrics"]
        marker = " **<- winner**" if name == best_name else ""
        summary_lines.append(
            f"| {name}{marker} | {m['accuracy']:.3f} | {m['sensitivity']:.3f} | "
            f"{m['specificity']:.3f} | {m['precision']:.3f} | {m['f1']:.3f} | "
            f"{results[name]['record_accuracy']:.3f} |"
        )
    (bank_dir / "classifier_comparison.md").write_text("\n".join(summary_lines) + "\n")
    print(f"Comparison written to {bank_dir}/classifier_comparison.md")

    return {
        "bank": bank_key, "label": spec["label"], "columns": columns,
        "best_classifier": best_name, "results": results,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in BANKS:
        print(f"Usage: python {sys.argv[0]} <{'|'.join(BANKS)}>")
        sys.exit(1)
    run_bank(sys.argv[1])
