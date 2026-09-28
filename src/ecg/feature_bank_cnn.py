"""
Feature study -- controlled per-minute apnea classification for a scalar HRV
feature bank (methods 3/4/5: time-domain, frequency-domain, nonlinear), using
the SAME 1D-CNN architecture and training routine as Phase 3
(src/ecg/cnn_model.py: RRCNN, train_cnn, predict_cnn -- completely unmodified,
reused verbatim), so any accuracy difference between methods reflects the
INPUT FEATURES, not the model.

Design decision (feature banks are scalar-per-minute, not a time sequence):
Phase 3's RR-interval / R-amplitude channels are genuine within-minute time
series resampled onto a 60-point grid. An HRV feature bank (SDNN, RMSSD, ...)
is instead a small fixed-length VECTOR of already-aggregated scalars -- there
is no "time" axis within a minute for these. To reuse the exact same RRCNN
class/training code unmodified (as the comparison requires), each minute's
feature vector is treated as a 1-channel length-K "sequence" (K = number of
features in the bank), matching the same (n, 1, seq_len) call signature
train_cnn/predict_cnn already expect for the 1-channel RRI case. This is a
deliberate, documented adaptation, not a claim that HRV features have
temporal structure -- with kernel_size=5, RRCNN's first conv layer sees a
small local mix of nearby feature indices then global-average-pools, so it
behaves close to a small MLP over the feature vector.

Normalization: per-feature-column standardization (mean/std per column, fit
on the training fold only -- no test-fold leakage), NOT the single scalar
per-channel mu/sd Phase 3 used for its homogeneous RR/amplitude channels.
That coarser normalization is wrong here because bank features have very
different native scales (e.g. pnn50 in [0,1] vs sdnn_ms in tens-to-hundreds)
-- collapsing them to one mean/std would drown the small-scale features.

Same 35-fold LOSO discipline, same eval metrics, same minute set (joined
against the Phase 2 cached feature table) as every other method in this
project.

Usage:
    python src/ecg/feature_bank_cnn.py time        # bank 3
    python src/ecg/feature_bank_cnn.py freq        # bank 4
    python src/ecg/feature_bank_cnn.py nonlinear   # bank 5
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ecg.cnn_model import predict_cnn, train_cnn  # noqa: E402
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


def run_loso(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    df = df.dropna(subset=feature_columns).reset_index(drop=True)
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

        # (n, seq_len) -> (n, 1, seq_len): one channel, feature index as "sequence".
        X_train_n = X_train_n[:, None, :]
        X_test_n = X_test_n[:, None, :]

        model = train_cnn(X_train_n, y_train, seq_len=seq_len, epochs=EPOCHS, seed=SEED)
        y_prob = predict_cnn(model, X_test_n)
        y_pred = (y_prob > 0.5).astype(int)

        all_preds.append(pd.DataFrame({
            "record": test_df["record"].values, "minute": test_df["minute"].values,
            "y_true": y_test, "y_pred": y_pred, "y_prob": y_prob,
        }))
        print(f"  fold {fold_i:2d}/{len(records)}: held out {test_record} "
              f"({len(test_df)} minutes, {time.time()-t0:.1f}s)")

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


def run_bank(bank_key: str) -> dict:
    spec = BANKS[bank_key]
    columns = spec["columns"]
    slug = spec["slug"]
    out_dir = RESULTS_ROOT / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {spec['label']}: {columns} ===")
    df = load_or_build_bank_features()
    df = restrict_to_phase2_minutes(df)

    n_before = len(df)
    df_usable = df.dropna(subset=columns)
    n_dropped = n_before - len(df_usable)
    print(f"{len(df_usable)} minutes usable ({n_dropped} dropped for NaN in this bank's features)")

    preds = run_loso(df_usable, columns)
    preds.to_csv(out_dir / "per_minute_predictions.csv", index=False)

    m = binary_metrics(preds["y_true"], preds["y_pred"])
    print(f"\n=== {spec['label']} result (subject-independent, LOSO, {df_usable['record'].nunique()} folds) ===")
    print(format_metrics(m))
    if m["accuracy"] > 0.95:
        print("  ** SUSPICIOUSLY HIGH accuracy. Check for subject leakage before trusting this. **")

    rec_eval = per_recording_evaluation(preds)
    rec_eval.to_csv(out_dir / "per_recording_predictions.csv", index=False)
    print(f"record-level accuracy: {int(rec_eval['correct'].sum())}/{len(rec_eval)} "
          f"= {rec_eval['correct'].mean():.3f}")

    print(f"Results saved to {out_dir}/")
    return {"bank": bank_key, "label": spec["label"], "metrics": m, "n_minutes": len(df_usable)}


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in BANKS:
        print(f"Usage: python {sys.argv[0]} <{'|'.join(BANKS)}>")
        sys.exit(1)
    run_bank(sys.argv[1])
