"""
Phase 3 ablation -- isolates which of v2's two changes (2-channel input,
sensitivity-boosted loss) drove the sensitivity gain over v1.

Runs two variants, same LOSO split and same exact (record, minute) set as
Phase 2 / v1 / v2:

  A) 2-channel (RR + R-amplitude) input, DEFAULT weighting (pos_weight_factor=1.0)
     -- isolates the channel change alone.
  B) 1-channel (RRI only) input, 1.5x sensitivity-boosted weighting
     -- isolates the loss re-weighting alone.

Compares each against v1 (0.775/0.653/0.851) and v2 (0.798/0.754/0.825).

Usage:
    python src/ecg/cnn_apnea_ablation.py
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
from src.eval.metrics import binary_metrics, format_metrics  # noqa: E402
from src.eval.splits import leave_one_subject_out  # noqa: E402

RESULTS_DIR = Path("results/phase3_cnn_ablation")
PHASE2_FEATURES = Path("results/phase2_cvhr_baseline/features.csv")
V1_SEQ_CACHE = Path("results/phase3_cnn/sequences.npz")       # 1-channel RRI
V2_SEQ_CACHE = Path("results/phase3_cnn_v2/sequences.npz")    # 2-channel RR+amp
V1_PER_MINUTE = Path("results/phase3_cnn/per_minute_predictions.csv")
V2_PER_MINUTE = Path("results/phase3_cnn_v2/per_minute_predictions.csv")

EPOCHS = 30
SEED = 0


def load_sequences(cache_path: Path) -> dict:
    if not cache_path.exists():
        raise FileNotFoundError(f"{cache_path} not found -- run its Phase 3 script first.")
    d = np.load(cache_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def restrict_to_phase2_minutes(ds: dict) -> np.ndarray:
    p2 = pd.read_csv(PHASE2_FEATURES).dropna(subset=FEATURE_COLUMNS)
    keys = set(zip(p2["record"], p2["minute"].astype(int)))
    mask = np.array([(r, int(m)) in keys for r, m in zip(ds["record"], ds["minute"])])
    assert mask.sum() == len(p2), "minute sets do not match across phases!"
    return mask


def run_loso(ds: dict, mask: np.ndarray, pos_weight_factor: float, label: str) -> pd.DataFrame:
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

        if X_train.ndim == 3:  # 2-channel: per-channel normalization
            mu = X_train.mean(axis=(0, 2), keepdims=True)
            sd = X_train.std(axis=(0, 2), keepdims=True)
        else:  # 1-channel
            mu, sd = X_train.mean(), X_train.std()
        X_train_n = (X_train - mu) / sd
        X_test_n = (X_test - mu) / sd

        model = train_cnn(X_train_n, y_train, epochs=EPOCHS, seed=SEED,
                           pos_weight_factor=pos_weight_factor)
        y_prob = predict_cnn(model, X_test_n)
        y_pred = (y_prob > 0.5).astype(int)

        all_preds.append(pd.DataFrame({
            "record": rec_col[test_idx], "minute": min_col[test_idx],
            "y_true": y_test.astype(int), "y_pred": y_pred, "y_prob": y_prob,
        }))
        print(f"  [{label}] fold {fold_i:2d}/{len(records)}: held out {test_record} "
              f"({test_idx.sum()} minutes, {time.time()-t0:.1f}s)")

    print(f"  [{label}] total LOSO time: {(time.time()-t_start_all)/60:.1f} min")
    return pd.concat(all_preds, ignore_index=True)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Variant A: 2-channel (RR + R-amplitude), DEFAULT weighting (1.0x) ===")
    ds_a = load_sequences(V2_SEQ_CACHE)
    mask_a = restrict_to_phase2_minutes(ds_a)
    preds_a = run_loso(ds_a, mask_a, pos_weight_factor=1.0, label="A: 2ch/1.0x")
    preds_a.to_csv(RESULTS_DIR / "variant_a_2ch_default_weight.csv", index=False)
    m_a = binary_metrics(preds_a["y_true"], preds_a["y_pred"])

    print("\n=== Variant B: 1-channel (RRI only), 1.5x weighting ===")
    ds_b = load_sequences(V1_SEQ_CACHE)
    mask_b = restrict_to_phase2_minutes(ds_b)
    preds_b = run_loso(ds_b, mask_b, pos_weight_factor=1.5, label="B: 1ch/1.5x")
    preds_b.to_csv(RESULTS_DIR / "variant_b_1ch_weighted.csv", index=False)
    m_b = binary_metrics(preds_b["y_true"], preds_b["y_pred"])

    v1 = pd.read_csv(V1_PER_MINUTE)
    v1_m = binary_metrics(v1["y_true"], v1["y_pred"])
    v2 = pd.read_csv(V2_PER_MINUTE)
    v2_m = binary_metrics(v2["y_true"], v2["y_pred"])

    print("\n=== Ablation summary (per-minute, LOSO, same 16,949 minutes) ===")
    print("v1  (1ch, 1.0x weight):        " + format_metrics(v1_m))
    print("A   (2ch, 1.0x weight):        " + format_metrics(m_a))
    print("B   (1ch, 1.5x weight):        " + format_metrics(m_b))
    print("v2  (2ch, 1.5x weight):        " + format_metrics(v2_m))

    def vs(name, m, ref_name, ref_m):
        print(f"\n{name} vs {ref_name}:")
        print(f"  accuracy:    {m['accuracy'] - ref_m['accuracy']:+.3f}  "
              f"({ref_m['accuracy']:.3f} -> {m['accuracy']:.3f})")
        print(f"  sensitivity: {m['sensitivity'] - ref_m['sensitivity']:+.3f}  "
              f"({ref_m['sensitivity']:.3f} -> {m['sensitivity']:.3f})")
        print(f"  specificity: {m['specificity'] - ref_m['specificity']:+.3f}  "
              f"({ref_m['specificity']:.3f} -> {m['specificity']:.3f})")

    vs("A (2ch, default weight)", m_a, "v1", v1_m)
    vs("B (1ch, 1.5x weight)", m_b, "v1", v1_m)
    vs("A (2ch, default weight)", m_a, "v2", v2_m)
    vs("B (1ch, 1.5x weight)", m_b, "v2", v2_m)

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
