"""
Metrics for apnea-minute classification.

Accuracy alone is misleading here: Apnea-ECG per-minute labels are class-
imbalanced both within records (a01: 470 apnea vs 19 normal) and across the
dataset, so a trivial majority-class classifier can score well on accuracy
while being useless. Always report sensitivity and specificity alongside it.
"""
from __future__ import annotations

import numpy as np


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute accuracy/sensitivity/specificity/precision/F1 for apnea=1, normal=0."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    n = tp + tn + fp + fn

    accuracy = (tp + tn) / n if n else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")  # recall on apnea
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")  # recall on normal
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if (precision + sensitivity) else float("nan"))

    return {
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy, "sensitivity": sensitivity,
        "specificity": specificity, "precision": precision, "f1": f1,
    }


def format_metrics(m: dict) -> str:
    return (f"n={m['n']}  acc={m['accuracy']:.3f}  "
            f"sens={m['sensitivity']:.3f}  spec={m['specificity']:.3f}  "
            f"prec={m['precision']:.3f}  f1={m['f1']:.3f}")
