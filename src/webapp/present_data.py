"""Numbers for the presentation dashboard — all read from saved result files at request time (nothing typed in)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[2] / "results"


def _acc(f):
    d = pd.read_csv(f); return float((d.y_true == d.y_pred).mean())


def physionet() -> dict:
    ps = pd.read_csv(R / "expA_multiseed/per_seed_metrics.csv"); ps = ps[ps.rule == "thr0.5"]
    names = {"cnn": "Neural net (1D-CNN)", "tcn": "Neural net (TCN)", "gru": "Neural net (GRU)", "lstm": "Neural net (LSTM)", "hgb_within": "Gradient boosting (HRV features)", "logreg_within": "Logistic regression (HRV features)"}
    g = ps.groupby("model")[["acc", "auroc"]].mean(); rows = [dict(name=names[k], acc=float(g.loc[k, "acc"]), auroc=float(g.loc[k, "auroc"])) for k in names if k in g.index]
    rows.sort(key=lambda r: r["name"])                                  # alphabetical: no ranking implied
    return dict(cvhr=_acc(R / "phase2_cvhr_baseline/per_minute_predictions.csv"), cnn_hist=_acc(R / "phase3_cnn/final_per_minute_predictions.csv"), controlled=rows,
                n_records=35, n_minutes=int(len(pd.read_csv(R / "phase2_cvhr_baseline/per_minute_predictions.csv"))))


def mesa() -> list[dict]:
    a = pd.read_csv(R / "mesa/canet/head_to_head.csv"); b = pd.read_csv(R / "mesa/canet_lstm/head_to_head.csv"); out = []
    def add(name, auroc, spo2, group): out.append(dict(name=name, auroc=float(auroc), spo2=spo2, group=group))
    m = {r.model: r.auroc for r in a.itertuples()}
    add("ECG only", m["ECG-only baseline (CardiacCNN)"], False, "Single signal")
    add("Breathing effort only", m["ablation: effort only"], False, "Single signal")
    add("SpO2 only", m["ablation C: SpO2 only (not on our wearable)"], True, "Single signal")
    add("ECG + effort, simple join", m["ablation: ECG + effort, concat (no attention)"], False, "Combined")
    add("ECG + effort, cross-attention (CANet)", m["CANet (ECG + effort)  [HEADLINE]"], False, "Combined")
    add("ECG + SpO2", m["ablation C: ECG + SpO2, no effort (not on our wearable)"], True, "Combined")
    add("ECG + effort + SpO2", m["CEILING: CANet + SpO2 (not on our wearable)"], True, "Combined")
    l = {r.model: r.auroc for r in b.itertuples()}
    add("ECG + SpO2, CNN-LSTM", l["CNN-LSTM ecg_spo2_lstm"], True, "Recurrent variants")
    add("ECG + effort + SpO2, CNN-LSTM", l["CNN-LSTM ecg_resp_spo2_lstm"], True, "Recurrent variants")
    return out


def domain_gap() -> dict:
    s = pd.read_csv(R / "inhouse_canet/negative_control/summary_by_model.csv"); h = s[(s.model == "headline") & (s.effort_source == "mag")].iloc[0]
    return dict(ours=float(h.frac_flagged_median), mesa_ref=float(h.mesa_lowAHI_frac_flagged_median), nights=int(h.n_nights))


def transfer() -> dict:
    r = json.loads((R / "transfer/results.json").read_text()); f, h = r["frozen_CANet_256"], r["handcrafted_8"]
    n = pd.read_csv(R / "transfer/healthy_nights_check.csv"); w = n.hours / n.hours.sum()
    return dict(frozen=dict(auroc=f["pooled_auroc"], ci=f["boot_ci95"], sens=f["sens"], spec=f["spec"], fold_min=f["per_fold_auroc_min"], fpr=f["baseline_fpr_mean"]),
                hand=dict(auroc=h["pooled_auroc"], ci=h["boot_ci95"], sens=h["sens"], spec=h["spec"], fold_min=h["per_fold_auroc_min"], fpr=h["baseline_fpr_mean"]),
                nights=[dict(night=f"Night {i+1}", hours=float(x.hours), frozen=float(x.flag_rate_frozen), hand=float(x.flag_rate_handcrafted)) for i, x in enumerate(n.itertuples())],
                pooled_frozen=float((n.flag_rate_frozen * w).sum()), pooled_hand=float((n.flag_rate_handcrafted * w).sum()), n_nights=len(n), hours_total=float(n.hours.sum()),
                hours_min=float(n.hours.min()), hours_max=float(n.hours.max()))
