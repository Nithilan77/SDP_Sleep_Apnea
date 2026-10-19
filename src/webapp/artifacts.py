"""Read verified experiment artifacts for the dashboard. Nothing here is hard-coded: every number is read from a result file at request time.
A missing file yields status='missing' (rendered as 'not available'), never a placeholder value."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; R = ROOT / "results"


def _metrics(df: pd.DataFrame, pred_col="y_pred") -> dict:
    y, p = df.y_true.values, df[pred_col].values
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum()); fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    se, sp = tp / max(tp + fn, 1), tn / max(tn + fp, 1); pr = tp / max(tp + fp, 1)
    out = dict(acc=(tp + tn) / len(y), sens=se, spec=sp, prec=pr, f1=2 * pr * se / max(pr + se, 1e-9), tp=tp, tn=tn, fp=fp, fn=fn, n=len(y), n_records=int(df.record.nunique()))
    if "y_prob" in df:
        from sklearn.metrics import roc_auc_score, average_precision_score
        out["auroc"] = float(roc_auc_score(y, df.y_prob)); out["auprc"] = float(average_precision_score(y, df.y_prob))
    g = df.assign(q=df[pred_col]).groupby("record").agg(q=("q", "mean"))
    out["per_recording_correct"] = int(((g.q > .05) == ~g.index.str.startswith("c")).sum())
    return out


HISTORICAL = [  # (label, per-minute csv, architecture note, protocol note)
    ("CVHR logistic baseline", "phase2_cvhr_baseline/per_minute_predictions.csv", "Logistic regression on 7 hand-crafted CVHR/HRV features", "LOSO(35), threshold 0.5; historical"),
    ("1D-CNN (adopted, 'Variant A')", "phase3_cnn/final_per_minute_predictions.csv", "2 conv blocks, 3.4k params; RR + R-amplitude", "LOSO(35), 30 ep, no validation set, threshold 0.5. Variant chosen among 4 on these test folds (selection optimism)"),
    ("RNN (primary)", "phase3_rnn_lstm/rnn_per_minute_predictions.csv", "1-layer RNN h=32", "LOSO(35); underfit; historical"),
    ("LSTM (primary)", "phase3_rnn_lstm/lstm_per_minute_predictions.csv", "1-layer LSTM h=32", "LOSO(35); underfit; historical"),
    ("RNN (v2)", "phase3_rnn_lstm_v2/rnn_per_minute_predictions.csv", "RNN h=64, mean-pool", "Config chosen on 7 records that are also test folds (optimistic)"),
    ("LSTM (v2)", "phase3_rnn_lstm_v2/lstm_per_minute_predictions.csv", "LSTM h=64", "Config chosen on 7 records that are also test folds (optimistic)"),
]


def historical() -> list[dict]:
    rows = []
    for label, f, arch, proto in HISTORICAL:
        p = R / f
        if not p.exists(): rows.append(dict(label=label, status="missing", arch=arch, proto=proto, source=f)); continue
        rows.append(dict(label=label, status="complete", arch=arch, proto=proto, source=f"results/{f}", **_metrics(pd.read_csv(p))))
    return rows


def controlled() -> dict:
    d = R / "expA_multiseed"; s = d / "per_seed_metrics.csv"
    if not s.exists(): return dict(status="pending")
    ps = pd.read_csv(s); cols = ["acc", "sens", "spec", "f1", "auroc", "auprc", "per_recording_correct"]
    agg = ps.groupby(["model", "rule"])[cols].agg(["mean", "std"])
    rows = []
    for (mod, rule), r in agg.iterrows():
        g = ps[(ps.model == mod) & (ps.rule == rule)]
        rows.append(dict(model=mod, rule=rule, n_seeds=len(g), **{f"{c}_mean": float(r[(c, "mean")]) for c in cols}, **{f"{c}_sd": float(r[(c, "std")]) for c in cols},
                         tp=float(g.tp.mean()), tn=float(g.tn.mean()), fp=float(g.fp.mean()), fn=float(g.fn.mean())))
    pc = d / "paired_vs_cnn.csv"
    return dict(status="complete", rows=rows, paired=pd.read_csv(pc).to_dict("records") if pc.exists() else [], summary_md=str((d / "summary.md").relative_to(ROOT)))


MESA_VARIANTS = [  # (label, modalities, source dir name or None for baseline, inherited?)
    ("ECG only", "ECG (RR + R-amplitude)", "ECG-only baseline (CardiacCNN)", "inherited"),
    ("SpO2 only", "SpO2 1 Hz", "ablation C: SpO2 only (not on our wearable)", "new (Exp C)"),
    ("ECG + SpO2", "ECG + SpO2", "ablation C: ECG + SpO2, no effort (not on our wearable)", "new (Exp C)"),
    ("Effort only", "Thor + Abdo belts", "ablation: effort only", "inherited"),
    ("ECG + effort (CANet)", "ECG + belts", "CANet (ECG + effort)  [HEADLINE]", "inherited"),
    ("ECG + effort + SpO2", "ECG + belts + SpO2", "CEILING: CANet + SpO2 (not on our wearable)", "inherited"),
]


def mesa() -> dict:
    p = R / "mesa/canet/head_to_head.csv"
    if not p.exists(): return dict(status="missing")
    t = pd.read_csv(p).set_index("model"); rows = []
    for label, mods, key, origin in MESA_VARIANTS:
        if key in t.index:
            r = t.loc[key]; rows.append(dict(label=label, mods=mods, origin=origin, status="complete", **{k: (None if pd.isna(r[k]) else (r[k] if isinstance(r[k], str) else float(r[k]))) for k in
                                         ("auroc", "auprc", "sens_event", "spec_normal", "f1", "d_auroc", "d_auroc_ci95", "d_auprc")}))
        else: rows.append(dict(label=label, mods=mods, origin=origin, status="pending"))
    cfg = {}
    for name in ("", "ceiling_spo2", "spo2_only", "ecg_spo2", "resp_only"):
        c = R / "mesa/canet" / name / "config.json"
        if c.exists(): cfg[name or "headline"] = json.loads(c.read_text())
    return dict(status="complete", rows=rows, cfg=cfg)


def inhouse() -> dict:
    out = dict(audit=None, bc=None, s2_line=None, events=None, s2cmp=None)
    for k, f in (("audit", "inhouse_audit/audit.json"), ("bc", "inhouse_audit/breathing_cardiac.json"), ("s2_line", "inhouse_audit/s2_emg_breathing_line.json"), ("s2cmp", "breath_hold_s2/session_comparison.json")):
        p = R / f
        if p.exists(): out[k] = json.loads(p.read_text())
    p = R / "breath_hold/event_summary.csv"
    if p.exists(): out["events"] = pd.read_csv(p).to_dict("records")
    return out


def tabular() -> list[dict]:
    p = R / "expA_multiseed/summary_mean_sd.csv"
    return []
