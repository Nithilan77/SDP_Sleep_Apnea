"""End-to-end deployment wrapper: raw SensorTile recording -> per-window hold/cessation-likely probability.

    python3 -m src.deploy.predict_session <recording.csv|.txt> [--stride 5] [--out DIR] [--events results/breath_hold/event_summary.csv]

Pipeline (no manual steps): load_sensortile -> R-peaks + accelerometer breathing trace -> MESA-belt-style preprocessing ->
150 s windows -> FROZEN MESA-trained CANet (cardiac + effort, cross-attention) pooled features -> linear head A,
and the 8 hand-crafted effort features of the 30 s centre epoch -> linear head B.

Two heads are shown side by side. Per docs/PROGRESS.md §7f the hand-crafted head is the steadier detector at this data scale;
the frozen-feature head is reported for comparison, NOT because it won (pooled AUROC ties; frozen is less consistent per event
and flags 24% of the baseline period). Frozen (150 s, cardiac+effort) vs hand-crafted (30 s, effort only) is not apples-to-apples.
Both heads were trained on ALL 7 holds of the 2026-10-07 session: predictions on that session are IN-SAMPLE, not a performance
estimate. Pilot scale: one subject, one session, 7 events. Screening/plausibility only, not a diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.inhouse import streams as S  # noqa: E402
from src.transfer import run_transfer as T  # noqa: E402
from src.transfer.train_heads import HEADS  # noqa: E402

THR = 0.5
EPOCH_S = 30.0


def predict_windows(s: S.InhouseStreams, heads: dict, stride_s: float = 5.0) -> pd.DataFrame:
    span = s.n_epochs * EPOCH_S
    c = np.arange(EPOCH_S / 2, span - EPOCH_S / 2 + 1e-9, stride_s)
    card, eff = T.build_inputs(s, c)
    F, p_mesa, _, _ = T.frozen_features(card, eff)
    H = T.handcrafted(s, c)
    ctx = 75.0
    return pd.DataFrame(dict(
        t_centre_s=c, p_frozen=heads["frozen_256"].predict_proba(F)[:, 1],
        p_handcrafted=heads["handcrafted_8"].predict_proba(H.values)[:, 1],
        pad_frac=(np.maximum(0, ctx - c) + np.maximum(0, c + ctx - span)) / (2 * ctx)))


def segments(t: np.ndarray, flag: np.ndarray, stride: float):
    out, i = [], 0
    while i < len(flag):
        if flag[i]:
            j = i
            while j + 1 < len(flag) and flag[j + 1]:
                j += 1
            out.append((float(t[i] - EPOCH_S / 2), float(t[j] + EPOCH_S / 2)))
            i = j + 1
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording"); ap.add_argument("--stride", type=float, default=5.0)
    ap.add_argument("--out", default=None, help="output dir (default results/transfer/predict/<hash of file name>)")
    ap.add_argument("--events", default=None, help="optional event_summary.csv to overlay known inner hold windows")
    a = ap.parse_args()
    path = Path(a.recording)
    bundle = joblib.load(HEADS); meta = bundle["meta"]
    s = S.build(path, "S00")
    if s is None:
        sys.exit(f"Recording too short (< {S.MIN_DURATION_S:.0f}s) or unreadable: {path.name}")
    df = predict_windows(s, bundle["heads"], a.stride)
    in_sample = path.name == meta["trained_on"]
    tag = "IN-SAMPLE (heads were trained on this session's 7 holds - NOT a performance estimate)" if in_sample else "OUT-OF-SAMPLE (heads never saw this recording)"
    out = Path(a.out or f"results/transfer/predict/{hashlib.sha1(path.name.encode()).hexdigest()[:8]}"); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_window_predictions.csv", index=False)

    print("=" * 96)
    print(f"Session: {s.duration_s/60:.1f} min, fs {s.fs:.1f} Hz, {len(df)} windows (30 s centre epoch, 150 s CANet input, stride {a.stride:g} s)")
    print(f"STATUS: {tag}")
    print("Pilot scale: heads fitted on 1 subject / 1 session / 7 holds. Screening plausibility only, not a diagnostic.")
    print("Heads: [A] frozen MESA-CANet features + linear head | [B] hand-crafted 30 s effort features + linear head")
    print("  -> per PROGRESS.md §7f, B is the steadier detector at this data scale; A is shown for comparison, not because it won.")
    print("  -> A (150 s, cardiac+effort) vs B (30 s, effort only) is NOT apples-to-apples.")
    print("=" * 96)
    for nm, col in (("A frozen", "p_frozen"), ("B hand-crafted", "p_handcrafted")):
        fl = df[col].values >= THR
        seg = segments(df.t_centre_s.values, fl, a.stride)
        print(f"\n[{nm}] windows flagged hold-like (p >= {THR}): {fl.sum()}/{len(df)} ({100*fl.mean():.1f}%) ; {len(seg)} contiguous segment(s)")
        for k, (t0, t1) in enumerate(seg[:20], 1):
            print(f"   {k:>2}. {t0:7.0f}s - {t1:7.0f}s   ({t1 - t0:.0f}s span of 30 s epochs)")
        if len(seg) > 20:
            print(f"   ... {len(seg) - 20} more")
    if (df.pad_frac > 0).any():
        print(f"\nNote: {(df.pad_frac>0).sum()} edge windows have zero-padded context (max {df.pad_frac.max():.0%}); treat those with caution.")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1, 1]))
    tt = np.arange(len(s.effort_raw["mag"])) / 32.0
    ax[0].plot(tt, s.effort_raw["mag"], lw=0.4, color="#43A047"); ax[0].set_ylabel("chest accel,\nbreathing band (g)")
    for k, (col, lab, colr) in enumerate((("p_handcrafted", "B hand-crafted head (steadier at this data scale)", "#1976D2"),
                                          ("p_frozen", "A frozen MESA-CANet features head (comparison)", "#EF6C00")), start=1):
        ax[k].plot(df.t_centre_s, df[col], color=colr, lw=1.4); ax[k].axhline(THR, color="grey", ls=":", lw=1)
        ax[k].set_ylim(-0.02, 1.02); ax[k].set_ylabel(f"P(hold-like)\n{lab[:1]}"); ax[k].set_title(lab, fontsize=9, loc="left")
    if a.events and Path(a.events).exists():
        ev = pd.read_csv(a.events)
        for _, e in ev.iterrows():
            for x in ax:
                x.axvspan(e.fixedpad_start, e.fixedpad_end, color="grey", alpha=0.18, label="labelled hold (inner window)" if e.event == 1 and x is ax[0] else None)
        ax[0].legend(loc="upper left", fontsize=8)
    ax[2].set_xlabel("recording time (s)")
    fig.suptitle(("IN-SAMPLE - heads trained on these 7 holds; not a performance estimate" if in_sample else "OUT-OF-SAMPLE session") +
                 "  |  pilot: 1 subject / 1 session / 7 events  |  screening plausibility, not a diagnostic", fontsize=10, color="#B71C1C" if in_sample else "black")
    if in_sample:
        cap = ("CAUTION: both heads were fitted on these same 7 holds, so Head A's strong peaks here are overfitting, not accuracy.\n"
               "The honest out-of-fold (leave-one-event-out) comparison is in docs/PROGRESS.md \u00a77f and must ALWAYS be shown alongside this plot:\n"
               "frozen A and hand-crafted B tie on pooled AUROC (0.83 vs 0.83); A is less consistent per event (min fold 0.16 vs 0.78) "
               "and has 24% vs 0.5% false positives on the baseline period. Pilot: 1 subject, 7 events.")
    else:
        cap = ("Pilot-scale heads (1 subject, 7 holds). Out-of-fold comparison of heads A vs B: docs/PROGRESS.md \u00a77f - show alongside this plot.")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.text(0.01, 0.01, cap, fontsize=8.5, color="#B71C1C" if in_sample else "#444444", va="bottom", ha="left", wrap=True,
             bbox=dict(boxstyle="round", fc="#FFF3E0" if in_sample else "#F5F5F5", ec="#B71C1C" if in_sample else "#999999"))
    fig.savefig(out / "predictions.png", dpi=120); plt.close(fig)
    print(f"\nSaved: {out/'per_window_predictions.csv'}, {out/'predictions.png'}")


if __name__ == "__main__":
    main()
