"""Step 1 of §7f: labelled window index for supervised transfer on the breath-hold session.

Window = one 150 s CANet input (target 30 s epoch +/- 2 context epochs), identified by its centre time c.
Label comes from the 30 s target epoch [c-15, c+15] vs the §7e INNER hold windows (nominal inset by the sync lag):
  positive : >= 15 s of the target epoch lies inside an inner hold            -> group = event id
  rest     : target epoch >= 15 s clear of every inner hold AND inside a between-hold gap -> group = gap id
  baseline : target epoch >= 15 s clear of hold 1 AND the whole 150 s input ends before inner hold 1 starts
             (separate secondary test, never pooled)
  (anything else, i.e. transition zones, is dropped)
Stride 5 s. Every window is tagged with its event (positives) or nearest event (rest), so CV can be leave-one-EVENT-out.

    python3 -m src.transfer.windows        # prints counts per event and per fold; no model is touched
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REC_PATH = Path("data/recordings/breath_hold/user_MukuBreathHold_ECG_07_10_2026.csv")
EVENTS_CSV = Path("results/breath_hold/event_summary.csv")
OUT = Path("results/transfer")
EPOCH_S, CTX_S, STRIDE_S, PAD_S = 30.0, 75.0, 5.0, 15.0
MIN_POS_OVERLAP_S = 15.0     # 50% of the 30 s target epoch
CLEAR_S = 15.0               # rest/baseline target epoch must be this far from any inner hold


def usable_span_s() -> float:
    """Streams are built on whole 30 s epochs (src/inhouse/streams.py): int(duration // 30) * 30."""
    from src.ingest.sensortile import load_sensortile
    return int(load_sensortile(REC_PATH).duration_s // EPOCH_S) * EPOCH_S


def build_index(span_s: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    span_s = span_s or usable_span_s()
    ev = pd.read_csv(EVENTS_CSV)[["event", "fixedpad_start", "fixedpad_end"]]
    hs, he = ev.fixedpad_start.values, ev.fixedpad_end.values
    mid = (hs + he) / 2
    rows = []
    for c in np.arange(EPOCH_S / 2, span_s - EPOCH_S / 2 + 1e-9, STRIDE_S):
        a, b = c - EPOCH_S / 2, c + EPOCH_S / 2
        ov = np.clip(np.minimum(b, he) - np.maximum(a, hs), 0, None)          # overlap with each hold (s)
        dist = np.maximum(hs - b, a - he)                                     # >0: gap between epoch and hold
        in_in, out_in = c - CTX_S, c + CTX_S
        pad_frac = (max(0, -in_in) + max(0, out_in - span_s)) / (2 * CTX_S)    # share of the 150 s input that is zero-padding
        kind, grp = "drop", -1
        if ov.max() >= MIN_POS_OVERLAP_S:
            kind, grp = "pos", int(ev.event.values[ov.argmax()])
        elif dist.min() >= CLEAR_S:
            gap_ok = (c > he.min()) and (c < hs.max())
            if gap_ok:
                kind, grp = "rest", int(ev.event.values[np.abs(mid - c).argmin()])   # nearest event (test-fold assignment)
            elif out_in <= hs[0]:
                kind, grp = "baseline", 0
        rows.append(dict(c=c, kind=kind, event=grp, label=int(kind == "pos"), max_overlap_s=float(ov.max()),
                         min_dist_s=float(dist.min()), pad_frac=pad_frac))
    W = pd.DataFrame(rows)
    W = W[W.kind != "drop"].reset_index(drop=True)
    W["in_start"], W["in_end"] = W.c - CTX_S, W.c + CTX_S
    return W, ev


def fold_masks(W: pd.DataFrame, ev: pd.DataFrame) -> dict:
    """Leave-one-event-out. Test = positives of event k + rest windows nearest to event k (each window tested exactly once).
    Train = every other pos/rest window, MINUS windows whose full 150 s input overlaps [hold_k start-PAD, hold_k end+PAD]
    (purge), MINUS windows whose 30 s target epoch overlaps a test window's target epoch. Baseline is test-only.
    Returns {event: (train_row_idx, test_row_idx, n_purged)} indexing into W."""
    out = {}
    prim = W[W.kind.isin(["pos", "rest"])]
    for _, e in ev.iterrows():
        k = int(e.event)
        lo, hi = e.fixedpad_start - PAD_S, e.fixedpad_end + PAD_S
        test = prim[prim.event == k]
        rest = prim[prim.event != k]
        overl = (rest.in_end > lo) & (rest.in_start < hi)
        t_lo, t_hi = test.c.min() - EPOCH_S / 2, test.c.max() + EPOCH_S / 2
        same = (rest.c + EPOCH_S / 2 > t_lo) & (rest.c - EPOCH_S / 2 < t_hi)
        out[k] = (rest.index[~overl & ~same].values, test.index.values, int((overl | same).sum()))
    return out


def fold_table(W: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, (tr, te, npg) in fold_masks(W, ev).items():
        tr_, te_ = W.loc[tr], W.loc[te]
        rows.append(dict(fold_event=k, test_pos=int((te_.kind == "pos").sum()), test_rest=int((te_.kind == "rest").sum()),
                         train_pos=int((tr_.kind == "pos").sum()), train_rest=int((tr_.kind == "rest").sum()),
                         purged=npg, train_pos_events=int(tr_[tr_.kind == "pos"].event.nunique())))
    return pd.DataFrame(rows)


def main():
    W, ev = build_index()
    OUT.mkdir(parents=True, exist_ok=True)
    W.to_csv(OUT / "window_index.csv", index=False)
    print(f"usable span {usable_span_s():.0f}s | stride {STRIDE_S:.0f}s | windows kept: {len(W)}")
    print("\nWINDOWS PER EVENT  (pos = in-hold windows; rest = between-hold rest windows nearest that event)")
    t = W[W.kind.isin(["pos", "rest"])].pivot_table(index="event", columns="kind", values="c", aggfunc="count", fill_value=0)
    t["pad_frac_max"] = W[W.kind.isin(["pos", "rest"])].groupby("event").pad_frac.max().round(2)
    print(t.to_string())
    print(f"\nBASELINE (secondary, test-only): {int((W.kind == 'baseline').sum())} windows, "
          f"c from {W[W.kind=='baseline'].c.min():.0f}s to {W[W.kind=='baseline'].c.max():.0f}s")
    F = fold_table(W, ev)
    F.to_csv(OUT / "fold_sizes.csv", index=False)
    print("\nLEAVE-ONE-EVENT-OUT FOLD SIZES (after purge)")
    print(F.to_string(index=False))
    print(f"\nTotals: positives {int((W.kind=='pos').sum())}, rest {int((W.kind=='rest').sum())} "
          f"(independent units: 7 events)")
    print("NOTE: n = 7 events from ONE subject / ONE session. Windows from one event overlap; this is a pilot-scale design.")


if __name__ == "__main__":
    main()
