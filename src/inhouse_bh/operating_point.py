"""Operating-point analysis of the Track-B hold detector over WHOLE-recording behaviour (leave-one-event-out, seed 0).
Per fold the model scores (a) the held-out block's labelled windows and (b) every unlabelled window of the recording (stride 2 s).  Segment-level metrics:
  event detected   = a flagged segment (>= MIN_RUN consecutive windows) overlaps >= 50% of the event's hold core
  false segment    = a flagged segment that overlaps NO annotated hold (nominal start-10 s .. nominal end+10 s)  [per hour of recording]
Because all 7 events are used both to build and to evaluate, sweeping thresholds here is OPTIMISTIC; the deployed threshold is NOT picked from this sweep but derived beforehand
from class priors:  models are trained with balanced weights (effective prior 0.5) while a protocol recording is ~12% hold time (210 s of 1796 s) -> odds shift 0.12/0.88 -> p >= 0.88.
    python3 -m src.inhouse_bh.operating_point --configs real aug+synth
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
os.environ.setdefault("SYNTH_SPEC_DIR", "results/synth/spec_noS1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.inhouse_bh import common as Cm, synthetic as Syn, run_experiments as R, predict as BH

OUT = Path("results/inhouse_bh")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--configs", nargs="+", default=["real", "aug+synth"]); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    x, dur, fs = Cm.trace32(Cm.PATHS["S1_ECGTILE"]); w, _ = Cm.labelled_windows("S1_ECGTILE", x); X = Cm.window_array(x, w.i0.values, w.scale.values); y = (w.label == "voluntary_breath_hold").astype(int).values; ev = w.event_no.values
    syn = Syn.synthetic_windows(100, 4242)[:2]
    i0 = np.arange(0, len(x) - Cm.W, int(2 * Cm.HZ)); Xa = Cm.window_array(x, i0, Cm.local_scales(x, i0)); t0 = i0 / Cm.HZ
    ann = pd.read_csv(Cm.ANN); ann = ann[(ann.recording_id == "S1_ECGTILE") & (ann.time_base == "recording")]
    cores = ann[ann.label == "voluntary_breath_hold"][["event_no", "start_s", "end_s"]].to_numpy()
    in_block = {k: (t0 >= ann[ann.event_no == k].start_s.min() - 130) & (t0 <= ann[ann.event_no == k].end_s.max() + 30) for k in range(1, 8)}
    rows = []
    for cfg in a.configs:
        P = {m: np.full((7, len(Xa)), np.nan) for m in ("LR", "HGB", "CNN", "ENS(mean)")}
        for k in range(1, 8):
            tr = ev != k; preds = R.run_fold(cfg, X[tr], y[tr], Xa, a.seed, syn, None); preds["ENS(mean)"] = np.mean([preds[m] for m in ("LR", "HGB", "CNN")], 0)
            for m, v in preds.items(): P[m][k - 1] = v
        # each time point's prediction comes from the fold that did NOT train on that point's event block: use fold of nearest event (by time); unlabelled time uses the mean over folds
        nearest = np.array([int(np.argmin([abs(t - (c[1] + c[2]) / 2) for c in cores])) for t in t0])
        for m, arr in P.items():
            p = np.where(np.isnan(arr).all(0), np.nan, arr[nearest, np.arange(len(t0))])
            np.save(OUT / f"op_pred_{cfg}_{m}.npy", p)
            for thr in (0.5, 0.7, 0.8, 0.88, 0.95):
                for min_run in (3, 5, 7):
                    seg = BH.segments(t0, p >= thr, 2.0, min_run); det = 0
                    for _, s, e in cores:
                        core = (s, e); ov = max((min(b, e) - max(a_, s) for a_, b in seg), default=0)
                        det += ov >= 0.5 * (e - s)
                    fseg = sum(1 for a_, b in seg if not any(a_ < e + 10 and b > s - 10 for _, s, e in cores))
                    rows.append(dict(config=cfg, model=m, thr=thr, min_run=min_run, events_detected=int(det), n_segments=len(seg), false_segments=int(fseg), false_segments_per_hour=round(fseg / (dur / 3600), 1)))
        print(cfg, "done", flush=True)
    D = pd.DataFrame(rows); D.to_csv(OUT / "operating_points.csv", index=False)
    print(D[(D.model == "ENS(mean)")].to_string())


if __name__ == "__main__":
    main()
