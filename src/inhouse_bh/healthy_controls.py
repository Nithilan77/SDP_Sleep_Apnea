"""Apply the deployed breath-hold models to every usable in-house recording (unlabelled; healthy volunteers) and report flag rates.
These recordings carry no breath-hold labels and were not used for training these models. The number is a false-flag-style rate for 'low-effort windows resembling a hold'
(sleep and stillness legitimately produce low effort), NOT a specificity estimate and NOT a statement about apnea.
    python3 -m src.inhouse_bh.healthy_controls"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.inhouse_bh import predict as BH, common as Cm

OUT = Path("results/inhouse_bh/healthy_controls.csv")


def main():
    models = BH.load_models(); rows = []
    for i, p in enumerate(sorted(Path("data/recordings/inhouse_ecg").glob("*.txt"))):
        try:
            x, dur, fs = BH.trace_from_path(p)
        except Exception as e:
            rows.append(dict(code=f"H{i+1:02d}", note=f"skipped: {str(e)[:60]}")); continue
        i0, X, reason = BH.windows_for(x); ok = reason == ""; pr = np.full(len(X), np.nan)
        if ok.any(): pr[ok] = BH.score(X[ok], models)["ENS(mean)"]
        seg = BH.segments(i0 / Cm.HZ, np.nan_to_num(pr) >= BH.THR, Cm.STRIDE_S)
        rows.append(dict(code=f"H{i+1:02d}", hours=round(dur / 3600, 2), windows=len(X), rejected=int((~ok).sum()), flagged_pct=round(100 * float(np.nanmean(pr >= BH.THR)), 2),
                         segments=len(seg), segments_per_hour=round(len(seg) / (dur / 3600), 2), note=""))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
