"""Held-out check of the deployed heads on the healthy in-house nights (the ONLY out-of-sample test available).

These recordings contain NO labelled holds and are never trained on, so every window is treated as 'not a hold' and the flag
rate is a false-positive-rate proxy. CAVEATS (stated, not hidden): (1) sleep state/posture unknown and different from the
awake supine breath-hold session, so quiet sleep breathing may genuinely look hold-like (the §7c domain shift); (2) a healthy
person can have brief true pauses, so the proxy is an upper bound on FPR; (3) 10 nights, no labels -> specificity-only.
Nights anonymised S01.. (file order); file names are never written.

    python3 -m src.transfer.healthy_check
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.deploy.predict_session import THR, predict_windows  # noqa: E402
from src.inhouse import streams as S  # noqa: E402
from src.transfer.train_heads import HEADS, OUT  # noqa: E402


def main(stride: float = 30.0):
    bundle = joblib.load(HEADS)
    rows = []
    for i, p in enumerate(S.list_recordings(), 1):
        s = S.build(p, f"S{i:02d}")
        if s is None:
            continue
        d = predict_windows(s, bundle["heads"], stride)
        rows.append(dict(night=f"S{i:02d}", hours=round(s.duration_s / 3600, 2), n_windows=len(d),
                         flag_rate_frozen=float((d.p_frozen >= THR).mean()), flag_rate_handcrafted=float((d.p_handcrafted >= THR).mean()),
                         median_p_frozen=float(d.p_frozen.median()), median_p_handcrafted=float(d.p_handcrafted.median())))
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / "healthy_nights_check.csv", index=False)
    print("\nHELD-OUT healthy nights (no labelled holds; flagged = false-positive proxy; stride %gs)" % stride)
    print(R.round(3).to_string(index=False))
    w = R.n_windows.values
    print(f"\nPooled flag rate: frozen {np.average(R.flag_rate_frozen, weights=w):.3f} | hand-crafted {np.average(R.flag_rate_handcrafted, weights=w):.3f}"
          f" | per-night median: frozen {R.flag_rate_frozen.median():.3f}, hand-crafted {R.flag_rate_handcrafted.median():.3f}")
    print("CAVEAT: sleep vs awake-supine domain shift; n=10 nights; unlabeled (any true pause counts as an error here).")


if __name__ == "__main__":
    main()
