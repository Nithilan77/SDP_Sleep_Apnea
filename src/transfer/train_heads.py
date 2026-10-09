"""§7f deployment heads: fit the two linear heads on ALL 7 labelled holds of the breath-hold session.

IN-SAMPLE by construction: these heads have seen every hold in this session, so predictions on that same session are NOT
a performance estimate (the honest estimates are the leave-one-event-out numbers in results/transfer/results.json and
the healthy-night check). The MESA CANet encoders stay frozen; only two L2 logistic-regression heads are fitted
(scoped exception to CLAUDE.md rule 5, docs/PROGRESS.md §7f).

    python3 -m src.transfer.train_heads
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.inhouse import streams as S  # noqa: E402
from src.ingest.sensortile import load_sensortile  # noqa: E402
from src.transfer import run_transfer as T  # noqa: E402
from src.transfer import windows as W_  # noqa: E402

OUT = Path("results/transfer")
HEADS = OUT / "deployed_heads.joblib"


def main():
    Wdf, _ = W_.build_index()
    s = S.build(W_.REC_PATH, "S00")
    card, eff = T.build_inputs(s, Wdf.c.values)
    F, _, _, _ = T.frozen_features(card, eff)
    H = T.handcrafted(s, Wdf.c.values)
    tr = Wdf.kind.isin(["pos", "rest"]).values
    y = Wdf.label.values[tr]
    heads = {"frozen_256": T.head(T.C_PRIMARY).fit(F[tr], y), "handcrafted_8": T.head(T.C_PRIMARY).fit(H.values[tr], y)}
    rec = load_sensortile(W_.REC_PATH)
    meta = dict(trained_on=W_.REC_PATH.name, trained_duration_s=float(rec.duration_s), n_train_windows=int(tr.sum()),
                n_pos=int(y.sum()), n_rest=int((1 - y).sum()), C=T.C_PRIMARY, hand_features=list(H.columns), threshold=0.5,
                note="IN-SAMPLE for the training session; encoders frozen; only the two linear heads are fitted.")
    joblib.dump(dict(heads=heads, meta=meta), HEADS)
    ins = {k: dict(sens=float((h.predict_proba(X[tr])[:, 1][y == 1] >= .5).mean()), spec=float((h.predict_proba(X[tr])[:, 1][y == 0] < .5).mean()))
           for k, h, X in (("frozen_256", heads["frozen_256"], F), ("handcrafted_8", heads["handcrafted_8"], H.values))}
    meta["in_sample_train_fit"] = ins
    json.dump(meta, open(OUT / "deployed_heads_meta.json", "w"), indent=1)
    print(f"Fitted on {tr.sum()} windows ({y.sum()} hold / {(1-y).sum()} rest) from all 7 events -> {HEADS}")
    print("IN-SAMPLE training-set fit (NOT a performance estimate):", {k: {a: round(b, 2) for a, b in v.items()} for k, v in ins.items()})


if __name__ == "__main__":
    main()
