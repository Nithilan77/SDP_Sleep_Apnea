"""§7f: frozen-CANet features vs hand-crafted effort features -> L2 logistic regression, leave-one-EVENT-out.

PILOT SCALE: n = 7 events, ONE subject, ONE session. Not a validated accuracy.
The ONLY trained parameters are the logistic-regression heads. The MESA CANet is frozen (eval, no grad).
COMPARISON CAVEAT: frozen features see a 150 s window (cardiac+effort, cross-attention); the hand-crafted baseline sees only the
30 s centre epoch of the effort trace (what epoch_features is defined on). This is NOT apples-to-apples.

    python3 -m src.transfer.run_transfer
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.inhouse import streams as S  # noqa: E402
from src.inhouse.run_canet import epoch_features  # noqa: E402
from src.mesa import canet as N  # noqa: E402
from src.mesa import respiratory_stream as R  # noqa: E402
from src.transfer import windows as W_  # noqa: E402

OUT = Path("results/transfer")
CACHE = Path("data/recordings/cache/transfer_feats.npz")     # gitignored (data/recordings/)
MODEL = Path("data/mesa/cache/final_models/headline.pt")
C_PRIMARY = 0.01          # strong L2, fixed a priori (not tuned); sensitivity to C reported separately
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PAD_TEST_S = 40.0         # padding-control: zero the last 40 s of the 150 s input (Event 7 has up to 40 s)
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------- inputs
def slice_pad(x: np.ndarray, start: int, n: int, fill: float) -> np.ndarray:
    """x: (..., T); returns (..., n) from [start, start+n) with `fill` outside [0, T)."""
    T = x.shape[-1]
    out = np.full(x.shape[:-1] + (n,), fill, dtype=np.float32)
    a, b = max(start, 0), min(start + n, T)
    if b > a:
        out[..., a - start:b - start] = x[..., a:b]
    return out


def build_inputs(s: S.InhouseStreams, c: np.ndarray, tail_pad_s: float = 0.0):
    rr = np.nan_to_num(s.rr); amp = np.nan_to_num(s.amp, nan=1.0)
    card, eff = [], []
    for ci in c:
        i0 = int(round((ci - W_.CTX_S) * 2.0)); j0 = int(round((ci - W_.CTX_S) * R.BELT_HZ))
        cw = np.stack([slice_pad(rr, i0, 300, 0.0), slice_pad(amp, i0, 300, 1.0)])
        cw[0] = np.clip(cw[0], -1, 1); cw[1] = np.clip(cw[1] - 1, -3, 3)
        ew = slice_pad(s.effort["mag"].astype(np.float32), j0, 4800, 0.0)
        if tail_pad_s > 0:                                    # padding control: neutral values in the last tail_pad_s
            cw[:, 300 - int(tail_pad_s * 2):] = 0.0; ew[:, 4800 - int(tail_pad_s * R.BELT_HZ):] = 0.0
        card.append(cw); eff.append(ew)
    return np.stack(card), np.stack(eff)


@torch.no_grad()
def frozen_features(card: np.ndarray, eff: np.ndarray, bs: int = 64):
    ck = torch.load(MODEL, map_location=DEV, weights_only=False)
    m = N.CANet(ck["streams"], ck["fusion"]).to(DEV); m.load_state_dict(ck["state"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    feats, probs = [], []
    for i in range(0, len(card), bs):
        xs = {"cardiac": torch.as_tensor(card[i:i + bs], device=DEV), "resp": torch.as_tensor(eff[i:i + bs], device=DEV)}
        t = {s_: m.enc[s_](xs[s_]) for s_ in m.streams}
        if m.cross is not None:
            t = m.cross(t)
        z = torch.cat([torch.cat([t[s_].mean(1), t[s_].amax(1)], 1) for s_ in m.streams], 1)
        feats.append(z.cpu().numpy()); probs.append(torch.sigmoid(m.head(z).squeeze(-1)).cpu().numpy())
    return np.concatenate(feats), np.concatenate(probs), list(ck["streams"]), float(ck["thr"])


def handcrafted(s: S.InhouseStreams, c: np.ndarray) -> pd.DataFrame:
    X = np.stack([slice_pad(s.effort["mag"][0].astype(np.float32), int(round((ci - 15) * R.BELT_HZ)), 960, 0.0) for ci in c])
    return epoch_features(X)


# ----------------------------------------------------------------------------- CV
def head(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))  # L2 is the default


def sens_spec(y, p, thr=0.5):
    pr = p >= thr
    return (float(pr[y == 1].mean()) if (y == 1).any() else np.nan), (float((~pr[y == 0]).mean()) if (y == 0).any() else np.nan)


def auroc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def loeo(X, Wdf, masks, C, base_X=None, pad_X=None, shuffle_labels=False):
    """X: (n_windows, d) features aligned with Wdf rows. Returns OOF frame, per-fold baseline FPR, padded-test OOF."""
    y = Wdf.label.values
    oof, base_fpr, padded = [], [], []
    bidx = np.where(Wdf.kind.values == "baseline")[0]
    for k, (tr, te, _) in masks.items():
        ytr = RNG.permutation(y[tr]) if shuffle_labels else y[tr]
        h = head(C).fit(X[tr], ytr)
        p = h.predict_proba(X[te])[:, 1]
        oof.append(pd.DataFrame(dict(fold=k, row=te, c=Wdf.c.values[te], y=y[te], p=p, kind=Wdf.kind.values[te], pad_frac=Wdf.pad_frac.values[te])))
        if base_X is not None:
            base_fpr.append(dict(fold=k, base_fpr=float((h.predict_proba(base_X)[:, 1] >= 0.5).mean())))
        if pad_X is not None:
            pp = h.predict_proba(pad_X[te])[:, 1]
            padded.append(pd.DataFrame(dict(fold=k, row=te, y=y[te], p=pp)))
    return pd.concat(oof, ignore_index=True), pd.DataFrame(base_fpr), (pd.concat(padded, ignore_index=True) if padded else None)


def summarize(oof: pd.DataFrame, name: str) -> dict:
    y, p = oof.y.values, oof.p.values
    se, sp = sens_spec(y, p)
    pf = [auroc(g.y.values, g.p.values) for _, g in oof.groupby("fold")]
    by_fold = {int(k): dict(auroc=auroc(g.y.values, g.p.values), sens=sens_spec(g.y.values, g.p.values)[0], spec=sens_spec(g.y.values, g.p.values)[1])
               for k, g in oof.groupby("fold")}
    o6 = oof[oof.fold != 7]; o7 = oof[oof.fold == 7]
    # dominance: drop each event, recompute pooled AUROC
    drop = {int(k): auroc(oof[oof.fold != k].y.values, oof[oof.fold != k].p.values) for k in sorted(oof.fold.unique())}
    # cluster bootstrap over events
    folds = sorted(oof.fold.unique()); g = {k: oof[oof.fold == k] for k in folds}; bs = []
    for _ in range(2000):
        pick = RNG.choice(folds, len(folds), replace=True)
        d = pd.concat([g[k] for k in pick]); a = auroc(d.y.values, d.p.values)
        if not np.isnan(a): bs.append(a)
    return dict(name=name, pooled_auroc=auroc(y, p), sens=se, spec=sp, per_fold_auroc_mean=float(np.nanmean(pf)), per_fold_auroc_min=float(np.nanmin(pf)),
                by_fold=by_fold, pooled_ex7_auroc=auroc(o6.y.values, o6.p.values), ex7_sens_spec=sens_spec(o6.y.values, o6.p.values),
                ev7_auroc=auroc(o7.y.values, o7.p.values), ev7_sens_spec=sens_spec(o7.y.values, o7.p.values),
                drop_one_event_auroc=drop, boot_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Wdf, ev = W_.build_index()
    masks = W_.fold_masks(Wdf, ev)
    s = S.build(W_.REC_PATH, "S00")
    assert s is not None
    c = Wdf.c.values
    if CACHE.exists():
        z = np.load(CACHE); F, p_mesa = z["F"], z["p_mesa"]; Fpad = z["Fpad"]
    else:
        card, eff = build_inputs(s, c)
        F, p_mesa, streams, thr = frozen_features(card, eff)
        cardp, effp = build_inputs(s, c, tail_pad_s=PAD_TEST_S)
        Fpad, _, _, _ = frozen_features(cardp, effp)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, F=F, Fpad=Fpad, p_mesa=p_mesa)
    H = handcrafted(s, c)
    hc = H.values.astype(np.float64); rms = H[["rms"]].values.astype(np.float64)
    print(f"frozen feature matrix {F.shape} (cardiac half = cols 0:128, effort half = 128:256); hand-crafted {hc.shape}")
    print("=" * 100)
    print("PILOT-SCALE RESULT: n = 7 events from ONE subject / ONE session. Not a validated accuracy.")
    print("COMPARISON CAVEAT: frozen features use a 150 s window (cardiac+effort); hand-crafted uses ONLY the 30 s effort centre epoch. NOT apples-to-apples.")
    print("=" * 100)
    prim_rows = np.where(Wdf.kind.isin(["pos", "rest"]))[0]
    bmask = Wdf.kind.values == "baseline"
    sets = {"frozen_CANet_256": F, "frozen_effort_half_128": F[:, 128:], "frozen_cardiac_half_128": F[:, :128],
            "handcrafted_8": hc, "rms_only": rms}
    res, oofs, bases = {}, {}, {}
    for name, X in sets.items():
        oof, bf, pdf = loeo(X, Wdf, masks, C_PRIMARY, base_X=X[bmask], pad_X=(Fpad if name == "frozen_CANet_256" else None))
        res[name] = summarize(oof, name); oofs[name] = oof; bases[name] = bf
        res[name]["baseline_fpr_mean"] = float(bf.base_fpr.mean()); res[name]["baseline_fpr_by_fold"] = bf.base_fpr.round(3).tolist()
        if pdf is not None:
            o_ = pdf.merge(oof[["row", "fold"]], on=["row", "fold"])
            ex7 = o_[o_.fold != 7]
            res[name]["padded_test_ex7_auroc"] = auroc(ex7.y.values, ex7.p.values)
            res[name]["padded_test_ex7_sens_spec"] = sens_spec(ex7.y.values, ex7.p.values)
            res[name]["unpadded_ex7_auroc"] = res[name]["pooled_ex7_auroc"]
            res[name]["unpadded_ex7_sens_spec"] = res[name]["ex7_sens_spec"] if False else sens_spec(oof[oof.fold != 7].y.values, oof[oof.fold != 7].p.values)
    # null control: shuffled training labels
    for name in ("frozen_CANet_256", "handcrafted_8"):
        nulls = [auroc(*(lambda o: (o.y.values, o.p.values))(loeo(sets[name], Wdf, masks, C_PRIMARY, shuffle_labels=True)[0])) for _ in range(20)]
        res[name]["null_shuffled_train_auroc_mean"] = float(np.nanmean(nulls)); res[name]["null_shuffled_train_auroc_sd"] = float(np.nanstd(nulls))
    # C sensitivity (reporting only; the primary C was fixed a priori)
    sens_C = {}
    for name in ("frozen_CANet_256", "handcrafted_8", "rms_only"):
        sens_C[name] = {str(C): summarize(loeo(sets[name], Wdf, masks, C)[0], name)["pooled_auroc"] for C in (0.001, 0.01, 0.1, 1.0)}
    res["C_sensitivity_pooled_auroc"] = sens_C
    # MESA original head probability on the same windows (reference, not trained here)
    prim = Wdf.loc[prim_rows]
    res["mesa_headline_prob_reference"] = dict(
        auroc_pos_vs_rest=auroc(prim.label.values, p_mesa[prim_rows]), mean_p_pos=float(p_mesa[prim_rows][prim.label.values == 1].mean()),
        mean_p_rest=float(p_mesa[prim_rows][prim.label.values == 0].mean()), baseline_mean_p=float(p_mesa[bmask].mean()))
    json.dump(res, open(OUT / "results.json", "w"), indent=1, default=float)
    for name in sets:
        oofs[name].to_csv(OUT / f"oof_{name}.csv", index=False)

    # ---------------------------------------------------------------- report
    def line(r):
        return (f"{r['name']:<26} pooled AUROC {r['pooled_auroc']:.3f} (boot95 {r['boot_ci95'][0]:.2f}-{r['boot_ci95'][1]:.2f}) | "
                f"sens {r['sens']:.2f} spec {r['spec']:.2f} @0.5 | per-fold AUROC mean {r['per_fold_auroc_mean']:.3f} min {r['per_fold_auroc_min']:.3f} | "
                f"baseline-period FPR {r['baseline_fpr_mean']:.2f}")
    print("\nPOOLED (all 7 events, out-of-fold), C = %.3g" % C_PRIMARY)
    for n in sets:
        print(" ", line(res[n]))
    print("\nPER-FOLD (held-out event): AUROC / sens / spec @0.5   [Event 7 has up to 27% zero-padded input]")
    hdr = "fold " + "".join(f"| {n[:22]:<24}" for n in sets)
    print(hdr)
    for k in sorted(masks):
        cells = []
        for n in sets:
            f = res[n]["by_fold"][k]
            cells.append(f"{f['auroc']:.2f}/{f['sens']:.2f}/{f['spec']:.2f}".ljust(24))
        print(f"{k}{'*' if k==7 else ' '}   " + "| ".join(cells))
    print("\nEVENT 7 — pooled with the other 6 (above) AND separately; Event 7 input is up to 27% zero-padded")
    for n in sets:
        r = res[n]
        print(f"  {n:<26} events 1-6 only: AUROC {r['pooled_ex7_auroc']:.3f} sens/spec {r['ex7_sens_spec'][0]:.2f}/{r['ex7_sens_spec'][1]:.2f} | "
              f"Event 7 alone: AUROC {r['ev7_auroc']:.3f} sens/spec {r['ev7_sens_spec'][0]:.2f}/{r['ev7_sens_spec'][1]:.2f}")
    r = res["frozen_CANet_256"]
    print(f"\nPADDING CONTROL (frozen features): events 1-6 test inputs with the last {PAD_TEST_S:.0f}s zero/neutral-padded:")
    print(f"  unpadded events 1-6: AUROC {r['unpadded_ex7_auroc']:.3f}, sens/spec {r['unpadded_ex7_sens_spec'][0]:.2f}/{r['unpadded_ex7_sens_spec'][1]:.2f}")
    print(f"  padded   events 1-6: AUROC {r['padded_test_ex7_auroc']:.3f}, sens/spec {r['padded_test_ex7_sens_spec'][0]:.2f}/{r['padded_test_ex7_sens_spec'][1]:.2f}")
    print("\nLEAKAGE / DOMINANCE CHECKS")
    for n in ("frozen_CANet_256", "handcrafted_8", "rms_only"):
        r = res[n]
        d = r["drop_one_event_auroc"]
        print(f"  {n}: pooled AUROC dropping each event -> " + ", ".join(f"{k}:{v:.2f}" for k, v in d.items()) + f" | >0.95 flag: {'YES' if r['pooled_auroc'] > 0.95 else 'no'}")
    for n in ("frozen_CANet_256", "handcrafted_8"):
        r = res[n]
        print(f"  {n}: shuffled-TRAIN-label null AUROC {r['null_shuffled_train_auroc_mean']:.3f} +/- {r['null_shuffled_train_auroc_sd']:.3f} (expect ~0.5)")
    print("\nC SENSITIVITY (pooled AUROC; primary C fixed a priori at 0.01):")
    for n, d in sens_C.items():
        print(f"  {n}: " + ", ".join(f"C={k}: {v:.3f}" for k, v in d.items()))
    m = res["mesa_headline_prob_reference"]
    print(f"\nREFERENCE (untouched MESA head, no training here): AUROC hold-vs-rest {m['auroc_pos_vs_rest']:.3f}; mean P(event) hold {m['mean_p_pos']:.2f} / rest {m['mean_p_rest']:.2f} / baseline {m['baseline_mean_p']:.2f}")
    print("\nREMINDER: n = 7 events, ONE subject. Pilot-scale. Frozen vs hand-crafted is NOT apples-to-apples (150 s vs 30 s context).")


if __name__ == "__main__":
    main()
