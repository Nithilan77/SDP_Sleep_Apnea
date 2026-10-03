"""In-house hardware track for CANet: deployability + belt-vs-accelerometer domain gap + healthy negative control.

CHARACTERISATION / PLAUSIBILITY / DEPLOYABILITY ONLY. The in-house cohort is healthy 19-20 y/o with no apnea and
no labels, so NOTHING here is an apnea-detection accuracy. In-house recordings are never used for any fitting;
the MESA-trained final models come from src/mesa/train_final.py.

Outputs (no names / ids; nights are anonymised S01.. by file order and the mapping is not written):
    results/inhouse_canet/deployability/   per-night CANet output (event-epoch fraction, epochs/h-equivalent)
    results/inhouse_canet/domain_gap/      belt-vs-accelerometer effort-stream gap (CSV + plots)
    results/inhouse_canet/negative_control/ healthy event rate, vs MESA out-of-fold reference, stream isolation

    python -m src.inhouse.run_canet
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kurtosis

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.inhouse import streams as S  # noqa: E402
from src.mesa import canet as N  # noqa: E402
from src.mesa import cardiac_stream as C  # noqa: E402
from src.mesa import ecg_baseline as B  # noqa: E402
from src.mesa import respiratory_stream as R  # noqa: E402

log = logging.getLogger("inhouse_canet")
OUT = Path("results/inhouse_canet")
CACHE = Path("data/recordings/cache")  # gitignored
MODELS = Path("data/mesa/cache/final_models")
DEV = B.DEV
MIN_BEATS = B.CFG["min_beats"]
SRC_LABEL = {"mag": "accel magnitude (envelope.py)", "pca": "accel principal axis"}
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------- models
def load_models():
    out = {}
    ck = torch.load(MODELS / "ecg_only.pt", map_location=DEV, weights_only=False)
    m = B.CardiacCNN(dropout=B.CFG["dropout"]).to(DEV); m.load_state_dict(ck["state"]); out["ecg_only"] = (m.eval(), ck["thr"], ("cardiac",))
    for name in ("resp_only", "headline"):
        ck = torch.load(MODELS / f"{name}.pt", map_location=DEV, weights_only=False)
        m = N.CANet(ck["streams"], ck["fusion"]).to(DEV); m.load_state_dict(ck["state"]); out[name] = (m.eval(), ck["thr"], tuple(ck["streams"]))
    return out


@torch.no_grad()
def forward(model_t, cw, ew, bs=1024):
    model, _, streams = model_t
    out = []
    for i in range(0, len(cw if cw is not None else ew), bs):
        if streams == ("cardiac",):
            p = torch.sigmoid(model(torch.as_tensor(cw[i:i + bs], device=DEV)))
        else:
            xs = {}
            if "cardiac" in streams:
                xs["cardiac"] = torch.as_tensor(cw[i:i + bs], device=DEV)
            if "resp" in streams:
                xs["resp"] = torch.as_tensor(ew[i:i + bs], device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = torch.sigmoid(model(xs).float())
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def resp_embedding(model_t, ew, bs=1024):
    model = model_t[0]
    out = []
    for i in range(0, len(ew), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t = model.enc["resp"](torch.as_tensor(ew[i:i + bs], device=DEV))
        out.append(torch.cat([t.float().mean(1), t.float().amax(1)], 1).cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------- MESA reference (out-of-fold)
def mesa_reference() -> pd.DataFrame:
    """Per-subject fraction of sleep epochs flagged by each model in the MESA CV (out-of-fold, fold thresholds)."""
    audit = pd.read_csv(C.AUDIT_CSV, dtype={"mesaid": str}).set_index("mesaid")
    rows = []
    for name, path in (("ecg_only", B.PRED_CACHE / "preds.npz"), ("resp_only", N.CACHE / "resp_only" / "preds.npz"),
                       ("headline", N.CACHE / "headline" / "preds.npz")):
        z = np.load(path)
        df = pd.DataFrame(dict(subject=z["subject"], flag=(z["p"] >= z["thr"]), p=z["p"], y=z["y"]))
        g = df.groupby("subject").agg(frac_flagged=("flag", "mean"), mean_p=("p", "mean"))
        g["frac_flagged_on_true_normal"] = df[df.y == 0].groupby("subject").flag.mean()
        g["model"] = name; g["ahi_a0h4"] = audit.loc[g.index, "ahi_a0h4"].values
        rows.append(g.reset_index(drop=True))
    return pd.concat(rows)


# ----------------------------------------------------------------------------- effort epoch features
def epoch_features(X: np.ndarray) -> pd.DataFrame:
    """Per-30 s-epoch descriptors of the normalised 32 Hz trace (what the effort encoder actually receives)."""
    X = X.astype(np.float32)
    rms = np.sqrt((X ** 2).mean(1))
    F = np.abs(np.fft.rfft(X * np.hanning(X.shape[1]), axis=1)) ** 2
    f = np.fft.rfftfreq(X.shape[1], 1 / R.BELT_HZ)
    tot, br, pk = (f >= .03) & (f <= 1.5), (f >= .1) & (f <= .5), (f >= .1) & (f <= .6)
    Pt = F[:, tot].sum(1) + 1e-12
    pn = F[:, tot] / Pt[:, None]
    return pd.DataFrame(dict(
        rms=rms, kurtosis=kurtosis(X, axis=1), clip_frac=(np.abs(X) >= R.CLIP - 1e-3).mean(1),
        zcr_per_min=(np.diff(np.signbit(X), axis=1) != 0).sum(1) * 2.0,
        band_ratio=F[:, br].sum(1) / Pt, resp_bpm=f[pk][np.argmax(F[:, pk], 1)] * 60,
        peak_prominence=F[:, pk].max(1) / (np.median(F[:, tot], 1) + 1e-12),
        spec_entropy=-(pn * np.log(pn + 1e-12)).sum(1) / np.log(pn.shape[1])))


def night_psd(X: np.ndarray) -> np.ndarray:
    F = (np.abs(np.fft.rfft(X.astype(np.float32) * np.hanning(X.shape[1]), axis=1)) ** 2).mean(0)
    f = np.fft.rfftfreq(X.shape[1], 1 / R.BELT_HZ)
    m = (f >= .03) & (f <= 1.5)
    return F[m] / F[m].sum()


# ----------------------------------------------------------------------------- main
def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for d in ("deployability", "domain_gap", "negative_control"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    # ---------- streams for every usable night
    import pickle, time
    from joblib import Parallel, delayed
    t0 = time.time()
    cache = CACHE / "streams.pkl"
    if cache.exists():
        nights = pickle.load(open(cache, "rb"))
    else:
        recs = S.list_recordings()
        built = Parallel(n_jobs=5)(delayed(S.build)(p, f"S{k + 1:02d}") for k, p in enumerate(recs))
        nights = [b for b in built if b is not None]
        pickle.dump(nights, open(cache, "wb"))
    for s in nights:
        log.info("%s: %.1f h, %d epochs, median RR %.2f s", s.code, s.duration_s / 3600, s.n_epochs, s.median_rr_s)
    log.info("%d usable nights (>= %.0f s) [%.0fs]", len(nights), S.MIN_DURATION_S, time.time() - t0)

    models = load_models()
    mesa_ref = mesa_reference()
    thr = {k: v[1] for k, v in models.items()}

    # ---------- TASK 1 + 3: per-night output for every model / effort source / diagnostic
    rows, curves = [], {}
    for s in nights:
        tgt = np.flatnonzero(s.n_beats >= MIN_BEATS)
        cw = S.cardiac_windows(s, tgt)
        hours = len(tgt) * C.EPOCH_S / 3600
        variants = {("ecg_only", "-"): (models["ecg_only"], cw, None)}
        for src in S.EFFORT_SOURCES:
            ew = S.effort_windows(s.effort[src], tgt)
            variants[("resp_only", src)] = (models["resp_only"], None, ew)
            variants[("headline", src)] = (models["headline"], cw, ew)
        zero = np.zeros_like(S.effort_windows(s.effort["mag"], tgt[:1]).repeat(len(tgt), 0))
        noise = RNG.standard_normal(zero.shape).astype(np.float32)
        variants[("headline", "diag: effort=0")] = (models["headline"], cw, zero)
        variants[("headline", "diag: effort=white noise")] = (models["headline"], cw, noise)
        variants[("resp_only", "diag: effort=white noise")] = (models["resp_only"], None, noise)
        for (mname, src), (m, c, e) in variants.items():
            pr = forward(m, c, e)
            flag = pr >= thr[mname]
            rows.append(dict(night=s.code, model=mname, effort_source=src, hours_scored=round(hours, 2), n_epochs=len(tgt),
                             frac_flagged=float(flag.mean()), event_epochs_per_hour=float(flag.sum() / hours), mean_prob=float(pr.mean()),
                             median_prob=float(np.median(pr))))
            curves[f"{s.code}|{mname}|{src}"] = pr.astype(np.float16)
        del zero, noise
    res = pd.DataFrame(rows)
    np.savez_compressed(OUT / "deployability" / "per_epoch_probabilities.npz", **curves)

    main_rows = res[~res.effort_source.str.startswith("diag")]
    main_rows[(main_rows.model == "headline")].to_csv(OUT / "deployability" / "per_night_canet.csv", index=False)
    res.to_csv(OUT / "negative_control" / "per_night_all_models.csv", index=False)

    # MESA out-of-fold reference
    ref_rows = []
    for name in ("ecg_only", "resp_only", "headline"):
        g = mesa_ref[mesa_ref.model == name]
        for lab, sub in (("MESA all subjects (n=%d)" % len(g), g), ("MESA low-AHI a0h4<5 (n=%d)" % (g.ahi_a0h4 < 5).sum(), g[g.ahi_a0h4 < 5])):
            q = sub.frac_flagged.quantile([.1, .5, .9])
            ref_rows.append(dict(model=name, reference=lab, frac_flagged_p10=q.iloc[0], frac_flagged_median=q.iloc[1], frac_flagged_p90=q.iloc[2],
                                 mean_prob_median=sub.mean_p.median(), flagged_on_true_normal_median=sub.frac_flagged_on_true_normal.median()))
    ref = pd.DataFrame(ref_rows); ref.to_csv(OUT / "negative_control" / "mesa_out_of_fold_reference.csv", index=False)

    summ = []
    for (mname, src), g in res.groupby(["model", "effort_source"], sort=False):
        lo = ref[(ref.model == mname) & ref.reference.str.contains("low-AHI")].iloc[0]
        summ.append(dict(model=mname, effort_source=src, n_nights=len(g), frac_flagged_median=g.frac_flagged.median(),
                         frac_flagged_min=g.frac_flagged.min(), frac_flagged_max=g.frac_flagged.max(),
                         epochs_per_hour_median=g.event_epochs_per_hour.median(), mean_prob_median=g.mean_prob.median(),
                         mesa_lowAHI_frac_flagged_median=lo.frac_flagged_median, mesa_lowAHI_p90=lo.frac_flagged_p90,
                         nights_above_mesa_lowAHI_p90=int((g.frac_flagged > lo.frac_flagged_p90).sum())))
    pd.DataFrame(summ).to_csv(OUT / "negative_control" / "summary_by_model.csv", index=False)

    log.info('stage 1 (per-night CANet output) done [%.0fs]', time.time() - t0)
    # ---------- TASK 2: belt vs accelerometer domain gap
    post = np.load(B.PRED_CACHE / "preds.npz")
    mesa_feats, mesa_psd, mesa_X, mesa_grp = [], [], [], []
    for mid in np.unique(post["subject"]):
        ep = post["epoch"][post["subject"] == mid]
        ep = np.sort(RNG.choice(ep, min(150, len(ep)), replace=False))
        belts = R.load_subject(mid)["belts"].astype(np.float32)
        spe = int(C.EPOCH_S * R.BELT_HZ)
        for ch, nm in enumerate(("thor", "abdo")):
            X = np.stack([belts[ch, e * spe:(e + 1) * spe] for e in ep])
            f = epoch_features(X); f["domain"] = f"MESA {nm} belt"; f["night"] = mid; f["resp_source"] = "belt"
            mesa_feats.append(f)
            mesa_psd.append((f"MESA {nm} belt", night_psd(X)))
        mesa_X.append((mid, S.effort_windows(belts, ep)))
    inh_feats, inh_psd = [], []
    for s in nights:
        tgt = np.flatnonzero(s.n_beats >= MIN_BEATS)
        tgt = np.sort(RNG.choice(tgt, min(600, len(tgt)), replace=False))
        spe = int(C.EPOCH_S * R.BELT_HZ)
        for src in S.EFFORT_SOURCES:
            X = np.stack([s.effort[src][0, e * spe:(e + 1) * spe] for e in tgt])
            f = epoch_features(X); f["domain"] = f"in-house {SRC_LABEL[src]}"; f["night"] = s.code; f["resp_source"] = src
            inh_feats.append(f); inh_psd.append((f"in-house {SRC_LABEL[src]}", night_psd(X)))
    M, I = pd.concat(mesa_feats), pd.concat(inh_feats)
    allf = pd.concat([M, I]); feat_cols = ["rms", "kurtosis", "clip_frac", "zcr_per_min", "band_ratio", "resp_bpm", "peak_prominence", "spec_entropy"]

    gap = []
    mesa_pool = M[feat_cols]
    for dom, g in allf.groupby("domain", sort=False):
        night_med = g.groupby("night")[feat_cols].median()
        for c in feat_cols:
            mn = M.groupby("night")[c].median()  # one value per MESA subject-channel pooled
            gap.append(dict(domain=dom, feature=c, median=g[c].median(), iqr=g[c].quantile(.75) - g[c].quantile(.25),
                            mesa_belt_median=mesa_pool[c].median(), mesa_belt_iqr=mesa_pool[c].quantile(.75) - mesa_pool[c].quantile(.25),
                            shift_in_mesa_iqr=(g[c].median() - mesa_pool[c].median()) / (mesa_pool[c].quantile(.75) - mesa_pool[c].quantile(.25) + 1e-9),
                            night_median_percentile_in_mesa=float(np.mean([(mn <= v).mean() for v in night_med[c]]) * 100)))
    gap = pd.DataFrame(gap); gap.to_csv(OUT / "domain_gap" / "feature_gap_summary.csv", index=False)

    log.info('MESA/in-house epoch features done [%.0fs]', time.time() - t0)
    # domain classifiers (grouped CV by night / subject; AUROC 0.5 = indistinguishable, 1.0 = trivially separable)
    DC = OUT / 'domain_gap' / 'domain_classifier_auroc.csv'
    if not DC.exists():  # slow (~1 h on this box); cached
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        def dom_auc(Xa, Xb, ga, gb, clf):
            n = min(len(Xa), len(Xb), 8000)
            ia, ib = RNG.choice(len(Xa), n, replace=False), RNG.choice(len(Xb), n, replace=False)
            X = np.concatenate([Xa[ia], Xb[ib]]); y = np.r_[np.zeros(n), np.ones(n)]; g = np.r_[ga[ia], gb[ib]]
            p = np.zeros(len(y)); log.info('   domain classifier fit on %d rows', len(y))
            for tr, te in GroupKFold(5).split(X, y, g):
                p[te] = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            return float(roc_auc_score(y, p))

        hgb = lambda: HistGradientBoostingClassifier(max_iter=150)
        lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        Mb = M[M.domain.str.contains("belt")]
        rows = []
        mesa_subj = Mb.night.to_numpy()
        half = np.isin(mesa_subj, RNG.choice(np.unique(mesa_subj), len(np.unique(mesa_subj)) // 2, replace=False))
        rows.append(dict(comparison="CONTROL: MESA subjects half A vs half B (belt vs belt)", features="8 handcrafted epoch features",
                         auroc=dom_auc(Mb[half][feat_cols].to_numpy(), Mb[~half][feat_cols].to_numpy(), mesa_subj[half], mesa_subj[~half], hgb)))
        emb_ctrl_models = models["resp_only"]
        mesa_E = np.concatenate([resp_embedding(emb_ctrl_models, x) for _, x in mesa_X]); mesa_Eg = np.concatenate([[m] * len(x) for m, x in mesa_X])
        hv = np.isin(mesa_Eg, RNG.choice(np.unique(mesa_Eg), len(np.unique(mesa_Eg)) // 2, replace=False))
        rows.append(dict(comparison="CONTROL: MESA subjects half A vs half B (belt vs belt)", features="resp-encoder embedding (128-d)",
                         auroc=dom_auc(mesa_E[hv], mesa_E[~hv], mesa_Eg[hv], mesa_Eg[~hv], lr)))
        for src in S.EFFORT_SOURCES:
            Ii = I[I.resp_source == src]
            rows.append(dict(comparison=f"MESA belts vs in-house {SRC_LABEL[src]}", features="8 handcrafted epoch features",
                             auroc=dom_auc(Mb[feat_cols].to_numpy(), Ii[feat_cols].to_numpy(), Mb.night.to_numpy(), Ii.night.to_numpy(), hgb)))
            E_in, g_in = [], []
            for s in nights:
                ok_ep = np.flatnonzero(s.n_beats >= MIN_BEATS); tgt = np.sort(RNG.choice(ok_ep, min(400, len(ok_ep)), replace=False))
                E_in.append(resp_embedding(models["resp_only"], S.effort_windows(s.effort[src], tgt))); g_in += [s.code] * len(tgt)
            E_in = np.concatenate(E_in); g_in = np.array(g_in)
            rows.append(dict(comparison=f"MESA belts vs in-house {SRC_LABEL[src]}", features="resp-encoder embedding (128-d)",
                             auroc=dom_auc(mesa_E, E_in, mesa_Eg, g_in, lr)))
            mu, sd = mesa_E.mean(0), mesa_E.std(0) + 1e-6
            rows[-1]["embedding_mean_shift_in_mesa_sd"] = float(np.abs((E_in.mean(0) - mu) / sd).mean())
        pd.DataFrame(rows).to_csv(DC, index=False)
    log.info('domain classifiers done [%.0fs]', time.time() - t0)

    # ---------- plots
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    f = np.fft.rfftfreq(int(C.EPOCH_S * R.BELT_HZ), 1 / R.BELT_HZ); f = f[(f >= .03) & (f <= 1.5)]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for dom in dict.fromkeys(d for d, _ in mesa_psd + inh_psd):
        P = np.array([p for d, p in mesa_psd + inh_psd if d == dom])
        ax[0].semilogy(f, np.median(P, 0), label=dom); ax[0].fill_between(f, np.percentile(P, 25, 0), np.percentile(P, 75, 0), alpha=.15)
    ax[0].axvspan(.1, .5, color="grey", alpha=.1); ax[0].set(xlabel="Hz", ylabel="normalised power (median, IQR)", title="Effort-stream spectrum"); ax[0].legend(fontsize=7)
    for j, c in enumerate(("band_ratio", "resp_bpm")):
        for dom, g in allf.groupby("domain", sort=False):
            ax[j + 1].hist(g[c].clip(upper=40 if c == "resp_bpm" else 1), bins=50, density=True, histtype="step", label=dom)
        ax[j + 1].set(xlabel=c, title=f"per-epoch {c}")
    fig.tight_layout(); fig.savefig(OUT / "domain_gap" / "spectrum_and_breathing_features.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    for dom, g in allf.groupby("domain", sort=False):
        ax[0].hist(g.rms.clip(upper=3), bins=60, density=True, histtype="step", label=dom)
        ax[1].hist(g["kurtosis"].clip(upper=20), bins=60, density=True, histtype="step", label=dom)
    ax[0].set(xlabel="epoch RMS of normalised trace", title="amplitude envelope (after MESA-style normalisation)"); ax[0].legend(fontsize=7); ax[1].set(xlabel="kurtosis", title="waveform shape")
    fig.tight_layout(); fig.savefig(OUT / "domain_gap" / "amplitude_and_shape.png", dpi=120); plt.close(fig)

    h = res[(res.model == "headline") & (res.effort_source == "mag")]
    names = ["ecg_only|-", "resp_only|mag", "headline|mag", "headline|pca", "resp_only|pca"]
    fig, ax = plt.subplots(figsize=(11, 4.5)); w = .16
    for i, key in enumerate(names):
        m, src = key.split("|"); g = res[(res.model == m) & (res.effort_source == src)].set_index("night")
        ax.bar(np.arange(len(g)) + i * w, g.frac_flagged, w, label=key)
    lo = ref[(ref.model == "headline") & ref.reference.str.contains("low-AHI")].iloc[0]
    ax.axhline(lo.frac_flagged_median, color="k", ls="--", lw=1, label="MESA low-AHI median (headline, out-of-fold)")
    ax.set_xticks(np.arange(len(h)) + 2 * w); ax.set_xticklabels(h.night); ax.set(ylabel="fraction of epochs flagged", title="Healthy in-house nights: flagged fraction (NOT an apnea accuracy)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "negative_control" / "healthy_flagged_fraction.png", dpi=120); plt.close(fig)

    json.dump(dict(n_nights=len(nights), thresholds=thr, note="characterisation only; no apnea labels; healthy cohort"), open(OUT / "config.json", "w"), indent=2)
    log.info("done")


if __name__ == "__main__":
    main()
