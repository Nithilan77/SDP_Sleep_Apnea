"""Track B experiments: voluntary-breath-hold (hold vs non-hold) classification on in-house session 1, leave-one-EVENT-out.

Task: positive = `voluntary_breath_hold` window (10 s); negative = `baseline_reference` + `recovery`; uncertain intervals excluded (see annotate.py).
Folds: the 7 events; fold k tests on every window of event block k (its hold, its baseline, its recovery) and trains on the other 6 blocks.
THESE ARE NOT INDEPENDENT-SUBJECT RESULTS: one volunteer, one session, 7 events; adjacent blocks are >= 5 s apart.  Synthetic data (spec fitted WITHOUT session 1) and
SSL data (other in-house recordings, never session 1) are added to TRAINING only.  Thresholds are fixed at 0.5 (not tuned).

    python3 -m src.inhouse_bh.run_experiments [--seeds 0 1 2]
"""
from __future__ import annotations
import argparse, json, os, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
os.environ.setdefault("SYNTH_SPEC_DIR", "results/synth/spec_noS1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.inhouse_bh import common as Cm, augment as Aug, synthetic as Syn

OUT = Path("results/inhouse_bh"); DEV = "cpu"; torch.set_num_threads(2)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(3, 16, 9, 2, 4), nn.ReLU(), nn.Conv1d(16, 32, 7, 2, 3), nn.ReLU(), nn.Conv1d(32, 32, 5, 2, 2), nn.ReLU())
        self.head = nn.Linear(64, 1)
    def feats(self, x): h = self.enc(x); return torch.cat([h.mean(2), h.amax(2)], 1)
    def forward(self, x): return self.head(self.feats(x)).squeeze(-1)


class AE(nn.Module):
    """masked-reconstruction autoencoder sharing Net's encoder architecture (SSL pretraining)."""
    def __init__(self):
        super().__init__(); self.enc = Net().enc
        self.dec = nn.Sequential(nn.ConvTranspose1d(32, 32, 4, 2, 1), nn.ReLU(), nn.ConvTranspose1d(32, 16, 4, 2, 1), nn.ReLU(), nn.ConvTranspose1d(16, 3, 4, 2, 1))
    def forward(self, x): return self.dec(self.enc(x))


def fit_cnn(X, y, seed, epochs=40, aug=False, init=None, lr=1e-3, freeze=False):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); m = Net()
    if init is not None: m.enc.load_state_dict(init)
    if freeze:
        for p in m.enc.parameters(): p.requires_grad_(False)
    pw = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)], dtype=torch.float32); lf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        order = rng.permutation(len(X)); m.train()
        for s in range(0, len(X), 64):
            i = order[s:s + 64]; xb = X[i]
            if aug: xb = np.stack([Aug.augment_window(w, rng) for w in xb])
            opt.zero_grad(); lf(m(torch.tensor(xb)), torch.tensor(y[i], dtype=torch.float32)).backward(); opt.step()
    return m.eval()


def prob_cnn(m, X):
    with torch.no_grad(): return torch.sigmoid(m(torch.tensor(X))).numpy()


def expand(X, y, seed, k=6, comps=Aug.ALL):
    rng = np.random.default_rng(seed); Xa = [X]; ya = [y]
    for _ in range(k): Xa.append(np.stack([Aug.augment_window(w, rng, comps) for w in X])); ya.append(y)
    return np.concatenate(Xa), np.concatenate(ya)


def F(X): return np.stack([Cm.features(w) for w in X])


def sk_models(seed):
    return {"LR": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
            "HGB": HGB(max_iter=120, max_depth=3, learning_rate=0.08, class_weight="balanced", random_state=seed)}


def load_ssl(n_per=1200, seed=0):
    """unlabelled windows from in-house recordings EXCLUDING session 1 (cached traces)."""
    cache = Path("data/recordings/cache/bh_ssl_windows.npy")
    if cache.exists(): return np.load(cache)
    H = Path.home() / "Desktop"; files = sorted(Path("data/recordings/inhouse_ecg").glob("*.txt")) + [H / "Mithin-BreathHold" / f"user_MithinBreathHold_{k}_000.csv" for k in ("ECG", "EMG", "EOG")] + \
            [H / "Mithin/user_Mithin10_ECG_000.csv", H / "Mithin/user_Mithin10_EOG_000.csv"]
    rng = np.random.default_rng(seed); out = []
    for f in files:
        try:
            if f.stat().st_size == 0 or f.stat().st_size < 2e6: continue
            x, d, fs = Cm.trace32(f); idx = rng.integers(0, len(x) - Cm.W, n_per)
            out.append(Cm.window_array(x, idx, Cm.local_scales(x, idx)))
            print("ssl source", f.name[:30], d, flush=True)
        except Exception as e: print("skip", f.name, e)
    A = np.concatenate(out).astype(np.float32); A = np.clip(A, -20, 20); cache.parent.mkdir(parents=True, exist_ok=True); np.save(cache, A); return A


def pretrain_ae(A, seed, epochs=8):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); m = AE(); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for ep in range(epochs):
        order = rng.permutation(len(A)); tot = 0
        for s in range(0, len(A), 128):
            x = torch.tensor(A[order[s:s + 128]]); mask = (torch.rand(x.shape[0], 1, x.shape[2] // 16) > 0.25).float().repeat_interleave(16, 2)
            rec = m(x * mask); loss = ((rec - x) ** 2 * (1 - mask) ).sum() / ((1 - mask).sum() * 3 + 1e-9) + 0.1 * ((rec - x) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
    return m.enc.state_dict()


CONFIGS = ["real", "aug", "synth", "aug+synth", "pretrain_ft", "ssl_probe", "ssl_ft"]


def run_fold(cfg, Xtr, ytr, Xte, seed, syn, ssl_state, models=("LR", "HGB", "CNN")):
    preds = {}
    a = "aug" in cfg; use_syn = "synth" in cfg or cfg == "pretrain_ft"
    Xr, yr = (expand(Xtr, ytr, seed) if a and cfg != "pretrain_ft" else (Xtr, ytr))
    Xt, yt = (np.concatenate([Xr, syn[0]]), np.concatenate([yr, syn[1]])) if "synth" in cfg else (Xr, yr)
    if cfg in ("real", "aug", "synth", "aug+synth"):
        Ft, Fe = F(Xt), F(Xte)
        for n, m in sk_models(seed).items():
            if n in models: m.fit(Ft, yt); preds[n] = m.predict_proba(Fe)[:, 1]
        if "CNN" in models: preds["CNN"] = prob_cnn(fit_cnn(Xt, yt, seed, aug=False), Xte) if not a else prob_cnn(fit_cnn(np.concatenate([Xtr, syn[0]]) if "synth" in cfg else Xtr, np.concatenate([ytr, syn[1]]) if "synth" in cfg else ytr, seed, aug=True), Xte)
    elif cfg == "pretrain_ft":
        m0 = fit_cnn(syn[0], syn[1], seed, epochs=15); m = fit_cnn(Xtr, ytr, seed, epochs=30, aug=True, init=m0.enc.state_dict(), lr=3e-4); preds["CNN"] = prob_cnn(m, Xte)
    elif cfg in ("ssl_probe", "ssl_ft"):
        m = fit_cnn(Xtr, ytr, seed, epochs=30 if cfg == "ssl_probe" else 40, aug=True, init=ssl_state, freeze=(cfg == "ssl_probe")); preds["CNN"] = prob_cnn(m, Xte)
    return preds


def summarize(P: pd.DataFrame):
    rows = []
    for (cfg, mod), g in P.groupby(["config", "model"]):
        pooled = g.groupby("seed").apply(lambda d: pd.Series(dict(auroc=roc_auc_score(d.y, d.p), ap=average_precision_score(d.y, d.p),
                                                                    sens=((d.p > .5) & (d.y == 1)).sum() / (d.y == 1).sum(), spec=((d.p <= .5) & (d.y == 0)).sum() / (d.y == 0).sum())))
        # event level (mean over seeds of per-event detection = fraction of hold windows flagged)
        ev = g[g.y == 1].groupby(["seed", "event_no"]).apply(lambda d: (d.p > .5).mean()).groupby("event_no").mean()
        fa = g[g.y == 0].groupby("seed").apply(lambda d: (d.p > .5).mean())
        rows.append(dict(config=cfg, model=mod, **{f"{k}_mean": v for k, v in pooled.mean().items()}, **{f"{k}_sd": v for k, v in pooled.std().items()},
                         events_detected_ge50pct=int((ev >= 0.5).sum()), per_event_flag_frac=";".join(f"{v:.2f}" for v in ev.values), false_flag_frac_nonhold=float(fa.mean())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--n-synth-rec", type=int, default=100); ap.add_argument("--configs", nargs="+", default=CONFIGS)
    a = ap.parse_args(); t0 = time.time()
    x, dur, fs = Cm.trace32(Cm.PATHS["S1_ECGTILE"]); w, sc = Cm.labelled_windows("S1_ECGTILE", x); X = Cm.window_array(x, w.i0.values, w.scale.values); y = (w.label == "voluntary_breath_hold").astype(int).values; ev = w.event_no.values
    print("windows", X.shape, "holds", y.sum(), flush=True)
    syn_all = Syn.synthetic_windows(a.n_synth_rec, 4242, return_stats=True); syn = syn_all[:2]; syn_stats = syn_all[3]
    print("synthetic windows", syn[0].shape, "hold frac", syn[1].mean(), syn_stats, flush=True)
    # unlabelled-time diagnostic: windows (stride 4 s) not overlapping ANY annotated interval; scored by every fold's model; reported as a flag rate, not an error rate
    iu = np.arange(0, len(x) - Cm.W, 4 * Cm.HZ); ann_ = pd.read_csv(Cm.ANN); ann_ = ann_[(ann_.recording_id == "S1_ECGTILE") & (ann_.time_base == "recording")]
    free = np.array([not ((ann_.start_s < i / Cm.HZ + Cm.WIN_S) & (ann_.end_s > i / Cm.HZ)).any() for i in iu]); iu = iu[free]; Xu = Cm.window_array(x, iu, Cm.local_scales(x, iu)); unl = []
    ssl = None
    if any(c.startswith("ssl") for c in a.configs): ssl = load_ssl(); print("ssl windows", ssl.shape, flush=True)
    rows = []
    for seed in a.seeds:
        ssl_state = pretrain_ae(ssl, seed) if ssl is not None else None
        for cfg in a.configs:
            for k in range(1, 8):
                tr, te = ev != k, ev == k
                both = run_fold(cfg, X[tr], y[tr], np.concatenate([X[te], Xu]), seed, syn, ssl_state); nte = int(te.sum())
                if len(both) > 1: both["ENS(mean)"] = np.mean([both[m] for m in both], 0)
                preds = {m: v[:nte] for m, v in both.items()}
                for m, v in both.items(): unl.append(dict(config=cfg, model=m, seed=seed, fold=k, n=len(Xu), flag_frac=float((v[nte:] >= .5).mean())))
                for mod, p in preds.items():
                    rows.append(pd.DataFrame(dict(config=cfg, model=mod, seed=seed, event_no=k, y=y[te], p=p, label=w.label.values[te], t0=w.t0.values[te], confidence=w.confidence.values[te])))
            print(f"seed {seed} {cfg} done {time.time()-t0:.0f}s", flush=True)
    P = pd.concat(rows, ignore_index=True); OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(unl).to_csv(OUT / "loeo_unlabelled_flag_rates.csv", index=False)
    P.to_csv(OUT / "loeo_predictions.csv.gz", index=False); S = summarize(P); S.to_csv(OUT / "loeo_summary.csv", index=False)
    json.dump(dict(n_windows=int(len(X)), n_hold=int(y.sum()), n_nonhold=int((1 - y).sum()), n_synthetic_windows=int(len(syn[0])), synthetic_qc=syn_stats, n_unlabelled_windows=int(len(Xu)), seeds=a.seeds, win_s=Cm.WIN_S, stride_s=Cm.STRIDE_S, runtime_s=time.time() - t0), open(OUT / "loeo_meta.json", "w"))
    print(S.round(3).to_string())


if __name__ == "__main__":
    main()
