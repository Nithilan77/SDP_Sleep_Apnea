"""Phase 4 experiment: does real-signal augmentation and/or simulator data improve LOSO apnea classification?

Identical to src/ecg/expA_multiseed.py in task, minutes, architecture (CNN), optimiser, epochs, inner-validation protocol and thresholding. ONLY the training set differs:
  baseline      real fit records only                                  (the control, run through this same code path)
  augA / augB / aug_<component>   real fit records, augmented on the fly (GPU)
  synthR{r}     real fit + r x (#real fit minutes) simulator minutes   (simulator priors estimated from the FIT records only, per fold)
  synth_only    simulator minutes only (r=2)
  pretrain_ft   pretrain on r=2 simulator minutes, then fine-tune on real fit records (lr/3)
  combined_*    augmentation + simulator
Validation (early stopping, threshold) uses REAL validation records only; the held-out record is never used before prediction.
    python3 -m src.synth.exp_synth --config synthR2 --seeds 0 1 2
"""
from __future__ import annotations
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ecg import expA_multiseed as E
from src.ecg.cnn_apnea_final import restrict_to_phase2_minutes
from src.eval.splits import leave_one_subject_out
from src.synth import augment as AUG
from src.synth.physio import generate_training_minutes, priors_from_sequences, GEN_VERSION

OUT = Path("results/synth/exp"); DEV = E.DEV


def parse(cfg: str):
    """-> dict(aug=list, ratio=float, mode)"""
    d = dict(aug=[], ratio=0.0, mode="mix")
    parts = cfg.split("+")
    for p in parts:
        if p == "baseline": pass
        elif p in AUG.SETS: d["aug"] = AUG.SETS[p]
        elif p.startswith("synthR"): d["ratio"] = float(p[6:])
        elif p == "synth_only": d["mode"], d["ratio"] = "only", 2.0
        elif p == "pretrain_ft": d["mode"], d["ratio"] = "pretrain", 2.0
        else: raise SystemExit(f"unknown config part {p}")
    return d


def train(Xtr, ytr, Xva, yva, seed, comps, mu, sd, kind="cnn", epochs=E.EPOCHS, model=None, lr=E.LR, n_real=None, real_mask=None):
    """Xtr raw (N,2,60) on GPU. Augmentation applies to the real part only (real_mask) so simulator data are not double-perturbed."""
    torch.manual_seed(seed); m = (model or E.make(kind)).to(DEV); g = torch.Generator(device=DEV); g.manual_seed(seed)
    pw = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], dtype=torch.float32, device=DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr); lf = nn.BCEWithLogitsLoss(pos_weight=pw)
    norm = lambda x: (x - mu) / sd; rng = np.random.default_rng(seed); best, bs, eb = -1, None, 0
    yva_np = yva.cpu().numpy(); Xva_n = norm(Xva)
    for ep in range(epochs):
        m.train(); order = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), E.BATCH):
            i = torch.tensor(order[s:s + E.BATCH], device=DEV); x = Xtr[i]
            if comps:
                ra = real_mask[i] if real_mask is not None else torch.ones(len(i), dtype=torch.bool, device=DEV)
                if ra.any(): x = torch.where(ra[:, None, None], AUG.augment(x, comps, g), x)
            opt.zero_grad(); lf(m(norm(x)), ytr[i]).backward(); nn.utils.clip_grad_norm_(m.parameters(), E.CLIP); opt.step()
        a = roc_auc_score(yva_np, E.probs(m, Xva_n))
        if a > best: best, bs, eb = a, copy.deepcopy(m.state_dict()), ep + 1
    m.load_state_dict(bs); return m, best, eb


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--model", default="cnn"); ap.add_argument("--folds", type=int, default=35); a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    c = parse(a.config)
    ds = dict(np.load("results/phase3_cnn_v2/sequences.npz", allow_pickle=True)); mask = restrict_to_phase2_minutes(ds)
    seq, lab, rc, mn = ds["sequence"][mask], (ds["label"][mask] == "A").astype(np.float32), ds["record"][mask], ds["minute"][mask]
    recs = sorted(np.unique(rc)); rows, meta = [], []
    for seed in a.seeds:
        t0 = time.time()
        for k, (tr, te) in enumerate(leave_one_subject_out(recs), 1):
            if k > a.folds: break
            r = np.random.default_rng(1000 * seed + k); val = list(r.choice(tr, E.N_VAL, replace=False)); fit_r = [x for x in tr if x not in val]
            fi, vi, ei = np.isin(rc, fit_r), np.isin(rc, val), rc == te
            mu = seq[fi].mean(axis=(0, 2), keepdims=True); sd = seq[fi].std(axis=(0, 2), keepdims=True)
            Tn = lambda x: torch.tensor(x, dtype=torch.float32, device=DEV); mu_t, sd_t = Tn(mu), Tn(sd)
            Xr, yr = Tn(seq[fi]), Tn(lab[fi]); Xv, yv = Tn(seq[vi]), Tn(lab[vi])
            n_syn = int(round(c["ratio"] * fi.sum())); Xs = None
            if n_syn:
                pr = priors_from_sequences(seq[fi], lab[fi].astype(int), rc[fi])           # FIT records only
                s, ys, _, _ = generate_training_minutes(n_syn, pr, seed=10_000_000 + 1000 * seed + k); Xs, ys = Tn(s), Tn(ys.astype(np.float32))
            if c["mode"] == "only":
                Xtr, ytr, rm = Xs, ys, None
            elif c["mode"] == "pretrain":
                m0, _, _ = train(Xs, ys, Xv, yv, seed, [], mu_t, sd_t, kind=a.model)
                m, vauc, ep = train(Xr, yr, Xv, yv, seed, c["aug"], mu_t, sd_t, kind=a.model, model=m0, lr=E.LR / 3, real_mask=torch.ones(len(yr), dtype=torch.bool, device=DEV))
                Xtr = None
            elif Xs is not None:
                Xtr, ytr = torch.cat([Xr, Xs]), torch.cat([yr, ys]); rm = torch.cat([torch.ones(len(yr), dtype=torch.bool, device=DEV), torch.zeros(len(ys), dtype=torch.bool, device=DEV)])
            else:
                Xtr, ytr, rm = Xr, yr, torch.ones(len(yr), dtype=torch.bool, device=DEV)
            if Xtr is not None: m, vauc, ep = train(Xtr, ytr, Xv, yv, seed, c["aug"], mu_t, sd_t, kind=a.model, real_mask=rm)
            thr = E.bal_thr(lab[vi], E.probs(m, (Xv - mu_t) / sd_t)); p = E.probs(m, (Tn(seq[ei]) - mu_t) / sd_t)
            rows.append(pd.DataFrame(dict(config=a.config, seed=seed, record=rc[ei], minute=mn[ei], y_true=lab[ei].astype(int), y_prob=p, thr_val=thr, val_auc=vauc, best_epoch=ep)))
            meta.append(dict(config=a.config, seed=seed, fold=k, heldout=te, n_real_fit=int(fi.sum()), n_synth=n_syn, n_fit_records=len(fit_r), val_auc=float(vauc), epoch=int(ep)))
            print(f"[{a.config} s{seed}] fold {k}/35 {te} valAUC={vauc:.3f} ep={ep} nreal={fi.sum()} nsyn={n_syn} {time.time()-t0:.0f}s", flush=True)
        if a.folds == 35:
            pd.concat(rows, ignore_index=True).query("seed == @seed").to_csv(OUT / f"{a.model}__{a.config}__s{seed}_predictions.csv", index=False)
            pd.DataFrame(meta).query("seed == @seed").to_csv(OUT / f"{a.model}__{a.config}__s{seed}_folds.csv", index=False)
    if a.folds == 35:
        json.dump(dict(config=a.config, parsed=c, generator=GEN_VERSION, seeds=a.seeds, model=a.model, epochs=E.EPOCHS, batch=E.BATCH, lr=E.LR), open(OUT / f"{a.model}__{a.config}_config.json", "w"))


if __name__ == "__main__":
    main()
