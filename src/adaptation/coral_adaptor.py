"""CORAL adaptor — corrected: applies alignment at token level BEFORE cross-attention.

Step 2 of the adaptation track.

    python -m src.adaptation.coral_adaptor
Outputs -> results/adaptation/coral/
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.mesa import canet as N
from src.mesa import cardiac_stream as C
from src.mesa import respiratory_stream as R
from src.mesa import ecg_baseline as B
from src.inhouse import streams as S

log  = logging.getLogger("coral")
OUT  = Path("results/adaptation/coral")
MODELS      = Path("data/mesa/cache/final_models")
STREAM_CACHE= Path("data/recordings/cache/streams.pkl")
EMB         = Path("results/adaptation/embed_gap/embeddings.npz")
DEV  = B.DEV
RNG  = np.random.default_rng(42)

N_MESA_SUBJ     = 60
N_MESA_PER_SUBJ = 30
N_INH_PER_NIGHT = 200


# ── CORAL maths ──────────────────────────────────────────────────────────────

def coral_transform(X_src: np.ndarray, X_tgt: np.ndarray,
                    reg: float = 1.0) -> np.ndarray:
    """W (D,D): maps X_tgt so its covariance matches X_src. No labels needed."""
    def _cov(X):
        Xc = X - X.mean(0)
        return Xc.T @ Xc / (len(X) - 1) + reg * np.eye(X.shape[1])
    def _sqrt(M, inv=False):
        vals, vecs = np.linalg.eigh(M)
        vals = np.maximum(vals, 1e-10)
        s = (1.0 / np.sqrt(vals)) if inv else np.sqrt(vals)
        return (vecs * s) @ vecs.T
    return (_sqrt(_cov(X_tgt), inv=True) @ _sqrt(_cov(X_src))).astype(np.float32)


def dom_auroc(Xa, Xb, ga, gb) -> float:
    n = min(len(Xa), len(Xb), 4000)
    ia = RNG.choice(len(Xa), n, replace=False)
    ib = RNG.choice(len(Xb), n, replace=False)
    X  = np.concatenate([Xa[ia], Xb[ib]])
    y  = np.r_[np.zeros(n), np.ones(n)]
    g  = np.r_[ga[ia], gb[ib]]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        p[te] = clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, p))


# ── embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def get_tokens(enc: torch.nn.Module,
               windows: np.ndarray, bs: int = 256) -> np.ndarray:
    """Resp encoder -> (N*75, 64) flattened tokens."""
    enc.eval()
    out = []
    for i in range(0, len(windows), bs):
        x = torch.as_tensor(windows[i:i + bs], device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t = enc(x)                          # (B, 75, 64)
        out.append(t.float().cpu().numpy().reshape(-1, 64))
    return np.concatenate(out)


@torch.no_grad()
def get_gap_gmp(enc: torch.nn.Module,
                windows: np.ndarray, bs: int = 256) -> np.ndarray:
    """Resp encoder -> (N, 128) GAP+GMP embeddings."""
    enc.eval()
    out = []
    for i in range(0, len(windows), bs):
        x = torch.as_tensor(windows[i:i + bs], device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t = enc(x)
        z = torch.cat([t.float().mean(1), t.float().amax(1)], 1)
        out.append(z.cpu().numpy())
    return np.concatenate(out)


# ── corrected inference: CORAL at TOKEN level, then full cross-attn + head ──

@torch.no_grad()
def forward_with_token_coral(model: N.CANet,
                              cw: np.ndarray, ew: np.ndarray,
                              W: np.ndarray,
                              src_mean: np.ndarray,
                              tgt_mean: np.ndarray,
                              bs: int = 512) -> np.ndarray:
    """
    CANet forward with CORAL injected at the effort token level.

    Normal CANet:
        enc["cardiac"](xc) -> tc  (B,75,64)
        enc["resp"](xe)    -> te  (B,75,64)
        cross({cardiac:tc, resp:te}) -> attended tokens
        GAP+GMP -> head -> logit

    Here: CORAL(te) before cross-attention; everything else unchanged.
    """
    model.eval()
    W_t = torch.as_tensor(W, device=DEV)         # (64,64)
    sm  = torch.as_tensor(src_mean, device=DEV)  # (64,)
    tm  = torch.as_tensor(tgt_mean, device=DEV)  # (64,)
    out = []
    for i in range(0, len(cw), bs):
        xc = torch.as_tensor(cw[i:i + bs], device=DEV)
        xe = torch.as_tensor(ew[i:i + bs], device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tc = model.enc["cardiac"](xc).float()   # (B,75,64)
            te = model.enc["resp"](xe).float()       # (B,75,64)

        # CORAL per token: (B,75,64) -> flatten -> align -> reshape
        B_sz, T, D = te.shape
        te_flat    = te.reshape(B_sz * T, D)
        te_aligned = (te_flat - tm) @ W_t + sm     # (B*75, 64)
        te_adapted = te_aligned.reshape(B_sz, T, D)

        # cross-attention with adapted effort tokens (cardiac unchanged)
        t_att = model.cross({"cardiac": tc, "resp": te_adapted})

        # GAP+GMP then head — identical to original CANet.forward
        z = torch.cat([
            torch.cat([t_att["cardiac"].mean(1), t_att["cardiac"].amax(1)], 1),
            torch.cat([t_att["resp"].mean(1),    t_att["resp"].amax(1)],    1),
        ], 1)                                       # (B, 256)
        p = torch.sigmoid(model.head(z).squeeze(-1))
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    # load model
    log.info("Loading headline model ...")
    ck    = torch.load(MODELS / "headline.pt", map_location=DEV, weights_only=False)
    model = N.CANet(ck["streams"], ck["fusion"]).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    enc   = model.enc["resp"]
    thr   = ck["thr"]

    # ── collect MESA belt TOKENS (64-dim) ─────────────────────────────────────
    log.info("Extracting MESA belt tokens (64-dim) ...")
    post     = np.load(B.PRED_CACHE / "preds.npz")
    all_subj = np.unique(post["subject"])
    sel_subj = RNG.choice(all_subj, min(N_MESA_SUBJ, len(all_subj)), replace=False)

    mesa_tokens, mesa_gaps, mesa_g = [], [], []
    for k, mid in enumerate(sel_subj):
        ep    = post["epoch"][post["subject"] == mid]
        ep    = np.sort(RNG.choice(ep, min(N_MESA_PER_SUBJ, len(ep)), replace=False))
        belts = R.load_subject(mid)["belts"].astype(np.float32)
        wins  = S.effort_windows(belts, ep)
        mesa_tokens.append(get_tokens(enc, wins))           # (N*75, 64)
        mesa_gaps.append(get_gap_gmp(enc, wins))            # (N, 128)
        mesa_g.append(np.full(len(wins), k))
    mesa_tokens = np.concatenate(mesa_tokens)               # (M_tok, 64)
    mesa_gaps   = np.concatenate(mesa_gaps)                 # (M_ep,  128)
    mesa_g_ep   = np.concatenate(mesa_g)
    log.info("  MESA: %d tokens, %d embeddings", len(mesa_tokens), len(mesa_gaps))

    # ── collect in-house accel TOKENS ─────────────────────────────────────────
    log.info("Extracting in-house accelerometer tokens (64-dim) ...")
    nights = pickle.load(open(STREAM_CACHE, "rb"))

    inh_tokens, inh_gaps, inh_g = [], [], []
    for k, s in enumerate(nights):
        tgt  = np.flatnonzero(s.n_beats >= B.CFG["min_beats"])
        tgt  = np.sort(RNG.choice(tgt, min(N_INH_PER_NIGHT, len(tgt)), replace=False))
        wins = S.effort_windows(s.effort["mag"], tgt)
        inh_tokens.append(get_tokens(enc, wins))
        inh_gaps.append(get_gap_gmp(enc, wins))
        inh_g.append(np.full(len(wins), k))
    inh_tokens = np.concatenate(inh_tokens)
    inh_gaps   = np.concatenate(inh_gaps)
    inh_g_ep   = np.concatenate(inh_g)
    log.info("  In-house: %d tokens, %d embeddings", len(inh_tokens), len(inh_gaps))

    # ── CORAL in token space (64-dim) ─────────────────────────────────────────
    log.info("\nSweeping CORAL reg in 64-dim token space ...")
    regs = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    # subsample tokens for the domain classifier (tokens from same epoch are correlated)
    step = 75  # one token per epoch to remove within-epoch correlation
    mt_sub = mesa_tokens[::step]; it_sub = inh_tokens[::step]
    g_mt   = np.arange(len(mt_sub)); g_it = np.arange(len(it_sub))

    auc_before = dom_auroc(mt_sub, it_sub, g_mt, g_it)
    log.info("  BEFORE CORAL (token space): %.4f", auc_before)

    sweep = []
    for reg in regs:
        W   = coral_transform(mesa_tokens, inh_tokens, reg)
        adapted = (it_sub - inh_tokens.mean(0)) @ W + mesa_tokens.mean(0)
        auc = dom_auroc(mt_sub, adapted, g_mt, g_it)
        log.info("  reg=%.2f -> AUROC %.4f", reg, auc)
        sweep.append((reg, auc))

    best_reg = min(sweep, key=lambda x: abs(x[1] - 0.5))[0]
    best_auc = dict(sweep)[best_reg]
    log.info("Best reg=%.2f (AUROC closest to 0.5) -> %.4f", best_reg, best_auc)

    W_best       = coral_transform(mesa_tokens, inh_tokens, best_reg)
    src_mean_tok = mesa_tokens.mean(0).astype(np.float32)
    tgt_mean_tok = inh_tokens.mean(0).astype(np.float32)

    np.savez_compressed(OUT / "coral_params_token.npz",
                        W=W_best,
                        src_mean=src_mean_tok,
                        tgt_mean=tgt_mean_tok,
                        best_reg=np.array(best_reg))
    log.info("Saved coral_params_token.npz")

    # ── reg sweep plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx([r for r, _ in sweep], [a for _, a in sweep], "o-", color="#2196F3")
    ax.axhline(auc_before, color="red",  ls="--", label=f"before CORAL ({auc_before:.3f})")
    ax.axhline(0.5,         color="grey", ls=":",  label="ideal (0.5 = indistinguishable)")
    ax.set(xlabel="regularisation", ylabel="domain classifier AUROC (token space)",
           title="CORAL reg sweep — closer to 0.5 is better")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "reg_sweep_token.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── corrected inference: flag rate after CORAL ─────────────────────────────
    log.info("\nRunning corrected inference (CORAL at token level) ...")
    import pandas as pd
    rows = []
    for s in nights:
        tgt = np.flatnonzero(s.n_beats >= B.CFG["min_beats"])
        if len(tgt) == 0:
            continue
        cw = S.cardiac_windows(s, tgt)
        ew = S.effort_windows(s.effort["mag"], tgt)
        p  = forward_with_token_coral(model, cw, ew, W_best,
                                      src_mean_tok, tgt_mean_tok)
        flag = (p >= thr).mean()
        rows.append(dict(night=s.code, n_epochs=len(tgt),
                         frac_flagged=float(flag),
                         mean_prob=float(p.mean())))
        log.info("  %s: %.1f%% flagged", s.code, flag * 100)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "flag_rate_token_coral.csv", index=False)

    log.info("\n=== RESULTS ===")
    log.info("Domain AUROC before CORAL : %.4f", auc_before)
    log.info("Domain AUROC after  CORAL : %.4f  (ideal: 0.50)", best_auc)
    log.info("Flag rate BEFORE CORAL    : ~93%% (from previous run)")
    log.info("Flag rate AFTER  CORAL    : %.1f%% median  (MESA low-AHI reference: ~28%%)",
             res.frac_flagged.median() * 100)

    (OUT / "summary.json").write_text(json.dumps(dict(
        token_space_auc_before=auc_before,
        token_space_auc_after=best_auc,
        best_reg=best_reg,
        flag_rate_median=float(res.frac_flagged.median()),
        mesa_lowAHI_reference=0.28,
    ), indent=2))
    log.info("All outputs in %s", OUT)


if __name__ == "__main__":
    main()
