"""Effort-encoder embedding visualisation: MESA belts vs in-house accelerometer.

Step 1 of the sensor-domain adaptation track — understand the gap topology
before building any adaptor. Saves embeddings + PCA/t-SNE plots.

    python -m src.adaptation.embed_gap
Outputs -> results/adaptation/embed_gap/
"""
from __future__ import annotations

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
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.mesa import canet as N
from src.mesa import cardiac_stream as C
from src.mesa import respiratory_stream as R
from src.mesa import ecg_baseline as B
from src.inhouse import streams as S

log = logging.getLogger("embed_gap")
OUT   = Path("results/adaptation/embed_gap")
MODELS = Path("data/mesa/cache/final_models")
STREAM_CACHE = Path("data/recordings/cache/streams.pkl")
DEV   = B.DEV
RNG   = np.random.default_rng(42)

# how much data to sample
N_MESA_SUBJ        = 60   # MESA subjects (random subset, enough to represent the belt distribution)
N_MESA_PER_SUBJ    = 30   # epochs per subject
N_INH_PER_NIGHT    = 200  # epochs per in-house night
MAX_TSNE           = 5000 # cap for t-SNE runtime


@torch.no_grad()
def get_embeddings(enc: torch.nn.Module, windows: np.ndarray, bs: int = 512) -> np.ndarray:
    """GAP+GMP from the resp encoder -> (N, 128) float32."""
    enc.eval()
    out = []
    for i in range(0, len(windows), bs):
        x = torch.as_tensor(windows[i:i + bs], device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t = enc(x)                                     # (B, 75, 64)
        z = torch.cat([t.float().mean(1), t.float().amax(1)], 1)   # (B, 128)
        out.append(z.cpu().numpy())
    return np.concatenate(out)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    # ── load effort encoder (from headline model: resp encoder trained with cross-attention)
    log.info("Loading effort encoder from headline.pt ...")
    ck = torch.load(MODELS / "headline.pt", map_location=DEV, weights_only=False)
    model = N.CANet(ck["streams"], ck["fusion"]).to(DEV)
    model.load_state_dict(ck["state"])
    enc = model.enc["resp"].eval()
    log.info("Effort encoder loaded (%d params)", sum(p.numel() for p in enc.parameters()))

    # ── MESA belt embeddings
    log.info("Extracting MESA belt embeddings (%d subjects x %d epochs) ...", N_MESA_SUBJ, N_MESA_PER_SUBJ)
    post     = np.load(B.PRED_CACHE / "preds.npz")
    all_subj = np.unique(post["subject"])
    sel_subj = RNG.choice(all_subj, min(N_MESA_SUBJ, len(all_subj)), replace=False)

    mesa_emb = []
    for mid in sel_subj:
        ep    = post["epoch"][post["subject"] == mid]
        ep    = np.sort(RNG.choice(ep, min(N_MESA_PER_SUBJ, len(ep)), replace=False))
        belts = R.load_subject(mid)["belts"].astype(np.float32)   # (2, T) already preprocessed
        wins  = S.effort_windows(belts, ep)                        # (N, 2, 4800)
        mesa_emb.append(get_embeddings(enc, wins))
    mesa_emb = np.concatenate(mesa_emb)
    log.info("  MESA belt: %d embeddings", len(mesa_emb))

    # ── in-house accelerometer embeddings
    log.info("Extracting in-house accelerometer embeddings ...")
    if not STREAM_CACHE.exists():
        log.error("Stream cache missing at %s — run src/inhouse/run_canet.py first", STREAM_CACHE)
        return
    nights = pickle.load(open(STREAM_CACHE, "rb"))

    inh_emb = {src: [] for src in S.EFFORT_SOURCES}
    for s in nights:
        tgt = np.flatnonzero(s.n_beats >= B.CFG["min_beats"])
        tgt = np.sort(RNG.choice(tgt, min(N_INH_PER_NIGHT, len(tgt)), replace=False))
        for src in S.EFFORT_SOURCES:
            wins = S.effort_windows(s.effort[src], tgt)
            inh_emb[src].append(get_embeddings(enc, wins))
    for src in S.EFFORT_SOURCES:
        inh_emb[src] = np.concatenate(inh_emb[src])
        log.info("  in-house %s: %d embeddings", src, len(inh_emb[src]))

    # ── save raw embeddings
    np.savez_compressed(OUT / "embeddings.npz",
                        mesa=mesa_emb,
                        inh_mag=inh_emb["mag"],
                        inh_pca=inh_emb["pca"])

    # ── PCA
    log.info("Running PCA (n_components=50) ...")
    n_mesa = len(mesa_emb)
    n_mag  = len(inh_emb["mag"])
    all_emb = np.concatenate([mesa_emb, inh_emb["mag"], inh_emb["pca"]])
    pca     = PCA(n_components=50, random_state=42)
    all_pca = pca.fit_transform(all_emb)
    ev      = pca.explained_variance_ratio_
    log.info("  Top-10 explained variance: %s", np.round(ev[:10] * 100, 1).tolist())

    pca_mesa = all_pca[:n_mesa]
    pca_mag  = all_pca[n_mesa:n_mesa + n_mag]
    pca_pca_ = all_pca[n_mesa + n_mag:]

    # per-dimension mean shift (in MESA SDs) — tells us which PCA dims drive the gap
    dim_shift_mag = np.abs(pca_mag.mean(0) - pca_mesa.mean(0)) / (pca_mesa.std(0) + 1e-6)
    dim_shift_pca = np.abs(pca_pca_.mean(0) - pca_mesa.mean(0)) / (pca_mesa.std(0) + 1e-6)
    top10 = np.argsort(dim_shift_mag)[::-1][:10]

    log.info("  Top-10 shifted PCA dims (mag): %s", [f"PC{d+1}:{dim_shift_mag[d]:.2f}SD" for d in top10])
    log.info("  Mean shift across all 50 dims — mag: %.3f SD  pca: %.3f SD",
             dim_shift_mag.mean(), dim_shift_pca.mean())

    # ── t-SNE on PCA-50 (capped for speed)
    log.info("Running t-SNE (this takes a few minutes) ...")
    labels = (["MESA belt"] * n_mesa +
              [f"in-house mag ({len(inh_emb['mag'])} epochs)"] * n_mag +
              [f"in-house pca ({len(inh_emb['pca'])} epochs)"] * len(inh_emb["pca"]))
    if len(all_pca) > MAX_TSNE:
        idx = RNG.choice(len(all_pca), MAX_TSNE, replace=False)
        tsne_in, tsne_labels = all_pca[idx], [labels[i] for i in idx]
    else:
        tsne_in, tsne_labels = all_pca, labels
    tsne_2d = TSNE(n_components=2, perplexity=40, random_state=42,
                   max_iter=1000, verbose=1).fit_transform(tsne_in)

    # ── plots
    COLORS = {"MESA belt": "#2196F3",
              f"in-house mag ({len(inh_emb['mag'])} epochs)": "#F44336",
              f"in-house pca ({len(inh_emb['pca'])} epochs)": "#FF9800"}
    unique_labels = list(dict.fromkeys(tsne_labels))

    # plot 1: PCA PC1/PC2 + t-SNE side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for pts, label, c in ((pca_mesa, "MESA belt", "#2196F3"),
                           (pca_mag,  "in-house mag", "#F44336"),
                           (pca_pca_, "in-house pca", "#FF9800")):
        axes[0].scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.35, c=c, label=label)
    axes[0].set(xlabel=f"PC1 ({ev[0]*100:.1f}%)", ylabel=f"PC2 ({ev[1]*100:.1f}%)",
                title="Effort encoder — PCA of embeddings")
    axes[0].legend(markerscale=3, fontsize=8)

    for lab in unique_labels:
        pts = tsne_2d[[i for i, l in enumerate(tsne_labels) if l == lab]]
        c   = COLORS.get(lab, "#9E9E9E")
        axes[1].scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.35, c=c, label=lab)
    axes[1].set(xlabel="t-SNE 1", ylabel="t-SNE 2",
                title="Effort encoder — t-SNE of embeddings")
    axes[1].legend(markerscale=3, fontsize=8)

    fig.suptitle("Belt-trained effort encoder: MESA training domain vs in-house sensor domain\n"
                 "Separated clusters = the gap we need to close", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "embedding_space.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved embedding_space.png")

    # plot 2: per-PCA-dim gap bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(10)
    ax.bar(x - 0.2, dim_shift_mag[top10], 0.4, label="mag", color="#F44336", alpha=0.8)
    ax.bar(x + 0.2, dim_shift_pca[top10], 0.4, label="pca", color="#FF9800", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{d+1}" for d in top10])
    ax.set(ylabel="mean shift (MESA SDs)",
           title="Which PCA dimensions drive the belt-vs-accelerometer gap?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "pca_dim_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved pca_dim_gap.png")

    # ── summary stats to console
    log.info("\n=== EMBEDDING GAP SUMMARY ===")
    log.info("MESA belt     : mean=%.3f  std=%.3f  norm=%.3f",
             mesa_emb.mean(), mesa_emb.std(), np.linalg.norm(mesa_emb, axis=1).mean())
    log.info("In-house mag  : mean=%.3f  std=%.3f  norm=%.3f",
             inh_emb["mag"].mean(), inh_emb["mag"].std(), np.linalg.norm(inh_emb["mag"], axis=1).mean())
    log.info("In-house pca  : mean=%.3f  std=%.3f  norm=%.3f",
             inh_emb["pca"].mean(), inh_emb["pca"].std(), np.linalg.norm(inh_emb["pca"], axis=1).mean())

    # mean L2 distance: within-domain vs cross-domain
    n = 500
    idx_a = RNG.choice(len(mesa_emb), n)
    idx_b = RNG.choice(len(mesa_emb), n)
    idx_c = RNG.choice(len(inh_emb["mag"]), n)
    d_within = np.linalg.norm(mesa_emb[idx_a] - mesa_emb[idx_b], axis=1).mean()
    d_cross  = np.linalg.norm(mesa_emb[idx_a] - inh_emb["mag"][idx_c], axis=1).mean()
    log.info("Mean L2 distance within MESA:  %.3f", d_within)
    log.info("Mean L2 distance MESA->inhouse: %.3f  (ratio %.2fx)", d_cross, d_cross / d_within)
    log.info("Top-3 shifted PCA dims (mag): %s", [(f"PC{top10[i]+1}", f"{dim_shift_mag[top10[i]]:.2f}SD") for i in range(3)])
    log.info("All outputs in %s", OUT)


if __name__ == "__main__":
    main()
