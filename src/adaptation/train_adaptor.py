"""Nonlinear adversarial domain adaptor for the effort encoder's tokens.

Motivated by CORAL's negative result (results/adaptation/coral/FINDINGS.md):
linear alignment did not close the belt-vs-accelerometer gap. This trains a
small residual MLP adaptor per-token, adversarially, with:
  1. domain-confusion loss (fool a belt-vs-accel discriminator)
  2. residual L2 penalty (anchor: adaptor starts near-identity, can't destroy signal)
  3. breathing-consistency loss (preserve the real spectral periodicity measured
     in the raw accelerometer signal — prevents the adversarial loss from just
     collapsing tokens to noise)

Train nights: S03, S06, S07, S08, S11, S12, S13, S15
Val nights (held out, never trained on): S04 (noisy), S10 (clean)

    python -m src.adaptation.train_adaptor
Outputs -> results/adaptation/adaptor/
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.mesa import canet as N
from src.mesa import cardiac_stream as C
from src.mesa import respiratory_stream as R
from src.mesa import ecg_baseline as B
from src.inhouse import streams as S

log = logging.getLogger("train_adaptor")
OUT   = Path("results/adaptation/adaptor")
MODELS = Path("data/mesa/cache/final_models")
STREAM_CACHE = Path("data/recordings/cache/streams.pkl")
DEV   = B.DEV
RNG_NP = np.random.default_rng(42)

TRAIN_NIGHTS = {"S03", "S06", "S07", "S08", "S11", "S12", "S13", "S15"}
VAL_NIGHTS   = {"S04", "S10"}

N_MESA_SUBJ        = 60
N_EPOCHS_PER_NIGHT = 400   # epochs sampled per night per training step pool
BATCH_EPOCHS       = 32    # epoch-sequences per batch (each = 75 tokens)
N_TRAIN_STEPS       = 30 * 40   # ~30 "epochs" of 40 steps each, short first run
VAL_EVERY           = 40        # steps between validation checks
LR                   = 3e-4
LAMBDA_RESIDUAL      = 0.05
LAMBDA_CONSISTENCY   = 0.3
ADAPTOR_ALPHA_INIT   = 0.1
MIN_LAG, MAX_LAG      = 4, 25   # tokens; 0.4s/token -> 1.6-10s period -> 6-37.5 bpm


# ───────────────────────────── models ─────────────────────────────

class ResidualAdaptor(nn.Module):
    """Per-token residual MLP: token' = token + alpha * MLP(token)."""
    def __init__(self, d=64, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(ADAPTOR_ALPHA_INIT)))

    def forward(self, tokens):             # (B, 75, 64)
        delta = self.net(tokens)
        alpha = torch.exp(self.log_alpha)
        return tokens + alpha * delta, delta


class Discriminator(nn.Module):
    """Pooled (GAP+GMP, 128-dim) -> belt (1) vs accelerometer (0)."""
    def __init__(self, d=64, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, tokens):              # (B, 75, 64)
        z = torch.cat([tokens.mean(1), tokens.amax(1)], 1)
        return self.net(z).squeeze(-1)


class ConsistencyReadout(nn.Module):
    """Linear projection of each token to a scalar -> sequence used for autocorrelation."""
    def __init__(self, d=64):
        super().__init__()
        self.w = nn.Linear(d, 1, bias=False)

    def forward(self, tokens):              # (B, 75, 64) -> (B, 75)
        return self.w(tokens).squeeze(-1)


def autocorr_at_lag(seq: torch.Tensor, lag: torch.Tensor) -> torch.Tensor:
    """seq: (B, T). lag: (B,) int tensor, per-sample expected lag. Returns (B,) normalised autocorr."""
    B_sz, T = seq.shape
    seq = seq - seq.mean(1, keepdim=True)
    out = torch.zeros(B_sz, device=seq.device)
    for i in range(B_sz):
        l = int(lag[i].item())
        l = max(MIN_LAG, min(MAX_LAG, l))
        if l >= T:
            continue
        a, b = seq[i, :-l], seq[i, l:]
        denom = (a.norm() * b.norm()).clamp_min(1e-6)
        out[i] = (a * b).sum() / denom
    return out


# ───────────────────────────── data ─────────────────────────────

@torch.no_grad()
def get_tokens(enc, windows, bs=256):
    enc.eval()
    out = []
    for i in range(0, len(windows), bs):
        x = torch.as_tensor(windows[i:i + bs], device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t = enc(x)
        out.append(t.float())
    return torch.cat(out).cpu()   # keep on CPU, move per-batch (saves GPU mem for discriminator grads)


def resp_bpm_from_raw(x32: np.ndarray) -> float:
    """Spectral-peak breathing rate from a 32 Hz, 30 s raw accel-effort epoch (same method as run_canet.py)."""
    F = np.abs(np.fft.rfft(x32 * np.hanning(len(x32)))) ** 2
    f = np.fft.rfftfreq(len(x32), 1 / R.BELT_HZ)
    pk = (f >= 0.1) & (f <= 0.6)
    return float(f[pk][np.argmax(F[pk])] * 60)


def build_night_data(nights, codes):
    """Returns dict: code -> (tokens (n_ep,75,64) torch cpu, resp_bpm (n_ep,) np)."""
    out = {}
    for s in nights:
        if s.code not in codes:
            continue
        tgt = np.flatnonzero(s.n_beats >= B.CFG["min_beats"])
        if len(tgt) == 0:
            continue
        n = min(N_EPOCHS_PER_NIGHT, len(tgt))
        tgt = np.sort(RNG_NP.choice(tgt, n, replace=False))
        ew = S.effort_windows(s.effort["mag"], tgt)                    # (n,2,4800) belt-like
        bpm = np.array([resp_bpm_from_raw(
            s.effort_raw["mag"][int(e * 30 * R.BELT_HZ):int((e + 1) * 30 * R.BELT_HZ)]) for e in tgt])
        out[s.code] = (ew, bpm)
        log.info("  %s: %d epochs prepared", s.code, n)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    log.info("Loading headline model (frozen) ...")
    ck = torch.load(MODELS / "headline.pt", map_location=DEV, weights_only=False)
    model = N.CANet(ck["streams"], ck["fusion"]).to(DEV)
    model.load_state_dict(ck["state"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    enc = model.enc["resp"]
    thr = ck["thr"]

    # ── MESA belt tokens (source domain)
    log.info("Extracting MESA belt tokens ...")
    post = np.load(B.PRED_CACHE / "preds.npz")
    all_subj = np.unique(post["subject"])
    sel_subj = RNG_NP.choice(all_subj, min(N_MESA_SUBJ, len(all_subj)), replace=False)
    mesa_tok = []
    for mid in sel_subj:
        ep = post["epoch"][post["subject"] == mid]
        ep = np.sort(RNG_NP.choice(ep, min(30, len(ep)), replace=False))
        belts = R.load_subject(mid)["belts"].astype(np.float32)
        wins = S.effort_windows(belts, ep)
        mesa_tok.append(get_tokens(enc, wins))
    mesa_tok = torch.cat(mesa_tok)             # (N, 75, 64) cpu
    log.info("  MESA belt tokens: %s", tuple(mesa_tok.shape))

    # ── in-house effort windows, split by night
    log.info("Loading in-house stream cache ...")
    nights = pickle.load(open(STREAM_CACHE, "rb"))
    train_data = build_night_data(nights, TRAIN_NIGHTS)
    val_data   = build_night_data(nights, VAL_NIGHTS)
    log.info("Train nights: %s | Val nights: %s", list(train_data), list(val_data))

    # pre-compute train accel tokens (frozen encoder, no grad needed for tokens themselves)
    train_tok, train_bpm = [], []
    for code, (ew, bpm) in train_data.items():
        train_tok.append(get_tokens(enc, ew))
        train_bpm.append(bpm)
    train_tok = torch.cat(train_tok)
    train_bpm = np.concatenate(train_bpm)
    log.info("Train accel tokens: %s", tuple(train_tok.shape))

    # ── adaptor + discriminator + consistency readout
    adaptor = ResidualAdaptor().to(DEV)
    disc    = Discriminator().to(DEV)
    readout = ConsistencyReadout().to(DEV)
    opt_a = torch.optim.Adam(list(adaptor.parameters()) + list(readout.parameters()), lr=LR)
    opt_d = torch.optim.Adam(disc.parameters(), lr=LR)

    history = []
    log.info("Starting training: %d steps, validate every %d", N_TRAIN_STEPS, VAL_EVERY)

    for step in range(1, N_TRAIN_STEPS + 1):
        # sample batch
        ib = RNG_NP.choice(len(mesa_tok), BATCH_EPOCHS, replace=False)
        ia = RNG_NP.choice(len(train_tok), BATCH_EPOCHS, replace=False)
        belt_b  = mesa_tok[ib].to(DEV)
        accel_b = train_tok[ia].to(DEV)
        bpm_b   = torch.as_tensor(train_bpm[ia], device=DEV)
        lag_b   = (60.0 / bpm_b.clamp(min=1e-3)) / 0.4   # seconds / (s/token)

        # ---- discriminator step
        with torch.no_grad():
            adapted_det, _ = adaptor(accel_b)
        d_belt = disc(belt_b)
        d_acc  = disc(adapted_det)
        loss_d = (nn.functional.binary_cross_entropy_with_logits(d_belt, torch.ones_like(d_belt)) +
                 nn.functional.binary_cross_entropy_with_logits(d_acc, torch.zeros_like(d_acc)))
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()

        # ---- adaptor step
        adapted, delta = adaptor(accel_b)
        d_acc2 = disc(adapted)
        loss_adv = nn.functional.binary_cross_entropy_with_logits(d_acc2, torch.ones_like(d_acc2))
        loss_res = (delta ** 2).mean()
        seq = readout(adapted)
        ac = autocorr_at_lag(seq, lag_b)
        loss_cons = -ac.mean()
        loss_a = loss_adv + LAMBDA_RESIDUAL * loss_res + LAMBDA_CONSISTENCY * loss_cons
        opt_a.zero_grad(); loss_a.backward(); opt_a.step()

        if step % VAL_EVERY == 0 or step == 1:
            with torch.no_grad():
                d_belt_auc = torch.sigmoid(d_belt).mean().item()
                d_acc_auc  = torch.sigmoid(d_acc2).mean().item()
                alpha_val  = torch.exp(adaptor.log_alpha).item()
            row = dict(step=step, loss_d=loss_d.item(), loss_adv=loss_adv.item(),
                      loss_res=loss_res.item(), loss_cons=loss_cons.item(),
                      autocorr=ac.mean().item(), alpha=alpha_val,
                      disc_p_belt=d_belt_auc, disc_p_adapted_after_adapt=d_acc_auc)
            history.append(row)
            log.info("step %4d | D: %.3f | adv: %.3f | res: %.4f | cons: %.3f (autocorr %.3f) | alpha %.3f | disc(belt)=%.2f disc(adapted)=%.2f",
                     step, loss_d.item(), loss_adv.item(), loss_res.item(), loss_cons.item(),
                     ac.mean().item(), alpha_val, d_belt_auc, d_acc_auc)

    # ── save training curves
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(OUT / "training_history.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(hist_df.step, hist_df.loss_d, label="disc loss")
    axes[0, 0].plot(hist_df.step, hist_df.loss_adv, label="adaptor adv loss")
    axes[0, 0].legend(); axes[0, 0].set(title="Adversarial losses", xlabel="step")
    axes[0, 1].plot(hist_df.step, hist_df.disc_p_belt, label="disc(belt)")
    axes[0, 1].plot(hist_df.step, hist_df.disc_p_adapted_after_adapt, label="disc(adapted)")
    axes[0, 1].axhline(0.5, color="grey", ls=":")
    axes[0, 1].legend(); axes[0, 1].set(title="Discriminator confidence (want both -> 0.5)", xlabel="step")
    axes[1, 0].plot(hist_df.step, hist_df.autocorr)
    axes[1, 0].set(title="Breathing-consistency autocorrelation (higher=better)", xlabel="step")
    axes[1, 1].plot(hist_df.step, hist_df.alpha)
    axes[1, 1].set(title="Adaptor alpha (residual magnitude)", xlabel="step")
    fig.tight_layout(); fig.savefig(OUT / "training_curves.png", dpi=130); plt.close(fig)

    # ── validation: real downstream flag rate on held-out nights
    log.info("\n=== VALIDATION on held-out nights %s ===", list(val_data))
    W_cross = model.cross
    val_rows = []
    adaptor.eval()
    for code, (ew, bpm) in val_data.items():
        s = next(x for x in nights if x.code == code)
        tgt = np.flatnonzero(s.n_beats >= B.CFG["min_beats"])
        n = min(N_EPOCHS_PER_NIGHT, len(tgt))
        tgt_sorted = np.sort(RNG_NP.choice(tgt, n, replace=False)) if len(tgt) > n else tgt
        cw = S.cardiac_windows(s, tgt_sorted)
        ew_full = S.effort_windows(s.effort["mag"], tgt_sorted)
        with torch.no_grad():
            probs = []
            for i in range(0, len(cw), 256):
                xc = torch.as_tensor(cw[i:i + 256], device=DEV)
                xe = torch.as_tensor(ew_full[i:i + 256], device=DEV)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tc = model.enc["cardiac"](xc).float()
                    te = model.enc["resp"](xe).float()
                te_adapted, _ = adaptor(te)
                t_att = model.cross({"cardiac": tc, "resp": te_adapted})
                z = torch.cat([torch.cat([t_att["cardiac"].mean(1), t_att["cardiac"].amax(1)], 1),
                              torch.cat([t_att["resp"].mean(1), t_att["resp"].amax(1)], 1)], 1)
                p = torch.sigmoid(model.head(z).squeeze(-1))
                probs.append(p.cpu().numpy())
            probs = np.concatenate(probs)
        flag = (probs >= thr).mean()
        val_rows.append(dict(night=code, n_epochs=len(probs), frac_flagged=float(flag), mean_prob=float(probs.mean())))
        log.info("  %s: %.1f%% flagged (was ~93-97%% before adaptation)", code, flag * 100)

    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(OUT / "validation_flag_rate.csv", index=False)

    # domain AUROC check on val-night tokens vs MESA (held out from training, honest check)
    val_tok = torch.cat([get_tokens(enc, ew) for ew, _ in val_data.values()])
    with torch.no_grad():
        val_adapted, _ = adaptor(val_tok.to(DEV))
    Xb = torch.cat([mesa_tok.mean(1), mesa_tok.amax(1)], 1).numpy()
    Xa = torch.cat([val_adapted.mean(1), val_adapted.amax(1)], 1).cpu().numpy()
    n = min(len(Xb), len(Xa), 2000)
    ib = RNG_NP.choice(len(Xb), n, replace=False); ia = RNG_NP.choice(len(Xa), n, replace=False)
    X = np.concatenate([Xb[ib], Xa[ia]]); y = np.r_[np.zeros(n), np.ones(n)]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(X, y)
    val_auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    log.info("Held-out domain AUROC (val nights, adapted, vs MESA): %.4f", val_auc)

    torch.save(dict(adaptor=adaptor.state_dict(), readout=readout.state_dict(),
                    disc=disc.state_dict(), alpha=torch.exp(adaptor.log_alpha).item()),
              OUT / "adaptor_checkpoint.pt")

    summary = dict(
        n_steps=N_TRAIN_STEPS, train_nights=sorted(TRAIN_NIGHTS), val_nights=sorted(VAL_NIGHTS),
        final_alpha=torch.exp(adaptor.log_alpha).item(),
        final_disc_belt=float(hist_df.disc_p_belt.iloc[-1]),
        final_disc_adapted=float(hist_df.disc_p_adapted_after_adapt.iloc[-1]),
        val_domain_auroc=val_auc,
        val_flag_rate_median=float(val_df.frac_flagged.median()),
        flag_rate_before_adaptation_reference="~93-100% (from CORAL run / original)",
        mesa_lowAHI_reference=0.28,
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("\n=== SUMMARY ===\n%s", json.dumps(summary, indent=2))
    log.info("All outputs in %s", OUT)


if __name__ == "__main__":
    main()
