"""Adaptor attempt 2: much smaller capacity, stronger anchor, balanced LRs,
   early-stops on held-out domain AUROC (never lets training overfit past
   the point where attempt 1 failed).

Same train/val split as attempt 1. See results/adaptation/adaptor/FINDINGS.md
for why attempt 1 (hidden=128, lambda_res=0.05) failed.

    python -m src.adaptation.train_adaptor_v2
Outputs -> results/adaptation/adaptor_v2/
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

log = logging.getLogger("train_adaptor_v2")
OUT   = Path("results/adaptation/adaptor_v2")
MODELS = Path("data/mesa/cache/final_models")
STREAM_CACHE = Path("data/recordings/cache/streams.pkl")
DEV   = B.DEV
RNG_NP = np.random.default_rng(42)

TRAIN_NIGHTS = {"S03", "S06", "S07", "S08", "S11", "S12", "S13", "S15"}
VAL_NIGHTS   = {"S04", "S10"}

N_MESA_SUBJ        = 60
N_EPOCHS_PER_NIGHT = 400
BATCH_EPOCHS       = 32
N_TRAIN_STEPS       = 1200
VAL_EVERY            = 40
LR_ADAPTOR            = 3e-4
LR_DISC                = 1e-4     # 3x slower than adaptor -> adaptor keeps pace
LAMBDA_RESIDUAL        = 0.3       # 6x stronger anchor than attempt 1
LAMBDA_CONSISTENCY     = 0.3
ADAPTOR_ALPHA_INIT      = 0.05     # smaller initial license to move tokens
ADAPTOR_HIDDEN           = 32      # 4x smaller than attempt 1 (was 128)
MIN_LAG, MAX_LAG          = 4, 25
PRE_ADAPTATION_AUROC_REF   = 0.94  # measured baseline; stop early if we exceed this


class ResidualAdaptor(nn.Module):
    def __init__(self, d=64, hidden=ADAPTOR_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(ADAPTOR_ALPHA_INIT)))

    def forward(self, tokens):
        delta = self.net(tokens)
        alpha = torch.exp(self.log_alpha)
        return tokens + alpha * delta, delta


class Discriminator(nn.Module):
    def __init__(self, d=64, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, tokens):
        z = torch.cat([tokens.mean(1), tokens.amax(1)], 1)
        return self.net(z).squeeze(-1)


class ConsistencyReadout(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.w = nn.Linear(d, 1, bias=False)

    def forward(self, tokens):
        return self.w(tokens).squeeze(-1)


def autocorr_at_lag(seq, lag):
    B_sz, T = seq.shape
    seq = seq - seq.mean(1, keepdim=True)
    out = torch.zeros(B_sz, device=seq.device)
    for i in range(B_sz):
        l = max(MIN_LAG, min(MAX_LAG, int(lag[i].item())))
        if l >= T:
            continue
        a, b = seq[i, :-l], seq[i, l:]
        out[i] = (a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-6)
    return out


@torch.no_grad()
def get_tokens(enc, windows, bs=256):
    enc.eval()
    out = []
    for i in range(0, len(windows), bs):
        x = torch.as_tensor(windows[i:i + bs], device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t = enc(x)
        out.append(t.float())
    return torch.cat(out).cpu()


def resp_bpm_from_raw(x32):
    F = np.abs(np.fft.rfft(x32 * np.hanning(len(x32)))) ** 2
    f = np.fft.rfftfreq(len(x32), 1 / R.BELT_HZ)
    pk = (f >= 0.1) & (f <= 0.6)
    return float(f[pk][np.argmax(F[pk])] * 60)


def build_night_data(nights, codes):
    out = {}
    for s in nights:
        if s.code not in codes:
            continue
        tgt = np.flatnonzero(s.n_beats >= B.CFG["min_beats"])
        if len(tgt) == 0:
            continue
        n = min(N_EPOCHS_PER_NIGHT, len(tgt))
        tgt = np.sort(RNG_NP.choice(tgt, n, replace=False))
        ew = S.effort_windows(s.effort["mag"], tgt)
        bpm = np.array([resp_bpm_from_raw(
            s.effort_raw["mag"][int(e * 30 * R.BELT_HZ):int((e + 1) * 30 * R.BELT_HZ)]) for e in tgt])
        out[s.code] = (ew, bpm)
        log.info("  %s: %d epochs prepared", s.code, n)
    return out


def held_out_domain_auroc(mesa_tok, adaptor, val_tok_dict):
    """val_tok_dict: code -> raw (pre-adaptor) tokens (cpu). Returns AUROC after adapting.

    FIXED (attempt 2 bug): classifier now fit on a TRAIN split and scored on a
    disjoint TEST split within this function -- the original version fit and
    scored on the same data (resubstitution), which can show AUROC=1.0 even
    when domains are not separable, making every "held-out" number upstream
    uninterpretable. This uses a simple random 50/50 split, re-done every call
    (cheap, and val_tok_dict itself is already held out from training epochs)."""
    adaptor.eval()
    with torch.no_grad():
        val_tok = torch.cat(list(val_tok_dict.values()))
        val_adapted, _ = adaptor(val_tok.to(DEV))
    Xb = torch.cat([mesa_tok.mean(1), mesa_tok.amax(1)], 1).numpy()
    Xa = torch.cat([val_adapted.mean(1), val_adapted.amax(1)], 1).cpu().numpy()
    n = min(len(Xb), len(Xa), 2000)
    ib = RNG_NP.choice(len(Xb), n, replace=False); ia = RNG_NP.choice(len(Xa), n, replace=False)
    X = np.concatenate([Xb[ib], Xa[ia]]); y = np.r_[np.zeros(n), np.ones(n)]
    idx = RNG_NP.permutation(len(X))
    split = len(X) // 2
    tr, te = idx[:split], idx[split:]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(X[tr], y[tr])
    adaptor.train()
    return roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])


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
    mesa_tok = torch.cat(mesa_tok)
    log.info("  MESA belt tokens: %s", tuple(mesa_tok.shape))

    log.info("Loading in-house stream cache ...")
    nights = pickle.load(open(STREAM_CACHE, "rb"))
    train_data = build_night_data(nights, TRAIN_NIGHTS)
    val_data   = build_night_data(nights, VAL_NIGHTS)

    train_tok, train_bpm = [], []
    for code, (ew, bpm) in train_data.items():
        train_tok.append(get_tokens(enc, ew)); train_bpm.append(bpm)
    train_tok = torch.cat(train_tok); train_bpm = np.concatenate(train_bpm)

    val_tok_dict = {code: get_tokens(enc, ew) for code, (ew, _) in val_data.items()}

    log.info("Pre-adaptation (identity) held-out domain AUROC check ...")
    identity = ResidualAdaptor().to(DEV)
    with torch.no_grad():
        identity.log_alpha.data = torch.log(torch.tensor(1e-8))  # ~no-op
    pre_auc = held_out_domain_auroc(mesa_tok, identity, val_tok_dict)
    log.info("  Pre-adaptation held-out AUROC (sanity check, should be ~0.93-0.95): %.4f", pre_auc)
    del identity

    adaptor = ResidualAdaptor().to(DEV)
    disc    = Discriminator().to(DEV)
    readout = ConsistencyReadout().to(DEV)
    opt_a = torch.optim.Adam(list(adaptor.parameters()) + list(readout.parameters()), lr=LR_ADAPTOR)
    opt_d = torch.optim.Adam(disc.parameters(), lr=LR_DISC)

    history = []
    best_val_auc, best_state, stopped_early, stop_step = 1.0, None, False, N_TRAIN_STEPS

    log.info("Starting training: %d steps max, early-stop if held-out AUROC > %.2f", N_TRAIN_STEPS, PRE_ADAPTATION_AUROC_REF)

    for step in range(1, N_TRAIN_STEPS + 1):
        ib = RNG_NP.choice(len(mesa_tok), BATCH_EPOCHS, replace=False)
        ia = RNG_NP.choice(len(train_tok), BATCH_EPOCHS, replace=False)
        belt_b  = mesa_tok[ib].to(DEV)
        accel_b = train_tok[ia].to(DEV)
        bpm_b   = torch.as_tensor(train_bpm[ia], device=DEV)
        lag_b   = (60.0 / bpm_b.clamp(min=1e-3)) / 0.4

        with torch.no_grad():
            adapted_det, _ = adaptor(accel_b)
        d_belt = disc(belt_b); d_acc = disc(adapted_det)
        loss_d = (nn.functional.binary_cross_entropy_with_logits(d_belt, torch.ones_like(d_belt)) +
                 nn.functional.binary_cross_entropy_with_logits(d_acc, torch.zeros_like(d_acc)))
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()

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
            val_auc = held_out_domain_auroc(mesa_tok, adaptor, val_tok_dict)
            alpha_val = torch.exp(adaptor.log_alpha).item()
            row = dict(step=step, loss_d=loss_d.item(), loss_adv=loss_adv.item(),
                      loss_res=loss_res.item(), loss_cons=loss_cons.item(),
                      autocorr=ac.mean().item(), alpha=alpha_val,
                      disc_p_belt=torch.sigmoid(d_belt).mean().item(),
                      disc_p_adapted=torch.sigmoid(d_acc2).mean().item(),
                      held_out_val_auc=val_auc)
            history.append(row)
            log.info("step %4d | D:%.3f adv:%.3f res:%.4f cons:%.3f | alpha %.4f | disc(belt)=%.2f disc(adapt)=%.2f | HELD-OUT AUC: %.4f",
                     step, loss_d.item(), loss_adv.item(), loss_res.item(), loss_cons.item(),
                     alpha_val, torch.sigmoid(d_belt).mean().item(), torch.sigmoid(d_acc2).mean().item(), val_auc)

            if val_auc < best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in adaptor.state_dict().items()}

            if val_auc > PRE_ADAPTATION_AUROC_REF and step > 200:
                log.warning("EARLY STOP at step %d: held-out AUROC %.4f exceeded pre-adaptation reference %.2f",
                           step, val_auc, PRE_ADAPTATION_AUROC_REF)
                stopped_early, stop_step = True, step
                break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(OUT / "training_history.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(hist_df.step, hist_df.loss_d, label="disc loss")
    axes[0, 0].plot(hist_df.step, hist_df.loss_adv, label="adaptor adv loss")
    axes[0, 0].legend(); axes[0, 0].set(title="Adversarial losses", xlabel="step")
    axes[0, 1].plot(hist_df.step, hist_df.disc_p_belt, label="disc(belt)")
    axes[0, 1].plot(hist_df.step, hist_df.disc_p_adapted, label="disc(adapted)")
    axes[0, 1].axhline(0.5, color="grey", ls=":")
    axes[0, 1].legend(); axes[0, 1].set(title="Discriminator confidence", xlabel="step")
    axes[1, 0].plot(hist_df.step, hist_df.held_out_val_auc, color="red")
    axes[1, 0].axhline(pre_auc, color="grey", ls="--", label=f"pre-adaptation ({pre_auc:.3f})")
    axes[1, 0].axhline(0.5, color="green", ls=":", label="ideal (0.5)")
    axes[1, 0].legend(); axes[1, 0].set(title="HELD-OUT domain AUROC (the real test)", xlabel="step")
    axes[1, 1].plot(hist_df.step, hist_df.alpha)
    axes[1, 1].set(title="Adaptor alpha", xlabel="step")
    fig.tight_layout(); fig.savefig(OUT / "training_curves.png", dpi=130); plt.close(fig)

    # use best checkpoint by held-out AUROC, not final
    if best_state is not None:
        adaptor.load_state_dict(best_state)
    log.info("\nBest held-out AUROC achieved during training: %.4f (pre-adaptation was %.4f)", best_val_auc, pre_auc)

    # downstream flag rate using best checkpoint
    log.info("Computing downstream flag rate with BEST checkpoint ...")
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
        log.info("  %s: %.1f%% flagged", code, flag * 100)

    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(OUT / "validation_flag_rate.csv", index=False)

    torch.save(dict(adaptor=adaptor.state_dict(), readout=readout.state_dict(), disc=disc.state_dict()),
              OUT / "adaptor_checkpoint_best.pt")

    verdict = "IMPROVED (generalizes)" if best_val_auc < pre_auc - 0.02 else \
             "NO IMPROVEMENT (within noise or worse)" if best_val_auc >= pre_auc - 0.02 else "UNCLEAR"

    summary = dict(
        attempt=2, hidden_dim=ADAPTOR_HIDDEN, lambda_residual=LAMBDA_RESIDUAL,
        lr_adaptor=LR_ADAPTOR, lr_disc=LR_DISC,
        stopped_early=stopped_early, stop_step=stop_step,
        pre_adaptation_held_out_auroc=pre_auc,
        best_held_out_auroc_during_training=best_val_auc,
        final_held_out_auroc=float(hist_df.held_out_val_auc.iloc[-1]),
        val_flag_rate_median=float(val_df.frac_flagged.median()),
        verdict=verdict,
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("\n=== SUMMARY ===\n%s", json.dumps(summary, indent=2))
    log.info("VERDICT: %s", verdict)


if __name__ == "__main__":
    main()
