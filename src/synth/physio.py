"""Beat-level cardio-respiratory simulator (vectorised over a batch of independent recordings).

One engine drives (a) the per-minute (RR, R-amplitude) sequences used for the classifier experiments and (b) the SensorTile waveform
renderer (`render.py`).  Ground truth comes from the event schedule, never from the generated signal.

Everything numeric below is either (M) estimated from data, (L) a literature-style prior, or (A) a simulator ASSUMPTION; see PRIORS.
Deterministic: all randomness comes from the np.random.Generator passed in.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

GEN_VERSION = "synth-physio-1.0"

# kind codes
HOLD, CENTRAL, OBSTRUCTIVE = 1, 2, 3


@dataclass
class Priors:
    """Population priors.  Fields marked (M) are set per experimental fold from the TRAINING records only (see priors_from_sequences)."""
    log_rr0_mean: float = np.log(0.92)      # (M) median RR across training records
    log_rr0_sd: float = 0.12                # (M)
    log_hrv_mean: float = np.log(0.05)      # (M) per-record SD of RR in non-apnea minutes (s)
    log_hrv_sd: float = 0.5                 # (M)
    log_amp_mean: float = 0.0               # (M) per-record median R amplitude (signal units)
    log_amp_sd: float = 0.8                 # (M)
    p_gap_minute: float = 0.005             # (M) minutes containing a long signal gap
    gap_values: np.ndarray = field(default_factory=lambda: np.array([5., 10., 20., 60.]))  # (M) empirical max-RR of gap minutes
    # (L)/(A) apnea structure ------------------------------------------------------------------------------------------
    severity_probs: tuple = (0.25, 0.15, 0.25, 0.35)       # (A) none / mild / moderate / severe  (events per hour ranges below)
    severity_rate_per_h: tuple = ((0, 0), (5, 15), (15, 30), (30, 70))
    event_dur_s: tuple = (10.0, 45.0)       # (L) apnea durations; AASM min 10 s
    brady_frac: tuple = (0.06, 0.28)        # (L) RR prolongation at end of event (Hayano: HR swing 10-30 bpm)
    surge_frac: tuple = (0.05, 0.20)        # (L)
    surge_tau_s: tuple = (5.0, 15.0)        # (L)
    cvhr_visibility_logsd: float = 0.45     # (A) record-to-record heterogeneity of the cardiac response
    p_weak_cvhr: float = 0.25               # (A) fraction of records whose CVHR is weak (x0.3)
    central_frac: float = 0.4               # (A) rest obstructive
    # (A) background physiology
    rsa_share: float = 0.6; lf_share: float = 0.4; white_frac: float = 0.15
    breath_hz: tuple = (0.15, 0.35)         # (M in-house p5-p95 0.18-0.45; L)
    edr_depth: tuple = (0.03, 0.15)
    p_ectopic_rec: float = 0.4; ectopic_rate: tuple = (0.001, 0.01); p_missed: tuple = (0.0, 0.004)
    rr_jitter_s: tuple = (0.002, 0.008)


def priors_from_sequences(seq: np.ndarray, y: np.ndarray, rec: np.ndarray) -> Priors:
    """Training-only estimation. seq (n,2,60) [RR s, R amp], y 0/1, rec record names (training records ONLY)."""
    p = Priors(); rr = seq[:, 0]; amp = seq[:, 1]
    med_rr, sd_rr, med_amp = [], [], []
    for r in np.unique(rec):
        m = rec == r; mn = m & (y == 0)
        ok = m & (rr.max(1) < 3)
        if ok.sum() < 20: continue
        med_rr.append(np.median(rr[ok].mean(1))); med_amp.append(np.median(amp[ok].mean(1)))
        k = mn & (rr.max(1) < 3)
        sd_rr.append(np.median(rr[k].std(1)) if k.sum() >= 10 else np.median(rr[ok].std(1)))
    med_rr, sd_rr, med_amp = map(np.array, (med_rr, sd_rr, med_amp))
    p.log_rr0_mean, p.log_rr0_sd = float(np.log(med_rr).mean()), float(np.log(med_rr).std() + 1e-3)
    p.log_hrv_mean, p.log_hrv_sd = float(np.log(np.clip(sd_rr, 1e-3, None)).mean()), float(np.log(np.clip(sd_rr, 1e-3, None)).std() + 1e-3)
    ma = np.clip(med_amp, 1e-3, None); p.log_amp_mean, p.log_amp_sd = float(np.log(ma).mean()), float(np.log(ma).std() + 1e-3)
    big = rr.max(1) >= 3
    p.p_gap_minute = float(big.mean()); p.gap_values = rr.max(1)[big] if big.sum() >= 3 else p.gap_values
    return p


@dataclass
class BatchRec:
    beat_t: list            # per recording: beat times (s) = time of the R peak ending each RR interval
    rr: list
    amp: list
    ev_start: np.ndarray    # (B,K) seconds, inf where unused
    ev_end: np.ndarray
    ev_kind: np.ndarray     # (B,K) 0 unused
    params: dict            # per-recording parameter arrays (for the manifest)
    resp_phase_fn: object = None


def draw_events(rng, B, T, pr: Priors, severity=None, forced=None):
    """Event schedule (B,K). `forced`: list (len B) of [(start,end,kind),..] overrides random schedule (used for breath-hold sessions)."""
    if forced is not None:
        K = max(1, max(len(f) for f in forced)); st = np.full((B, K), np.inf); en = np.full((B, K), np.inf); kd = np.zeros((B, K), int)
        for b, f in enumerate(forced):
            for k, (a, e, c) in enumerate(f): st[b, k], en[b, k], kd[b, k] = a, e, c
        return st, en, kd, np.full(B, -1)
    sev = rng.choice(4, B, p=np.array(pr.severity_probs) / np.sum(pr.severity_probs)) if severity is None else np.full(B, severity)
    lo = np.array([pr.severity_rate_per_h[s][0] for s in sev]); hi = np.array([pr.severity_rate_per_h[s][1] for s in sev])
    lam = rng.uniform(lo, np.maximum(hi, lo + 1e-9)); K = int(max(3, 75 * T / 3600 + 10))
    D = rng.uniform(*pr.event_dur_s, (B, K))
    cyc = np.where(lam[:, None] > 0, 3600.0 / np.maximum(lam[:, None], 1e-9), np.inf)
    gap = np.maximum(8.0, rng.exponential(np.maximum(cyc - D.mean(1, keepdims=True), 5.0), (B, K)))
    first = rng.uniform(60, 600, B)
    st = first[:, None] + np.cumsum(np.concatenate([np.zeros((B, 1)), D[:, :-1] + gap[:, :-1]], 1), 1)
    en = st + D; kd = np.where(rng.random((B, K)) < pr.central_frac, CENTRAL, OBSTRUCTIVE)
    dead = (lam[:, None] == 0) | (st > T); st = np.where(dead, np.inf, st); en = np.where(dead, np.inf, en); kd = np.where(dead, 0, kd)
    return st, en, kd, sev


def simulate_batch(rng: np.random.Generator, B: int, T: float, pr: Priors, severity=None, forced=None, voluntary=False) -> BatchRec:
    """B independent recordings of T seconds. Beat-by-beat loop vectorised over the batch."""
    st, en, kd, sev = draw_events(rng, B, T, pr, severity, forced)
    K = st.shape[1]
    rr0 = np.clip(np.exp(rng.normal(pr.log_rr0_mean, pr.log_rr0_sd, B)), 0.55, 1.4)
    hrv_sd = np.clip(np.exp(rng.normal(pr.log_hrv_mean, pr.log_hrv_sd, B)), 0.008, 0.25)         # per-record SD of RR (s) excluding apnea
    tot = hrv_sd / rr0
    a_rsa = tot * np.sqrt(pr.rsa_share) * np.sqrt(2); a_lf = tot * np.sqrt(pr.lf_share) * np.sqrt(2); a_wh = tot * pr.white_frac
    f_r = rng.uniform(*pr.breath_hz, B); f_lf = rng.uniform(0.04, 0.12, B); ph_r = rng.uniform(0, 2 * np.pi, B); ph_lf = rng.uniform(0, 2 * np.pi, B)
    drift = np.zeros(B)                                                           # slow breathing-rate drift (AR(1) on log f)
    G = np.exp(rng.normal(pr.log_amp_mean, pr.log_amp_sd, B)); edr = rng.uniform(*pr.edr_depth, B); c_hr = rng.uniform(-0.8, 0.8, B); amp_noise = np.exp(rng.normal(np.log(0.03), 0.6, B)); th = rng.uniform(0, 2 * np.pi, B)
    kappa = np.exp(rng.normal(0, pr.cvhr_visibility_logsd, B)) * np.where(rng.random(B) < pr.p_weak_cvhr, 0.3, 1.0)
    bk = rng.uniform(*pr.brady_frac, (B, K)) * kappa[:, None]; sg = rng.uniform(*pr.surge_frac, (B, K)) * kappa[:, None]; tau = rng.uniform(*pr.surge_tau_s, (B, K))
    if voluntary:   # awake voluntary hold: mild response (A)
        bk *= 0.35; sg *= 0.35
    resp_scale = np.where(kd == CENTRAL, 0.1, np.where(kd == OBSTRUCTIVE, 1.2, 0.05))
    t = rng.uniform(0, 1, B); ptr = np.full(B, -1); n_max = int(T / 0.3) + 5
    ph_a = np.full((B, n_max), np.nan); bt = np.full((B, n_max), np.nan); rr_a = np.full((B, n_max), np.nan); am = np.full((B, n_max), np.nan); n = 0; idx = np.arange(B)
    active = np.ones(B, bool)
    while active.any() and n < n_max:
        # advance event pointer
        nxt = np.minimum(ptr + 1, K - 1)
        adv = (st[idx, nxt] <= t) & (ptr + 1 < K); ptr = np.where(adv, ptr + 1, ptr)
        has = ptr >= 0; pc = np.maximum(ptr, 0)
        s_, e_, k_ = st[idx, pc], en[idx, pc], kd[idx, pc]
        inside = has & (t >= s_) & (t < e_) & (k_ > 0); after = has & (t >= e_) & (k_ > 0)
        D = np.maximum(e_ - s_, 1e-3); u = np.clip((t - s_) / D, 0, 1)
        mod = np.zeros(B)
        mod = np.where(inside, bk[idx, pc] * u ** 1.5, mod)
        dt_e = np.where(after, t - e_, 0.0)
        post = bk[idx, pc] * np.exp(-dt_e / 1.5) - sg[idx, pc] * (1 - np.exp(-dt_e / 1.5)) * np.exp(-dt_e / tau[idx, pc])
        mod = np.where(after & (dt_e < 60), post, mod)
        rs = np.where(inside, resp_scale[idx, pc], 1.0)
        drift = 0.98 * drift + 0.02 * rng.normal(0, 0.25, B)
        fr = f_r * np.exp(drift)
        rr = rr0 * (1 + a_rsa * rs * np.sin(ph_r) + a_lf * np.sin(ph_lf) + mod) + a_wh * rr0 * rng.normal(0, 1, B)
        rr = np.clip(rr, 0.3, 2.2)
        t = t + rr; ph_r = ph_r + 2 * np.pi * fr * rr; ph_lf = ph_lf + 2 * np.pi * f_lf * rr
        a = G * (1 + edr * rs * np.sin(ph_r + th) + c_hr * mod + 0.04 * np.sin(2 * np.pi * 0.02 * t + th) + rng.normal(0, 1, B) * amp_noise)
        wr = active & (t <= T)
        bt[wr, n], rr_a[wr, n], am[wr, n], ph_a[wr, n] = t[wr], rr[wr], np.maximum(a[wr], 0.02 * G[wr]), ph_r[wr]
        active = active & (t < T); n += 1
    # per-recording post-processing: ectopics, missed beats, detector jitter
    out_t, out_rr, out_a, out_p = [], [], [], []
    ect_on = rng.random(B) < pr.p_ectopic_rec; ect = np.where(ect_on, rng.uniform(*pr.ectopic_rate, B), 0.0)
    miss = rng.uniform(*pr.p_missed, B); jit = rng.uniform(*pr.rr_jitter_s, B)
    for b in range(B):
        m = ~np.isnan(bt[b]); r = rr_a[b][m].copy(); a = am[b][m].copy(); ph = ph_a[b][m].copy(); L = len(r)
        e = np.where(rng.random(L - 1) < ect[b])[0]
        if len(e): r[e] *= 0.65; r[e + 1] *= 1.35
        mi = np.where(rng.random(L - 1) < miss[b])[0]
        if len(mi): r[mi] = r[mi] + r[mi + 1]; r[mi + 1] = np.nan; a[mi + 1] = np.nan
        r = r + rng.normal(0, jit[b], L)                                           # beat-time (R-peak localisation) jitter, differenced
        keep = ~np.isnan(r); r, a, ph = r[keep], a[keep], ph[keep]; tt = np.cumsum(r) + (bt[b][m][0] - rr_a[b][m][0])
        ok = (r > 0.2) & (tt <= T); out_t.append(tt[ok]); out_rr.append(r[ok]); out_a.append(a[ok]); out_p.append(ph[ok])
    params = dict(rr0=rr0, hrv_sd=hrv_sd, f_resp=f_r, G=G, edr=edr, c_hr=c_hr, kappa=kappa, severity=sev, ectopic_rate=ect, missed=miss, jitter=jit)
    return BatchRec(out_t, out_rr, out_a, st, en, kd, params, resp_phase_fn=out_p)


def to_minutes(br: BatchRec, rng, pr: Priors, T: float, label_min_overlap_s: float = 10.0, seq_len: int = 60, add_gaps: bool = True):
    """Group beats into 60 s windows exactly like src/ecg/rr_amp_sequence (RR/amplitude linearly interpolated on `seq_len` points by beat
    index; windows with < 3 beats dropped). Label A if scheduled events overlap the minute by >= label_min_overlap_s (assumption)."""
    seqs, lab, recid, mins = [], [], [], []
    for b in range(len(br.beat_t)):
        t, r, a = br.beat_t[b], br.rr[b], br.amp[b]
        nmin = int(T // 60); edges = np.searchsorted(t, np.arange(nmin + 1) * 60.0)
        s, e = br.ev_start[b], br.ev_end[b]; valid = np.isfinite(s)
        for m in range(nmin):
            i0, i1 = edges[m], edges[m + 1]
            if i1 - i0 < 3: continue
            x = np.linspace(0, 1, i1 - i0); xd = np.linspace(0, 1, seq_len)
            rr_g = np.interp(xd, x, r[i0:i1]); am_g = np.interp(xd, x, a[i0:i1])
            if add_gaps and rng.random() < pr.p_gap_minute:
                v = float(rng.choice(pr.gap_values)); c = rng.integers(2, seq_len - 2); w = rng.integers(1, 3)
                rr_g[c:c + w] = v
            ov = np.clip(np.minimum(e[valid], (m + 1) * 60.0) - np.maximum(s[valid], m * 60.0), 0, None).sum()
            seqs.append(np.stack([rr_g, am_g])); lab.append(int(ov >= label_min_overlap_s)); recid.append(b); mins.append(m)
    return np.array(seqs, np.float32), np.array(lab, np.int8), np.array(recid), np.array(mins)


def generate_training_minutes(n_minutes: int, pr: Priors, seed: int, rec_minutes: int = 60, balance: float | None = None):
    """~n_minutes labelled minutes from independent synthetic recordings of `rec_minutes` each. Returns seq, y, recording-id, params."""
    rng = np.random.default_rng(seed); B = int(np.ceil(n_minutes / rec_minutes * 1.02)); T = rec_minutes * 60.0
    br = simulate_batch(rng, B, T, pr)
    seq, y, rid, mn = to_minutes(br, rng, pr, T)
    if len(y) > n_minutes:
        keep = rng.choice(len(y), n_minutes, replace=False); keep.sort(); seq, y, rid, mn = seq[keep], y[keep], rid[keep], mn[keep]
    return seq, y, rid, br.params
