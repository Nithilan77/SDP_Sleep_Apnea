"""Label-preserving augmentations for per-minute (RR s, R-amplitude) x 60 sequences, applied on GPU in RAW units.
Each component is channel-specific and physically motivated (see COMPONENTS).  Applied per sample with probability p.
"""
from __future__ import annotations
import torch

COMPONENTS = {
    # name: (channel, physical meaning)
    "gain":     ("amp", "electrode/gain variation between recordings: R-amplitude x logU(0.8,1.25)"),
    "noise_rr": ("rr",  "R-peak localisation jitter: N(0, 4 ms) on each RR"),
    "noise_amp":("amp", "amplitude measurement noise: N(0, 2% of the minute's mean amplitude)"),
    "shift":    ("both","small time shift of the 60-step grid (+-2 steps, edge-replicated): label stays within the minute"),
    "warp":     ("both","heart-rate-dependent time scaling: grid resampled at speed U(0.92,1.08)"),
    "drift":    ("amp", "baseline/respiratory slow amplitude drift: +-8% sinusoid over the minute"),
    "ectopic":  ("rr",  "isolated premature beat: one RR x0.65 followed by x1.35 (p=0.05 per sample)"),
    "dropout":  ("both","1-3 consecutive steps replaced by linear interpolation (missed beat / bad segment)"),
}
SETS = {"augA": ["gain", "noise_rr", "noise_amp", "shift"],                    # conservative
        "augB": ["gain", "drift", "noise_amp", "noise_rr", "ectopic", "warp", "dropout"]}   # channel-specific
SETS.update({f"aug_{k}": [k] for k in COMPONENTS})


def augment(x: torch.Tensor, comps: list[str], g: torch.Generator, p: float = 0.5) -> torch.Tensor:
    """x (B,2,60) raw units, returns a new tensor."""
    B, _, L = x.shape; dev = x.device; x = x.clone()
    R = lambda *s: torch.rand(*s, device=dev, generator=g)
    N = lambda *s: torch.randn(*s, device=dev, generator=g)
    for c in comps:
        on = (R(B) < p)
        if c == "gain":
            f = torch.exp(torch.log(torch.tensor(0.8, device=dev)) + R(B) * (torch.log(torch.tensor(1.25, device=dev)) - torch.log(torch.tensor(0.8, device=dev))))
            x[:, 1] = torch.where(on[:, None], x[:, 1] * f[:, None], x[:, 1])
        elif c == "noise_rr":
            x[:, 0] = torch.where(on[:, None], x[:, 0] + 0.004 * N(B, L), x[:, 0])
        elif c == "noise_amp":
            x[:, 1] = torch.where(on[:, None], x[:, 1] + 0.02 * x[:, 1].abs().mean(1, keepdim=True) * N(B, L), x[:, 1])
        elif c == "drift":
            u = torch.linspace(0, 1, L, device=dev)[None]; ph = 2 * torch.pi * R(B)[:, None]
            x[:, 1] = torch.where(on[:, None], x[:, 1] * (1 + 0.08 * (2 * R(B)[:, None] - 1) * torch.sin(2 * torch.pi * u * (0.5 + 1.5 * R(B)[:, None]) + ph)), x[:, 1])
        elif c == "ectopic":
            on2 = R(B) < 0.05; pos = (R(B) * (L - 2)).long() + 1; r = x[:, 0].clone(); ar = torch.arange(B, device=dev)
            r[ar, pos] = torch.where(on2, r[ar, pos] * 0.65, r[ar, pos]); r[ar, pos + 1] = torch.where(on2, r[ar, pos + 1] * 1.35, r[ar, pos + 1]); x[:, 0] = r
        elif c == "shift":
            k = ((R(B) * 5).long() - 2) * on.long(); idx = (torch.arange(L, device=dev)[None] - k[:, None]).clamp(0, L - 1)
            x = torch.gather(x, 2, idx[:, None, :].expand(B, 2, L))
        elif c == "warp":
            s = 0.92 + 0.16 * R(B); pos = (torch.arange(L, device=dev)[None].float() * s[:, None] + (1 - s[:, None]) * (L - 1) / 2).clamp(0, L - 1)
            pos = torch.where(on[:, None], pos, torch.arange(L, device=dev)[None].float().expand(B, L))
            lo = pos.floor().long(); hi = (lo + 1).clamp(max=L - 1); w = (pos - lo.float())[:, None]
            x = torch.gather(x, 2, lo[:, None].expand(B, 2, L)) * (1 - w) + torch.gather(x, 2, hi[:, None].expand(B, 2, L)) * w
        elif c == "dropout":
            n = 1 + (R(B) * 3).long(); st = (R(B) * (L - 6)).long() + 1; ar = torch.arange(L, device=dev)[None]
            m = on[:, None] & (ar >= st[:, None]) & (ar < (st + n)[:, None]); a = torch.gather(x, 2, (st - 1)[:, None, None].expand(B, 2, 1)); b = torch.gather(x, 2, (st + n)[:, None, None].expand(B, 2, 1))
            t = ((ar - st[:, None] + 1).float() / (n[:, None] + 1).float())[:, None]
            x = torch.where(m[:, None], a + (b - a) * t, x)
    return x
