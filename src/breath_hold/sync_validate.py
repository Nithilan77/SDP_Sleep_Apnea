"""Validation gate for src/breath_hold/sync.py: synthetic-injection test of the FINAL detection logic on the real session-1 noise.

Session 1 contains no deliberate taps, so (a) false-alarm behaviour is measured on the unmodified recording and (b) detection rate and
timing error are measured by injecting synthetic double-taps (40-90 Hz ring-down, tau 6 ms, random 3-D direction and sub-sample phase)
into quiet sites of the real recording. The injection proves what tap STRENGTH the detector needs; it does not measure real tap strength
(the mandatory pre-session tap check does that).

Uses the SAME functions that run on session 2 (sync.find_pairs / select_marker / tap_check logic). High-pass filtering is linear, so
the injected waveform is high-passed locally and added to the baseline high-passed signal (verified against the direct path below).

    python3 -m src.breath_hold.sync_validate
Outputs: results/breath_hold/sync_validation/{detection_table.csv, validation.md}  (aggregates only; no recordings).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.breath_hold import sync as S  # noqa: E402
from src.ingest.sensortile import load_sensortile  # noqa: E402

REC = Path("data/recordings/breath_hold/user_MukuBreathHold_ECG_07_10_2026.csv")
OUT = Path("results/breath_hold/sync_validation")
AMPS = (0.02, 0.04, 0.08, 0.12, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.2)
N_TRIALS = 200
RNG = np.random.default_rng(0)


def waveform(amp, fs, rng):
    kk = np.arange(8) / fs + rng.uniform(0, 1) / fs
    return amp * np.exp(-kk / 0.006) * np.sin(2 * np.pi * rng.uniform(40, 90) * kk)


class Baseline:
    def __init__(self):
        self.rec = load_sensortile(REC)
        self.a = self.rec.accel_mg / 1000.0
        self.t = (self.rec.timestamp_us - self.rec.timestamp_us[0]) / 1e6
        self.fs = self.rec.fs
        self.hp = sosfiltfilt(S._sos(self.fs), self.a - self.a.mean(0), axis=0)          # baseline high-passed (3-axis)
        m = np.sqrt((self.hp ** 2).sum(1)); self.z0, self.med, self.sig = S.robust_z(m)

    def z_with(self, injections):
        """injections: list of (sample_index, waveform(8,), direction(3,)). Returns the z array and each injection's TRUE tap-peak sample
        = the largest-magnitude sample of the UNFILTERED injected waveform (independent of the detector's own filtered peak)."""
        hp = self.hp.copy(); peaks = []
        for i0, w, ax in injections:
            seg = np.zeros((int(10 * self.fs), 3)); c = int(5 * self.fs)
            for j, v in enumerate(w):
                seg[c + j] += v * ax
            f = sosfiltfilt(S._sos(self.fs), seg, axis=0)
            lo = i0 - c; hp[lo:lo + len(seg)] += f
            peaks.append(i0 + int(np.argmax(np.abs(w))))
        m = np.sqrt((hp ** 2).sum(1))
        return (m - self.med) / self.sig, peaks


def run():
    B = Baseline(); t, fs = B.t, B.fs
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# Sync-tap detector validation (final logic of `src/breath_hold/sync.py`)\n",
          f"Session-1 noise: robust sigma {B.sig:.4f} g (z = 25 -> {25*B.sig:.3f} g; z = 60 -> {60*B.sig:.3f} g; z = 100 -> {100*B.sig:.3f} g); fs {fs:.2f} Hz; {t[-1]:.0f} s.\n"]
    # --- linear-shortcut check vs the direct production path
    direct, _, _, _ = S.analyse(B.a, t, fs)
    md.append(f"Shortcut check: baseline z via high-pass of the stored 3-axis array vs the production path `sync.analyse`: max |dz| = {np.abs(direct - B.z0).max():.2e}.\n")
    # --- false alarms on the unmodified recording
    pairs0 = S.find_pairs(B.z0, t, fs)
    md.append("## False alarms on the real recording (no deliberate taps)\n")
    md.append(f"Candidate pairs (two sharp peaks, z>=25, 0.25-1.2 s apart): {len(pairs0)}. **Accepted markers anywhere: {int(pairs0.accepted.sum()) if len(pairs0) else 0}**; "
              f"start window (first 60 s): {'found' if S.select_marker(pairs0, t, 'start') is not None else 'none'}; end window (last 120 s): {'found' if S.select_marker(pairs0, t, 'end') is not None else 'none'}.\n")
    if len(pairs0):
        md.append("| t1 (s) | z1 / z2 | gap (s) | dominance | accepted | reason(s) rejected |\n|---|---|---|---|---|---|")
        for _, r in pairs0.iterrows():
            md.append(f"| {r.t1:.1f} | {r.z1:.0f} / {r.z2:.0f} | {r.gap_s:.2f} | {r.dom_ratio:.2f}x | {r.accepted} | {r.reasons or '-'} |")
    # --- injection: start window (taps at 10-30 s) and end window (last 110..10 s)
    def quiet(t0, lo=1.0, hi=3.0):
        a, b = np.searchsorted(t, [t0 - lo, t0 + hi]); return B.z0[a:b].max() < 20
    rows = []
    for which, rng_t in (("start", (10.0, 30.0)), ("end", (t[-1] - 110.0, t[-1] - 10.0))):
        sites = [x for x in np.arange(rng_t[0], rng_t[1], 0.5) if quiet(x)]
        for amp in AMPS:
            acc = cand = 0; e_pk, e_on, zmin = [], [], []
            for _ in range(N_TRIALS):
                s0 = RNG.choice(sites); gap = RNG.uniform(0.35, 0.8); ax = RNG.normal(size=3); ax /= np.linalg.norm(ax)
                inj = []
                for ts, am in ((s0, amp), (s0 + gap, amp * RNG.uniform(0.7, 1.3))):
                    inj.append((int(np.searchsorted(t, ts)), waveform(am, fs, RNG), ax))
                z, pk_true = B.z_with(inj)
                pairs = S.find_pairs(z, t, fs)
                near = pairs[(abs(pairs.t1 - t[pk_true[0]]) < 0.1)] if len(pairs) else pairs
                if len(near): cand += 1
                m = S.select_marker(pairs, t, which)
                if m is not None and abs(m.t2 - t[pk_true[1]]) < 0.1:
                    acc += 1; e_pk.append((m.t2 - t[pk_true[1]]) * 1000); e_on.append((m.t2 - t[inj[1][0]]) * 1000)
                    zmin.append(min(m.z1, m.z2))
            rows.append(dict(window=which, amp_g_nominal=amp, candidate_pair_rate=cand / N_TRIALS, accepted_marker_rate=acc / N_TRIALS,
                             median_weaker_tap_z=np.median(zmin) if zmin else np.nan,
                             timing_err_ms_median_abs=np.median(np.abs(e_pk)) if e_pk else np.nan, timing_err_ms_p95_abs=np.percentile(np.abs(e_pk), 95) if e_pk else np.nan,
                             timing_bias_ms_mean=np.mean(e_pk) if e_pk else np.nan, timing_vs_inject_onset_ms_median=np.median(e_on) if e_on else np.nan, n_trials=N_TRIALS))
    D = pd.DataFrame(rows); D.to_csv(OUT / "detection_table.csv", index=False)
    md.append("\n## Detection of injected double-taps (final logic; 200 trials per cell; quiet sites in the search windows)\n")
    md.append("accepted = passes ALL rules incl. >=60 sigma and >=2x dominance; candidate = a sharp pair was seen but may have been rejected (e.g. too weak). "
              "Timing error = detected second-tap time minus the TRUE peak sample of the injected tap (largest-magnitude sample of the unfiltered injected waveform; independent of the detector's filtered peak; one sample = 4.1 ms): median and 95th percentile of |error|, and mean signed bias. The last column compares to the injection START (onset), a looser reference.\n")
    md.append("| window | nominal peak (g) | candidate pair seen | **accepted marker** | median weaker-tap z (accepted) | |timing err| median (ms) | p95 (ms) | mean bias (ms) | median vs inject-onset (ms) |\n|---|---|---|---|---|---|---|---|---|")
    for _, r in D.iterrows():
        md.append(f"| {r.window} | {r.amp_g_nominal:.2f} | {100*r.candidate_pair_rate:.0f}% | **{100*r.accepted_marker_rate:.0f}%** | {r.median_weaker_tap_z:.0f} | {r.timing_err_ms_median_abs:.1f} | {r.timing_err_ms_p95_abs:.1f} | {r.timing_bias_ms_mean:+.1f} | {r.timing_vs_inject_onset_ms_median:.1f} |")
    # --- tap-check mode: 3 trial double-taps 20-30 s apart in the middle of the recording
    md.append("\n## Pre-session tap check (3 trial double-taps >= 20 s apart; PASS needs 3 accepted)\n")
    md.append("| nominal peak (g) | check PASS rate (100 trials) |\n|---|---|")
    mids = [x for x in np.arange(120, t[-1] - 200, 1.0) if quiet(x)]
    for amp in AMPS:
        ok = 0
        for _ in range(100):
            starts = []
            while len(starts) < 3:
                c0 = RNG.choice(mids)
                if all(abs(c0 - s) > 22 for s in starts): starts.append(c0)
            ax = RNG.normal(size=3); ax /= np.linalg.norm(ax); inj = []
            for s0 in starts:
                gap = RNG.uniform(0.35, 0.8)
                for ts, am in ((s0, amp), (s0 + gap, amp * RNG.uniform(0.7, 1.3))):
                    inj.append((int(np.searchsorted(t, ts)), waveform(am, fs, RNG), ax))
            z, _ = B.z_with(inj); pairs = S.find_pairs(z, t, fs)
            acc = pairs[pairs.accepted].sort_values("t1"); kept, last = 0, -1e9
            for _, r in acc.iterrows():
                if r.t1 - last >= S.CHECK_MIN_SPACING_S: kept += 1; last = r.t2
            ok += kept >= S.CHECK_MIN_TRIALS
        md.append(f"| {amp:.2f} | {ok}% |")
    # --- stress: tap near the real transient at ~46.8 s (dominance rule)
    md.append("\n## Stress: taps injected 6 s before the real transient at ~46.8 s (z = 64) inside the start window\n")
    md.append("| nominal peak (g) | accepted marker (200 trials) | note |\n|---|---|---|")
    for amp in (0.2, 0.4, 0.8):
        acc = 0
        for _ in range(200):
            s0 = 40.5 + RNG.uniform(-1, 1); gap = RNG.uniform(0.35, 0.8); ax = RNG.normal(size=3); ax /= np.linalg.norm(ax)
            inj = [(int(np.searchsorted(t, s0)), waveform(amp, fs, RNG), ax), (int(np.searchsorted(t, s0 + gap)), waveform(amp * RNG.uniform(0.7, 1.3), fs, RNG), ax)]
            z, pk_true = B.z_with(inj); m = S.select_marker(S.find_pairs(z, t, fs), t, "start")
            acc += (m is not None and abs(m.t2 - t[pk_true[1]]) < 0.1)
        md.append(f"| {amp:.2f} | {100*acc/200:.0f}% | accepted only if both taps are >= 60 sigma and >= 2x the real transient |")
    # --- lap alignment self-test (synthetic laps with known offset and drift)
    md.append("\n## Lap-alignment self-test (synthetic laps, known truth)\n")
    sched = [0.0]; P = 240.0; holds = []
    for d in S.ORDER_A:
        holds.append((P, P + d)); P += d + max(90, 3 * d)
    t_end = holds[-1][1] + 150.0
    def make(offset, drift_scale):
        true = {1: offset}
        for k, (a, b) in enumerate(holds):
            true[2 + 2 * k] = offset + a * drift_scale; true[3 + 2 * k] = offset + b * drift_scale
        true[20] = offset + t_end * drift_scale
        return true
    md.append("| case | start tap (s) | drift added (s over 25 min) | recovered max abs error of all 20 lap times (ms) | anchor rule applied |\n|---|---|---|---|---|")
    for off, ds in ((17.3, 1.0), (17.3, 1 + 0.2 / t_end), (17.3, 1 + 2.0 / t_end), (8.9, 1 - 3.0 / t_end)):
        true = make(off, ds); sw0 = 3.7      # stopwatch total at lap 1; stopwatch runs at the true rate (drift lives in the device clock)
        laps = pd.DataFrame(dict(lap=list(range(1, 21)), total_s=[sw0 + (true[l] - off) / ds for l in range(1, 21)]))
        t_lap, info = S.lap_times_to_recording(laps, true[1], true[20])
        err = max(abs(t_lap[l] - true[l]) for l in range(1, 21)) * 1000
        md.append(f"| offset {off}, device clock x{ds:.5f} | {off} | {(ds-1)*t_end:+.2f} | {err:.2f} | {info['anchor']} |")
    (OUT / "validation.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    run()
