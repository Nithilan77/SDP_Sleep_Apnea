"""First labelled in-house event analysis: voluntary breath-hold session.

Chest SensorTile box only (QVAR/ECG + accelerometer). EOG/EMG boxes were
placed elsewhere for unrelated purposes and are not analysed here.

v2 (fixes two bugs found by inspecting v1's plots):
  1. HR-based boundary refinement was latching onto isolated R-peak
     detection artifacts (single-beat spikes, e.g. ~150bpm instantaneous
     jumps with no buildup -- confirmed NOT physiological by their
     appearance outside hold windows too, in the full-session trace).
     Fixed: despike the HR series (reject single-sample jumps beyond a
     physiologically implausible rate-of-change) before any refinement
     or reporting uses it.
  2. Effort-ratio computation used NOMINAL (stopwatch) boundaries even
     when refined boundaries were already computed, so a recovery gasp
     that leaks past the nominal boundary (confirmed in event 7: the
     refined end caught the true transition correctly, but the ratio
     calc didn't use it) inflated the ratio. Fixed: effort-envelope
     transition detection is now the PRIMARY refinement method (cleaner
     signal, confirmed by visual inspection), HR is secondary/diagnostic
     only, and all summary statistics use the same refined boundaries
     consistently.

v3 (fixes the boundary-refinement bug that v2 still had):
  v2's refine_from_effort ran on the SIGNED bandpassed trace and called a
  stretch "quiet" after 5 consecutive samples (~20 ms at 243 Hz) with
  |x| < a percentile threshold. |bandpassed| passes through zero twice per
  breath, so every zero-crossing satisfied that test; the first match was
  therefore always the left edge of the search window (all 7 start offsets
  landed at -11.0..-11.9 s of a +/-12 s window). v3 instead:
    * works on the Hilbert ENVELOPE (no zero-crossing dips),
    * takes the quiet threshold as a fraction of the event's own pre-hold
      median envelope (same 30% rule as src/effort/cessation.py),
    * requires a sustained quiet run of ~2 breath periods, where the period
      is measured per event from the pre-hold baseline (not hardcoded),
    * bridges sub-second envelope blips, and
    * REFUSES to return a boundary that sits on the search-window edge or
      has no valid run: the event falls back to nominal boundaries and is
      flagged in event_summary.csv (refinement_fell_back_to_nominal).
  Also computes the fixed-pad alternative (nominal window inset by the
  stopwatch-vs-recording sync lag) as an independent sensitivity check.

    python -m src.breath_hold.analyze_session
Outputs -> results/breath_hold/
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingest.sensortile import load_sensortile
from src.ecg.rpeaks import detect_rpeaks, RR_MIN_S, RR_MAX_S
from src.effort import envelope as E

log = logging.getLogger("breath_hold")
REC_PATH = Path("data/recordings/breath_hold/user_MukuBreathHold_ECG_07_10_2026.csv")
OUT = Path("results/breath_hold")

LAP_CUMULATIVE = {
    1: 991.21, 2: 1020.60, 3: 1110.95, 4: 1140.56, 5: 1231.07,
    6: 1261.61, 7: 1352.50, 8: 1383.34, 9: 1470.63, 10: 1501.41,
    11: 1592.01, 12: 1622.55, 13: 1713.18, 14: 1743.79, 15: 1800.46,
}
HOLD_LAPS = [2, 4, 6, 8, 10, 12, 14]

PAD_S = 15.0
REFINE_SEARCH_S = 12.0
MAX_HR_JUMP_BPM = 40.0   # reject a beat-to-beat HR jump bigger than this as a likely artifact

# --- effort-envelope boundary refinement (v3)
BASELINE_FAR_S = 30.0       # baseline = [nominal_start-30, nominal_start-REFINE_SEARCH_S]: ends before the search window
QUIET_FRAC = 0.30           # quiet = envelope within 30% of the way from the event's noise floor up to its
                            # pre-hold median envelope (floor-relative: the hold-time envelope sits on a
                            # sensor/heartbeat floor above zero, so a fraction of baseline alone is too strict)
MIN_RUN_PERIODS = 2.0       # a quiet run must last >= 2 breath periods (measured per event)
MIN_RUN_CLIP_S = (4.0, 10.0)  # sanity clip on the above, in case the period estimate is off
SEG_MIN_S = 1.0             # a below-threshold segment must last this long to count as part of a pause
                            # (ordinary exhale dips last well under a second)
MAX_GAP_PERIODS = 1.0       # activity lasting <= 1 breath period (one swallow/twitch/breath) does not end a pause
MIN_QUIET_FRAC = 0.8        # a pause, gaps included, must be >= 80% below threshold
EDGE_TOL_S = 0.5            # a boundary within this of the search-window limit = detector hit its bound
BREATH_BAND_HZ = (0.1, 0.6)


# MANUAL visual verdict on the detector's output, from reading all 7 event_XX.png
# plots after the third (final) approach. Not computed -- a human judgment recorded
# so the CSV is self-describing. See results/breath_hold/FINDINGS.md.
VISUAL_VERDICT = {
    1: "FAIL: fell back to nominal; visible quiet ~997-1015s not found",
    2: "OK: boundaries on visible quiet stretch",
    3: "FAIL: truncated at ~1240.6 by a ~3.5s mid-hold burst; quiet continues to ~1257",
    4: "OK: boundaries on visible quiet stretch (quiet starts ~6s before nominal)",
    5: "FAIL: no clean quiet stretch; detector latched a 6s fragment",
    6: "OK: boundaries on visible quiet stretch",
    7: "FAIL: start ~13s late (quiet begins ~1710.5, detector says 1724.6); end OK",
}


def build_events():
    events = []
    for i, lap in enumerate(HOLD_LAPS, start=1):
        start = LAP_CUMULATIVE[lap - 1] if lap > 1 else 0.0
        end = LAP_CUMULATIVE[lap]
        events.append(dict(idx=i, lap=lap, nominal_start=start, nominal_end=end,
                           nominal_duration=end - start))
    return events


def despike_hr(t_rr, hr, max_jump=MAX_HR_JUMP_BPM):
    """Reject isolated single-beat HR spikes: a beat whose HR differs from
    BOTH neighbours by more than max_jump is flagged as an artifact and
    dropped (not interpolated -- just removed, so it can't contaminate any
    downstream gradient or mean calculation)."""
    if len(hr) < 3:
        return t_rr, hr, np.zeros(len(hr), dtype=bool)
    d_prev = np.abs(np.diff(hr, prepend=hr[0]))
    d_next = np.abs(np.diff(hr, append=hr[-1]))
    is_spike = (d_prev > max_jump) & (d_next > max_jump)
    return t_rr[~is_spike], hr[~is_spike], is_spike


def _runs(mask):
    """(start_idx, end_idx_exclusive) of each True run in a boolean array."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def baseline_breath_period(t, bp, lo, hi, fs):
    """Dominant breathing period (s) in bp over [lo, hi): spectral peak in BREATH_BAND_HZ."""
    seg = bp[(t >= lo) & (t < hi)]
    if len(seg) < int(8 * fs):
        return np.nan
    seg = (seg - seg.mean()) * np.hanning(len(seg))
    n = 1 << int(np.ceil(np.log2(len(seg) * 8)))
    spec = np.abs(np.fft.rfft(seg, n))
    f = np.fft.rfftfreq(n, 1 / fs)
    band = (f >= BREATH_BAND_HZ[0]) & (f <= BREATH_BAND_HZ[1])
    return float(1.0 / f[band][np.argmax(spec[band])])


def refine_from_effort(t, env, bp, fs, nominal_start, nominal_end, search_s=REFINE_SEARCH_S):
    """Find the sustained quiet stretch of the Hilbert envelope near the nominal hold.

    Returns a dict. If no valid sustained quiet run is found, or a boundary lands
    on the search-window limit (the detector hit its bound rather than finding
    something), `fell_back` is True and the boundaries are the nominal ones.
    """
    out = dict(refined_start=nominal_start, refined_end=nominal_end, fell_back=True,
               reason="", quiet_thresh=np.nan, breath_period_s=np.nan, min_run_s=np.nan)
    base_lo, base_hi = nominal_start - BASELINE_FAR_S, nominal_start - search_s
    base = (t >= base_lo) & (t < base_hi)
    if base.sum() < int(8 * fs):
        out["reason"] = "baseline too short"; return out
    base_med = float(np.median(env[base]))
    win_ = (t >= nominal_start - search_s) & (t <= nominal_end + search_s)
    floor = float(env[win_].min())          # noise floor from this event's own window
    thr = floor + QUIET_FRAC * (base_med - floor)
    period = baseline_breath_period(t, bp, base_lo, base_hi, fs)
    if not np.isfinite(period):
        out["reason"] = "no breathing period in baseline"; return out
    min_run = float(np.clip(MIN_RUN_PERIODS * period, *MIN_RUN_CLIP_S))
    out.update(quiet_thresh=thr, breath_period_s=period, min_run_s=min_run)

    lo, hi = nominal_start - search_s, nominal_end + search_s
    w = (t >= lo) & (t <= hi)
    tw, ew = t[w], env[w]
    quiet = ew < thr
    segs = [(i0, i1) for i0, i1 in _runs(quiet) if (i1 - i0) / fs >= SEG_MIN_S]
    # chain substantial quiet segments separated by <= 1 breath period of activity
    chains = []
    for sg in segs:
        if chains and (sg[0] - chains[-1][1]) / fs <= MAX_GAP_PERIODS * period:
            chains[-1] = (chains[-1][0], sg[1])
        else:
            chains.append(sg)

    for i0, i1 in chains:
        if (i1 - i0) / fs < min_run or quiet[i0:i1].mean() < MIN_QUIET_FRAC:
            continue
        s_t, e_t = float(tw[i0]), float(tw[i1 - 1])
        if s_t > nominal_start + search_s:
            break
        if i0 == 0 or s_t <= lo + EDGE_TOL_S:
            out["reason"] = "quiet run already under way at search-window start (start bound hit)"; return out
        if i1 == len(tw) or e_t >= hi - EDGE_TOL_S:
            out["reason"] = "quiet run extends to search-window end (end bound hit)"; return out
        out.update(refined_start=s_t, refined_end=e_t, fell_back=False, reason="")
        return out
    out["reason"] = f"no quiet run >= {min_run:.1f}s below {thr:.5f}"
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    log.info("Loading chest box recording ...")
    rec = load_sensortile(REC_PATH)
    log.info("Recording duration: %.1fs (%.2f min)", rec.duration_s, rec.duration_s / 60)
    log.info("Stopwatch total: %.2fs -- diff: %.1fs (expected ~5-10s sync lag)",
             LAP_CUMULATIVE[15], LAP_CUMULATIVE[15] - rec.duration_s)

    log.info("Detecting R-peaks across full session ...")
    rp = detect_rpeaks(rec)
    t_peaks = rp.r_peaks / rec.fs
    rr = np.diff(t_peaks)
    t_rr = t_peaks[1:]
    ok = (rr >= RR_MIN_S) & (rr <= RR_MAX_S)
    t_rr_ok, rr_ok = t_rr[ok], rr[ok]
    hr_raw = 60.0 / rr_ok
    log.info("  %d beats detected, %d RR intervals, %.1f%% plausible (range filter)",
             len(rp.r_peaks), len(rr), 100 * ok.mean())

    t_rr_clean, hr_clean, is_spike = despike_hr(t_rr_ok, hr_raw)
    log.info("  Despiking: %d / %d beats flagged as isolated artifacts and removed (%.2f%%)",
             is_spike.sum(), len(hr_raw), 100 * is_spike.mean())

    log.info("Computing effort envelope (accelerometer) across full session ...")
    a_g = rec.accel_mg / 1000.0
    es = E.compute_effort_signal(a_g[:, 0], a_g[:, 1], a_g[:, 2], rec.fs)
    t_effort = np.arange(len(es.bandpassed)) / rec.fs

    sync_lag = float(LAP_CUMULATIVE[15] - rec.duration_s)   # stopwatch-vs-recording total difference
    events = build_events()
    log.info("\n%d hold events to analyse (laps %s)", len(events), HOLD_LAPS)

    summary_rows = []
    for ev in events:
        idx, start, end = ev["idx"], ev["nominal_start"], ev["nominal_end"]
        win_start, win_end = start - BASELINE_FAR_S, end + PAD_S

        m_rr = (t_rr_clean >= win_start) & (t_rr_clean <= win_end)
        m_eff = (t_effort >= win_start) & (t_effort <= win_end)
        t_rr_w, hr_w = t_rr_clean[m_rr], hr_clean[m_rr]
        t_eff_w, eff_w, env_w = t_effort[m_eff], es.bandpassed[m_eff], es.envelope[m_eff]

        # ---- (a) PRIMARY: sustained-quiet-run refinement on the Hilbert envelope
        r = refine_from_effort(t_effort, es.envelope, es.bandpassed, rec.fs, start, end)
        refined_start, refined_end = r["refined_start"], r["refined_end"]

        def effort_ratio(b0, b1):
            hold = (t_eff_w >= b0) & (t_eff_w <= b1)
            pre = (t_eff_w >= b0 - PAD_S) & (t_eff_w < b0)
            h = float(np.sqrt(np.mean(eff_w[hold] ** 2))) if hold.sum() else np.nan
            p = float(np.sqrt(np.mean(eff_w[pre] ** 2))) if pre.sum() else np.nan
            return p, h, (h / p if p and p > 1e-9 else np.nan)

        eff_pre_rms, eff_hold_rms, eff_ratio = effort_ratio(refined_start, refined_end)

        # ---- (b) SENSITIVITY: nominal window inset by the sync lag (no detection at all).
        # Pre-hold baseline ends at nominal_start - lag so it cannot contain hold time.
        b_start, b_end = start + sync_lag, end - sync_lag
        _, eff_hold_rms_b, _ = effort_ratio(b_start, b_end)
        pre_b = (t_eff_w >= start - sync_lag - PAD_S) & (t_eff_w < start - sync_lag)
        eff_pre_rms_b = float(np.sqrt(np.mean(eff_w[pre_b] ** 2))) if pre_b.sum() else np.nan
        eff_ratio_b = eff_hold_rms_b / eff_pre_rms_b if eff_pre_rms_b > 1e-9 else np.nan

        # ---- (c) WIDE: the nominal stopwatch window itself (includes transition/gasp bleed)
        _, _, eff_ratio_w = effort_ratio(start, end)

        # ---- HR reported as diagnostic only, using despiked data + refined boundaries
        hr_hold = hr_w[(t_rr_w >= refined_start) & (t_rr_w <= refined_end)]
        hr_pre = hr_w[(t_rr_w >= refined_start - PAD_S) & (t_rr_w < refined_start)]
        hr_hold_mean = float(hr_hold.mean()) if len(hr_hold) else np.nan
        hr_pre_mean = float(hr_pre.mean()) if len(hr_pre) else np.nan
        hr_delta = hr_hold_mean - hr_pre_mean if not (np.isnan(hr_hold_mean) or np.isnan(hr_pre_mean)) else np.nan

        row = dict(event=idx, lap=ev["lap"], nominal_start=start, nominal_end=end,
                  refinement_fell_back_to_nominal=bool(r["fell_back"]), fallback_reason=r["reason"],
                  refined_start=refined_start, refined_end=refined_end,
                  refined_duration=refined_end - refined_start,
                  start_offset=refined_start - start, end_offset=refined_end - end,
                  baseline_breath_period_s=r["breath_period_s"], min_quiet_run_s=r["min_run_s"],
                  quiet_thresh=r["quiet_thresh"],
                  hr_pre_mean=hr_pre_mean, hr_hold_mean=hr_hold_mean, hr_delta=hr_delta,
                  effort_pre_rms=eff_pre_rms, effort_hold_rms=eff_hold_rms, effort_ratio=eff_ratio,
                  effort_ratio_fixedpad=eff_ratio_b, fixedpad_start=b_start, fixedpad_end=b_end,
                  effort_ratio_nominal=eff_ratio_w,
                  effort_ratio_range_lo=min(eff_ratio_b, eff_ratio_w), effort_ratio_range_hi=max(eff_ratio_b, eff_ratio_w),
                  detector_visual_verdict=VISUAL_VERDICT[idx],
                  n_hr_beats_in_window=int(m_rr.sum()))
        summary_rows.append(row)
        log.info("Event %d (lap %d): nominal [%.1f,%.1f] (%.1fs) | %s [%.1f,%.1f] (%.1fs, offsets %+.1f/%+.1f) | "
                 "breath period %.1fs, min run %.1fs | HR pre=%.1f hold=%.1f (d=%+.1f) | "
                 "effort ratio=%.3f (fixed-pad %.3f)",
                 idx, ev["lap"], start, end, end - start,
                 "FALLBACK(nominal)" if r["fell_back"] else "refined",
                 refined_start, refined_end, refined_end - refined_start,
                 row["start_offset"], row["end_offset"], r["breath_period_s"], r["min_run_s"],
                 hr_pre_mean, hr_hold_mean, hr_delta, eff_ratio, eff_ratio_b)
        if r["fell_back"]:
            log.info("    fallback reason: %s", r["reason"])

        fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
        axes[0].plot(t_rr_w, hr_w, "o-", ms=3, color="#E53935")
        axes[0].axvspan(start, end, color="grey", alpha=0.15, label="nominal hold (stopwatch)")
        axes[0].axvline(refined_start, color="#1976D2", ls="--", lw=1.5,
                        label="boundary (nominal FALLBACK)" if r["fell_back"] else "refined boundary (effort)")
        axes[0].axvline(refined_end, color="#1976D2", ls="--", lw=1.5)
        axes[0].set(ylabel="HR (bpm, despiked)", title=f"Event {idx} (lap {ev['lap']}) -- nominal {end-start:.1f}s, "
                    f"{'FALLBACK to nominal' if r['fell_back'] else 'refined'} {refined_end-refined_start:.1f}s")
        axes[0].legend(fontsize=7, loc="upper right")

        axes[1].plot(t_eff_w, eff_w, color="#43A047", lw=0.8, label="bandpassed accel (g)")
        axes[1].plot(t_eff_w, env_w, color="#EF6C00", lw=1.0, label="Hilbert envelope")
        axes[1].axvspan(start, end, color="grey", alpha=0.15)
        axes[1].axvspan(b_start, b_end, ymin=0, ymax=0.06, color="#8E24AA", alpha=0.6, label="fixed-pad (b) window")
        axes[1].axvline(refined_start, color="#1976D2", ls="--", lw=1.5)
        axes[1].axvline(refined_end, color="#1976D2", ls="--", lw=1.5)
        if np.isfinite(r["quiet_thresh"]):
            axes[1].axhline(r["quiet_thresh"], color="#9E9E9E", ls=":", lw=1.2,
                            label=f"quiet thresh = {QUIET_FRAC:.0%} of pre-hold median env ({r['quiet_thresh']:.5f})")
        axes[1].set(xlabel="recording time (s)", ylabel="effort (g)")
        axes[1].legend(fontsize=7, loc="upper right")

        fig.tight_layout()
        fig.savefig(OUT / f"event_{idx:02d}.png", dpi=120)
        plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "event_summary.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    axes[0].plot(t_rr_clean, hr_clean, lw=0.5, color="#E53935")
    axes[0].scatter(t_rr_ok[is_spike], hr_raw[is_spike], color="black", s=15, marker="x", label=f"despiked artifacts (n={is_spike.sum()})", zorder=5)
    for ev in events:
        axes[0].axvspan(ev["nominal_start"], ev["nominal_end"], color="grey", alpha=0.2)
    axes[0].set(ylabel="HR (bpm, despiked)", title="Full session: HR trace (despiked) with nominal hold windows shaded; x = removed artifacts")
    axes[0].legend(fontsize=8)
    axes[1].plot(t_effort, es.bandpassed, lw=0.3, color="#43A047")
    for ev in events:
        axes[1].axvspan(ev["nominal_start"], ev["nominal_end"], color="grey", alpha=0.2)
    axes[1].set(xlabel="recording time (s)", ylabel="effort envelope")
    fig.tight_layout()
    fig.savefig(OUT / "session_overview.png", dpi=100)
    plt.close(fig)

    log.info("\n=== SUMMARY (v3: envelope-based sustained-quiet refinement + fixed-pad sensitivity) ===")
    log.info("Mean HR delta (hold - pre-hold, despiked): %.2f bpm (SD %.2f)",
             summary.hr_delta.mean(), summary.hr_delta.std())
    log.info("Events with HR increase during hold: %d / %d", (summary.hr_delta > 0).sum(), len(summary))
    log.info("Fell back to nominal: %d / %d events", summary.refinement_fell_back_to_nominal.sum(), len(summary))
    ref = summary[~summary.refinement_fell_back_to_nominal]
    log.info("Refined events: start-offset mean %.1fs SD %.1fs (range %.1f..%.1f); end-offset mean %.1fs SD %.1fs (range %.1f..%.1f)",
             ref.start_offset.mean(), ref.start_offset.std(), ref.start_offset.min(), ref.start_offset.max(),
             ref.end_offset.mean(), ref.end_offset.std(), ref.end_offset.min(), ref.end_offset.max())
    log.info("Mean refined duration: %.1fs (SD %.1f) vs nominal ~30s", summary.refined_duration.mean(), summary.refined_duration.std())
    log.info("Effort ratio (a, refined):   mean %.3f (SD %.3f), median %.3f; <0.8: %d/%d; <0.5: %d/%d",
             summary.effort_ratio.mean(), summary.effort_ratio.std(), summary.effort_ratio.median(),
             (summary.effort_ratio < 0.8).sum(), len(summary), (summary.effort_ratio < 0.5).sum(), len(summary))
    log.info("Effort ratio (b, fixed-pad): mean %.3f (SD %.3f), median %.3f; <0.8: %d/%d; <0.5: %d/%d",
             summary.effort_ratio_fixedpad.mean(), summary.effort_ratio_fixedpad.std(), summary.effort_ratio_fixedpad.median(),
             (summary.effort_ratio_fixedpad < 0.8).sum(), len(summary), (summary.effort_ratio_fixedpad < 0.5).sum(), len(summary))
    log.info("Effort ratio (c, nominal/wide): mean %.3f (SD %.3f), median %.3f; <0.8: %d/%d; <0.5: %d/%d",
             summary.effort_ratio_nominal.mean(), summary.effort_ratio_nominal.std(), summary.effort_ratio_nominal.median(),
             (summary.effort_ratio_nominal < 0.8).sum(), len(summary), (summary.effort_ratio_nominal < 0.5).sum(), len(summary))
    log.info("Per-event range [inner(b) .. wide(c)]: %s", [f"{a:.2f}..{b:.2f}" for a, b in zip(summary.effort_ratio_range_lo, summary.effort_ratio_range_hi)])
    log.info("Per-event (a) vs (b) ratio: %s", [f"{a:.2f}/{b:.2f}" for a, b in zip(summary.effort_ratio, summary.effort_ratio_fixedpad)])

    (OUT / "summary.json").write_text(json.dumps(dict(
        n_events=len(summary),
        n_hr_artifacts_despiked=int(is_spike.sum()),
        mean_hr_delta=float(summary.hr_delta.mean()),
        mean_refined_duration_s=float(summary.refined_duration.mean()),
        mean_start_offset_s=float(summary.start_offset.mean()),
        mean_end_offset_s=float(summary.end_offset.mean()),
        n_refinement_fell_back=int(summary.refinement_fell_back_to_nominal.sum()),
        sd_start_offset_s=float(summary.start_offset.std()),
        mean_effort_ratio=float(summary.effort_ratio.mean()),
        mean_effort_ratio_fixedpad=float(summary.effort_ratio_fixedpad.mean()),
        mean_effort_ratio_nominal=float(summary.effort_ratio_nominal.mean()),
        n_detector_visually_ok=3,
        n_effort_ratio_below_0_8=int((summary.effort_ratio < 0.8).sum()),
        n_effort_ratio_below_0_5=int((summary.effort_ratio < 0.5).sum()),
        sync_lag_check_stopwatch_vs_recording_s=float(LAP_CUMULATIVE[15] - rec.duration_s),
    ), indent=2))
    log.info("\nAll outputs in %s", OUT)


if __name__ == "__main__":
    main()
