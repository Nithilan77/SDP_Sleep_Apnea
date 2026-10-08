"""First labelled in-house event analysis: voluntary breath-hold session.

Chest SensorTile box only (QVAR/ECG + accelerometer). EOG/EMG boxes were
placed elsewhere for unrelated purposes and are not analysed here.

Event timeline derived from stopwatch lap screenshots (15 laps); see
docs/PROGRESS.md Phase 5 section for the full lap table. Stopwatch total
(1800.46s) vs recording duration (1795.6s) differ by ~4.9s, consistent
with the reported 5-10s manual-start sync lag between devices.

This script does NOT trust the stopwatch boundaries as ground truth -- it
uses them as an approximate window, then looks for the expected
bradycardia (during hold) / tachycardia (on release) signature in the
real RR-interval trace to refine actual event boundaries independently.

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

# ── event timeline, derived from stopwatch screenshots (cumulative seconds) ──
# (lap_end_cumulative_s, label) -- lap boundaries in stopwatch time
LAP_CUMULATIVE = {
    1: 991.21, 2: 1020.60, 3: 1110.95, 4: 1140.56, 5: 1231.07,
    6: 1261.61, 7: 1352.50, 8: 1383.34, 9: 1470.63, 10: 1501.41,
    11: 1592.01, 12: 1622.55, 13: 1713.18, 14: 1743.79, 15: 1800.46,
}
HOLD_LAPS = [2, 4, 6, 8, 10, 12, 14]  # even laps = ~30s breath holds (confirmed by user)

PAD_S = 15.0          # window padding each side, to absorb 5-10s sync uncertainty
REFINE_SEARCH_S = 12.0  # how far from nominal boundary to search for the HR signature


def build_events():
    """Returns list of dicts: hold events with nominal [start,end] in stopwatch time."""
    events = []
    for i, lap in enumerate(HOLD_LAPS, start=1):
        start = LAP_CUMULATIVE[lap - 1] if lap > 1 else 0.0
        end = LAP_CUMULATIVE[lap]
        events.append(dict(idx=i, lap=lap, nominal_start=start, nominal_end=end,
                           nominal_duration=end - start))
    return events


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    log.info("Loading chest box recording ...")
    rec = load_sensortile(REC_PATH)
    t_full = (rec.timestamp_us - rec.timestamp_us[0]) / 1e6  # seconds, recording-local time
    log.info("Recording duration: %.1fs (%.2f min)", rec.duration_s, rec.duration_s / 60)
    log.info("Stopwatch total: %.2fs -- diff: %.1fs (expected ~5-10s sync lag)",
             LAP_CUMULATIVE[15], LAP_CUMULATIVE[15] - rec.duration_s)

    # ── R-peak detection across the full session
    log.info("Detecting R-peaks across full session ...")
    rp = detect_rpeaks(rec)
    t_peaks = rp.r_peaks / rec.fs
    rr = np.diff(t_peaks)
    t_rr = t_peaks[1:]  # RR interval ending at this time
    ok = (rr >= RR_MIN_S) & (rr <= RR_MAX_S)
    t_rr_ok, rr_ok = t_rr[ok], rr[ok]
    hr_ok = 60.0 / rr_ok
    log.info("  %d beats detected, %d RR intervals, %.1f%% plausible",
             len(rp.r_peaks), len(rr), 100 * ok.mean())

    # ── effort envelope across the full session
    log.info("Computing effort envelope (accelerometer) across full session ...")
    a_g = rec.accel_mg / 1000.0
    es = E.compute_effort_signal(a_g[:, 0], a_g[:, 1], a_g[:, 2], rec.fs)
    t_effort = np.arange(len(es.bandpassed)) / rec.fs

    events = build_events()
    log.info("\n%d hold events to analyse (laps %s)", len(events), HOLD_LAPS)

    summary_rows = []
    for ev in events:
        idx, start, end = ev["idx"], ev["nominal_start"], ev["nominal_end"]
        win_start, win_end = start - PAD_S, end + PAD_S

        m_rr = (t_rr_ok >= win_start) & (t_rr_ok <= win_end)
        m_eff = (t_effort >= win_start) & (t_effort <= win_end)
        t_rr_w, hr_w = t_rr_ok[m_rr], hr_ok[m_rr]
        t_eff_w, eff_w = t_effort[m_eff], es.bandpassed[m_eff]

        if len(hr_w) < 5:
            log.warning("Event %d: too few RR points in window (%d) -- skipping", idx, len(hr_w))
            continue

        # ---- refine hold START: look for the steepest HR deceleration (onset of
        # bradycardia) within REFINE_SEARCH_S of the nominal start
        search_s = (t_rr_w >= start - REFINE_SEARCH_S) & (t_rr_w <= start + REFINE_SEARCH_S)
        refined_start = start
        if search_s.sum() >= 3:
            t_s, hr_s = t_rr_w[search_s], hr_w[search_s]
            dhr = np.gradient(hr_s, t_s)
            refined_start = float(t_s[np.argmin(dhr)])  # steepest drop

        # ---- refine hold END: look for the steepest HR acceleration (tachycardic
        # rebound) within REFINE_SEARCH_S of the nominal end
        search_e = (t_rr_w >= end - REFINE_SEARCH_S) & (t_rr_w <= end + REFINE_SEARCH_S)
        refined_end = end
        if search_e.sum() >= 3:
            t_e, hr_e = t_rr_w[search_e], hr_w[search_e]
            dhr = np.gradient(hr_e, t_e)
            refined_end = float(t_e[np.argmax(dhr)])  # steepest rise

        # effort amplitude: mean |envelope| during nominal hold vs during the
        # pre-hold baseline (last 15s before nominal start)
        in_hold = (t_eff_w >= start) & (t_eff_w <= end)
        pre_hold = (t_eff_w >= start - PAD_S) & (t_eff_w < start)
        eff_hold_rms = float(np.sqrt(np.mean(eff_w[in_hold] ** 2))) if in_hold.sum() else np.nan
        eff_pre_rms = float(np.sqrt(np.mean(eff_w[pre_hold] ** 2))) if pre_hold.sum() else np.nan
        eff_ratio = eff_hold_rms / eff_pre_rms if eff_pre_rms and eff_pre_rms > 1e-9 else np.nan

        # HR: mean during nominal hold vs pre-hold baseline
        hr_hold = hr_w[(t_rr_w >= start) & (t_rr_w <= end)]
        hr_pre = hr_w[(t_rr_w >= start - PAD_S) & (t_rr_w < start)]
        hr_hold_mean = float(hr_hold.mean()) if len(hr_hold) else np.nan
        hr_pre_mean = float(hr_pre.mean()) if len(hr_pre) else np.nan

        row = dict(event=idx, lap=ev["lap"], nominal_start=start, nominal_end=end,
                  refined_start=refined_start, refined_end=refined_end,
                  start_offset=refined_start - start, end_offset=refined_end - end,
                  hr_pre_mean=hr_pre_mean, hr_hold_mean=hr_hold_mean,
                  hr_delta=hr_hold_mean - hr_pre_mean if not (np.isnan(hr_hold_mean) or np.isnan(hr_pre_mean)) else np.nan,
                  effort_pre_rms=eff_pre_rms, effort_hold_rms=eff_hold_rms, effort_ratio=eff_ratio)
        summary_rows.append(row)
        log.info("Event %d (lap %d): nominal [%.1f,%.1f] | refined start %+.1fs, end %+.1fs | "
                 "HR pre=%.1f hold=%.1f (Δ=%+.1f) | effort ratio=%.3f",
                 idx, ev["lap"], start, end, row["start_offset"], row["end_offset"],
                 hr_pre_mean, hr_hold_mean, row["hr_delta"], eff_ratio)

        # ---- per-event plot
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(t_rr_w, hr_w, "o-", ms=3, color="#E53935")
        axes[0].axvspan(start, end, color="grey", alpha=0.15, label="nominal hold (stopwatch)")
        axes[0].axvline(refined_start, color="#1976D2", ls="--", lw=1.5, label="refined start (HR)")
        axes[0].axvline(refined_end, color="#1976D2", ls="--", lw=1.5, label="refined end (HR)")
        axes[0].set(ylabel="HR (bpm)", title=f"Event {idx} (lap {ev['lap']}) -- nominal duration {ev['nominal_duration']:.1f}s")
        axes[0].legend(fontsize=7, loc="upper right")

        axes[1].plot(t_eff_w, eff_w, color="#43A047", lw=0.8)
        axes[1].axvspan(start, end, color="grey", alpha=0.15)
        axes[1].axvline(refined_start, color="#1976D2", ls="--", lw=1.5)
        axes[1].axvline(refined_end, color="#1976D2", ls="--", lw=1.5)
        axes[1].set(xlabel="recording time (s)", ylabel="effort envelope (g, bandpassed)")

        fig.tight_layout()
        fig.savefig(OUT / f"event_{idx:02d}.png", dpi=120)
        plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "event_summary.csv", index=False)

    # ── full-session overview plot
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    axes[0].plot(t_rr_ok, hr_ok, lw=0.5, color="#E53935")
    for ev in events:
        axes[0].axvspan(ev["nominal_start"], ev["nominal_end"], color="grey", alpha=0.2)
    axes[0].set(ylabel="HR (bpm)", title="Full session: HR trace with nominal hold windows shaded")
    axes[1].plot(t_effort, es.bandpassed, lw=0.3, color="#43A047")
    for ev in events:
        axes[1].axvspan(ev["nominal_start"], ev["nominal_end"], color="grey", alpha=0.2)
    axes[1].set(xlabel="recording time (s)", ylabel="effort envelope")
    fig.tight_layout()
    fig.savefig(OUT / "session_overview.png", dpi=100)
    plt.close(fig)

    # ── honest summary
    log.info("\n=== SUMMARY ===")
    log.info("Mean HR delta (hold - pre-hold): %.2f bpm (SD %.2f)",
             summary.hr_delta.mean(), summary.hr_delta.std())
    log.info("Events with HR INCREASE during hold (unexpected direction): %d / %d",
             (summary.hr_delta > 0).sum(), len(summary))
    log.info("Mean |start offset|: %.1fs, mean |end offset|: %.1fs (sync refinement magnitude)",
             summary.start_offset.abs().mean(), summary.end_offset.abs().mean())
    log.info("Mean effort ratio (hold RMS / pre-hold RMS): %.3f (SD %.3f) -- <1.0 means "
             "effort DROPPED during hold, as expected if cessation is detectable",
             summary.effort_ratio.mean(), summary.effort_ratio.std())
    log.info("Events with effort ratio < 0.8 (clear drop): %d / %d",
             (summary.effort_ratio < 0.8).sum(), len(summary))

    (OUT / "summary.json").write_text(json.dumps(dict(
        n_events=len(summary),
        mean_hr_delta=float(summary.hr_delta.mean()),
        mean_start_offset_s=float(summary.start_offset.mean()),
        mean_end_offset_s=float(summary.end_offset.mean()),
        mean_effort_ratio=float(summary.effort_ratio.mean()),
        n_effort_ratio_below_0_8=int((summary.effort_ratio < 0.8).sum()),
        sync_lag_check_stopwatch_vs_recording_s=float(LAP_CUMULATIVE[15] - rec.duration_s),
    ), indent=2))
    log.info("\nAll outputs in %s", OUT)


if __name__ == "__main__":
    main()
