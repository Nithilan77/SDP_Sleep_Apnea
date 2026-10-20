"""Figures + audit table for the Mithin session-2 reconstruction (reads outputs of s2_timeline.py; read-only on raw data).
    python3 -m src.breath_hold.s2_report"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.breath_hold import s2_timeline as T
from src.ingest.sensortile import load_sensortile
from src.ecg.rpeaks import detect_rpeaks

OUT = T.OUT


def fig_timeline(lt):
    fig, ax = plt.subplots(figsize=(11, 2.6))
    for _, r in lt.iterrows():
        hold = "HOLD" in r.inferred_role
        ax.barh(0, r.end_s_calculated - r.start_s_calculated, left=r.start_s_calculated, color="#C62828" if hold else ("#90A4AE" if r.lap != 1 else "#CFD8DC"), edgecolor="w")
    ax.set_yticks([]); ax.set_xlabel("stopwatch time (s)  [lap boundaries calculated from observed lap durations; hold roles INFERRED]")
    ax.set_title("Mithin session-2 stopwatch timeline (red = inferred holds, lap 3,5,..,15)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig_timeline.png", dpi=130); plt.close(fig)


def fig_scans():
    holds = T.hold_windows(); fig, ax = plt.subplots(3, 2, figsize=(12, 8))
    for k, name in enumerate(T.TILES):
        env, fs, dur, n = T.load_effort_env(T.TILES[name])
        ax[k, 0].plot(env, lw=.5); ax[k, 0].set_yscale("log"); ax[k, 0].set_title(f"{name} tile: accel breathing-band envelope (1 Hz), {dur:.0f} s", fontsize=9, loc="left")
        s = pd.read_csv(OUT / f"scan_{name}.csv")
        ax[k, 1].plot(s.tau_s, s.score_log_env_hold_minus_flank, lw=.6)
        ax[k, 1].set_title(f"{name}: offset scan (lower = quieter at inferred holds); no peak beats the shuffled null", fontsize=9, loc="left")
        ax[k, 1].set_xlabel("offset tau: recording time of stopwatch zero (s)")
    fig.tight_layout(); fig.savefig(OUT / "fig_sync_scans.png", dpi=120); plt.close(fig)


def audit_mithin10():
    rows = []
    for nm in ("ECG", "EOG", "EMG"):
        p = Path.home() / f"Desktop/Mithin/user_Mithin10_{nm}_000.csv"
        if p.stat().st_size == 0:
            rows.append(dict(file=p.name, bytes=0, note="EMPTY file (0 bytes)")); continue
        r = load_sensortile(p); k = detect_rpeaks(r) if nm == "ECG" else None
        rows.append(dict(file=p.name, bytes=p.stat().st_size, rows=r.n_samples, duration_s=round(r.duration_s, 1), fs_hz=round(r.fs_measured, 2),
                         gaps=r.n_gaps, qvar_sd_lsb=round(float(np.nanstd(r.qvar)), 1),
                         beats=(len(k.r_peaks) if k else None), mean_bpm=(round(float(60 / np.median(k.rr_intervals_s)), 1) if k else None),
                         note="no timestamps/annotations exist for this recording; separate from the breath-hold session"))
    pd.DataFrame(rows).to_csv(OUT / "mithin10_audit.csv", index=False); return rows


if __name__ == "__main__":
    lt = pd.read_csv(OUT / "laps_extracted.csv"); fig_timeline(lt); fig_scans(); print(audit_mithin10())
