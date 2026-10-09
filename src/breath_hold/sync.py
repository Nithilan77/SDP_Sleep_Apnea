"""Session-2 sync-tap detection, pre-session tap check, and lap -> recording-time alignment.

Protocol: docs/breath_hold_protocol.md (v2). Everything here is pre-specified there; thresholds are NOT to be tuned on
session-2 data. Validation gate: src/breath_hold/sync_validate.py (injection test on the FINAL logic below).

Tap detection runs on a high-frequency path. The effort pipeline's 0.08-0.6 Hz band removes a ~10 ms tap completely, so:
    accelerometer (g) -> subtract mean -> 4th-order Butterworth high-pass 10 Hz (zero phase) -> vector magnitude
    -> robust z (median / 1.4826*MAD over the whole recording).
A SYNC MARKER is a pair of taps that is
    sharp      : each peak z >= 25, FWHM <= 60 ms
    paired     : 0.25-1.2 s apart, amplitude ratio <= 3x
    isolated   : no other z >= 25 peak within 0.6 s either side of the pair
    strong     : both taps z >= 60                                       (acceptance gate)
    dominant   : both taps >= 2x the largest other transient within +/-10 s (acceptance gate)
Start marker: searched in the first 60 s of the recording. End marker: last 120 s. The tap time is the SECOND tap's peak
sample (that is when LAP 1 / LAP 20 were pressed).

    python -m src.breath_hold.sync --check <tap_check_file>                 # mandatory pre-session tap check
    python -m src.breath_hold.sync --detect <recording>                     # start/end markers
    python -m src.breath_hold.sync --align <recording> --laps <laps.csv> [--out DIR]
    python -m src.breath_hold.sync --align <recording> --laps <laps.csv> --stopwatch-only --stop-total <seconds>

laps.csv: columns `lap,total` (stopwatch TOTAL time at each lap press, seconds or mm:ss[.xx]); 20 rows (Lap 1 = sync).
Every alignment output carries a `sync_mode` column and banner: `taps` or `STOPWATCH-ONLY` (the latter must be labelled as
such in every downstream analysis output, as the in-sample/out-of-sample label is in section 7f).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, peak_widths, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingest.sensortile import load_sensortile  # noqa: E402

# ----------------------------------------------------------------------------- pre-specified constants
HP_HZ, HP_ORDER = 10.0, 4
Z_CAND = 25.0
FWHM_MAX_MS = 60.0
MIN_SEP_S = 0.08           # peaks closer than this are one tap's ring-down (keep the largest)
GAP_S = (0.25, 1.2)
RATIO_MAX = 3.0
ISO_S = 0.6
Z_ACCEPT = 60.0            # pilot check: each trial tap >= 60 sigma
Z_TARGET = 100.0           # pilot check: target (not required)
DOMINANCE = 2.0            # >= 2x any other nearby transient
DOM_WIN_S = 10.0
EXCL_S = 0.08              # samples this close to a tap are the tap's own ring-down (== MIN_SEP_S)
START_WINDOW_S, END_WINDOW_S = 60.0, 120.0
CHECK_MIN_TRIALS, CHECK_MIN_SPACING_S = 3, 15.0
DRIFT_TOL_S = 0.5          # |drift| below this -> start anchor only
INSET_S = 2.0              # approved primary hold-window inset
INSET_STOPWATCH_ONLY_S = 5.0
PRE_GAP_S, PRE_LEN_S = 5.0, 15.0
ORDER_A = [30, 10, 45, 5, 30, 60, 10, 5, 30]   # planned hold durations (s), protocol order A
N_LAPS = 20


# ----------------------------------------------------------------------------- tap signal
_SOS = {}


def _sos(fs: float):
    k = round(fs, 3)
    if k not in _SOS:
        _SOS[k] = butter(HP_ORDER, HP_HZ, btype="high", fs=fs, output="sos")
    return _SOS[k]


def hf_magnitude(accel_g: np.ndarray, fs: float) -> np.ndarray:
    hp = sosfiltfilt(_sos(fs), accel_g - accel_g.mean(0), axis=0)
    return np.sqrt((hp ** 2).sum(1))


def robust_z(m: np.ndarray):
    med = float(np.median(m)); sig = float(1.4826 * np.median(np.abs(m - med)))
    return (m - med) / sig, med, sig


def find_pairs(z: np.ndarray, t: np.ndarray, fs: float) -> pd.DataFrame:
    """Evaluate every consecutive pair of candidate peaks against the marker rules; keep the reason for each decision."""
    pk, _ = find_peaks(z, height=Z_CAND, distance=max(1, int(MIN_SEP_S * fs)))
    if len(pk) < 2:
        return pd.DataFrame(columns=["i1", "i2", "t1", "t2", "z1", "z2", "gap_s", "ratio", "fwhm1_ms", "fwhm2_ms",
                                     "iso_ok", "dom_ratio", "accepted", "reasons"])
    fw = peak_widths(z, pk, rel_height=0.5)[0] / fs * 1000.0
    ex = int(EXCL_S * fs)
    rows = []
    for a in range(len(pk) - 1):
        i1, i2 = pk[a], pk[a + 1]
        gap = t[i2] - t[i1]
        if gap > GAP_S[1] + 1e-9 or gap < GAP_S[0] - 1e-9:
            # not a candidate pair at all (kept out of the table to keep it readable)
            continue
        ratio = max(z[i1], z[i2]) / min(z[i1], z[i2])
        lo, hi = np.searchsorted(t, [t[i1] - ISO_S, t[i2] + ISO_S])
        others = [k for k in pk if lo <= k < hi and abs(t[k] - t[i1]) > EXCL_S and abs(t[k] - t[i2]) > EXCL_S]
        iso_ok = len(others) == 0
        dlo, dhi = np.searchsorted(t, [t[i1] - DOM_WIN_S, t[i2] + DOM_WIN_S])
        mask = np.ones(dhi - dlo, bool)
        for i in (i1, i2):
            mask[max(0, i - ex - dlo):max(0, i + ex - dlo + 1)] = False
        other_max = float(z[dlo:dhi][mask].max()) if mask.any() else 0.0
        dom = min(z[i1], z[i2]) / max(other_max, 1e-9)
        reasons = []
        if fw[a] > FWHM_MAX_MS or fw[a + 1] > FWHM_MAX_MS: reasons.append("too wide")
        if ratio > RATIO_MAX: reasons.append("amplitude ratio")
        if not iso_ok: reasons.append("not isolated")
        if min(z[i1], z[i2]) < Z_ACCEPT: reasons.append(f"weak (<{Z_ACCEPT:.0f} sigma)")
        if dom < DOMINANCE: reasons.append(f"not dominant (<{DOMINANCE:.0f}x)")
        rows.append(dict(i1=int(i1), i2=int(i2), t1=float(t[i1]), t2=float(t[i2]), z1=float(z[i1]), z2=float(z[i2]), gap_s=float(gap),
                         ratio=float(ratio), fwhm1_ms=float(fw[a]), fwhm2_ms=float(fw[a + 1]), iso_ok=iso_ok, dom_ratio=float(dom),
                         accepted=not reasons, reasons="; ".join(reasons)))
    return pd.DataFrame(rows)


def select_marker(pairs: pd.DataFrame, t: np.ndarray, which: str):
    """Best accepted pair inside the search window ('start': first 60 s; 'end': last 120 s). None if there is none."""
    if pairs.empty:
        return None
    acc = pairs[pairs.accepted]
    acc = acc[acc.t1 <= START_WINDOW_S] if which == "start" else acc[acc.t2 >= t[-1] - END_WINDOW_S]
    if acc.empty:
        return None
    return acc.assign(strength=np.minimum(acc.z1, acc.z2)).sort_values("strength", ascending=False).iloc[0]


def analyse(accel_g: np.ndarray, t: np.ndarray, fs: float):
    z, med, sig = robust_z(hf_magnitude(accel_g, fs))
    return z, med, sig, find_pairs(z, t, fs)


# ----------------------------------------------------------------------------- pre-session tap check
def tap_check(path: str | Path) -> bool:
    rec = load_sensortile(path)
    t = (rec.timestamp_us - rec.timestamp_us[0]) / 1e6
    z, med, sig, pairs = analyse(rec.accel_mg / 1000.0, t, rec.fs)
    print("=" * 88)
    print(f"TAP CHECK  {Path(path).name}   ({rec.duration_s:.0f} s, fs {rec.fs:.1f} Hz, noise sigma {sig:.4f} g)")
    print(f"Pass rule (fixed in advance): >= {CHECK_MIN_TRIALS} trial double-taps (>= {CHECK_MIN_SPACING_S:.0f} s apart), each tap >= {Z_ACCEPT:.0f} sigma "
          f"and >= {DOMINANCE:.0f}x any other transient within +/-{DOM_WIN_S:.0f} s. Target >= {Z_TARGET:.0f} sigma.")
    acc = pairs[pairs.accepted].sort_values("t1") if not pairs.empty else pairs
    kept, last = [], -1e9
    for _, r in acc.iterrows():
        if r.t1 - last >= CHECK_MIN_SPACING_S:
            kept.append(r); last = r.t2
    for k, r in enumerate(kept, 1):
        weak = min(r.z1, r.z2)
        print(f"  trial {k}: t={r.t1:7.1f}s  taps {r.z1:6.0f} / {r.z2:6.0f} sigma ({r.z1*sig+med:.2f} / {r.z2*sig+med:.2f} g), gap {r.gap_s:.2f}s, "
              f"dominance {r.dom_ratio:.1f}x  -> {'meets target' if weak >= Z_TARGET else 'passes'}")
    if not pairs.empty:
        rej = pairs[~pairs.accepted]
        for _, r in rej.iterrows():
            print(f"  (rejected candidate pair at t={r.t1:.1f}s: {r.reasons})")
    ok = len(kept) >= CHECK_MIN_TRIALS
    print("-" * 88)
    print(f"TAP CHECK {'PASSED' if ok else 'FAILED'}: {len(kept)} accepted trial double-tap(s), need {CHECK_MIN_TRIALS}.")
    if not ok:
        print("FAILED -> tap harder / on the box edge / tighten the strap and repeat the check. If it still fails the session proceeds on "
              "STOPWATCH-ONLY sync and ALL its analysis outputs must be labelled 'STOPWATCH-ONLY' (protocol section 10).")
    print("=" * 88)
    return ok


# ----------------------------------------------------------------------------- laps
def _to_s(x) -> float:
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        sec = 0.0
        for p in parts:
            sec = sec * 60 + p
        return sec
    return float(s)


def parse_laps(path: str | Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d.columns = [c.strip().lower() for c in d.columns]
    if not {"lap", "total"} <= set(d.columns):
        raise ValueError("laps.csv needs columns: lap,total")
    d = d[["lap", "total"] + [c for c in d.columns if c not in ("lap", "total")]].copy()
    d["total_s"] = d.total.map(_to_s)
    d = d.sort_values("lap").reset_index(drop=True)
    if len(d) != N_LAPS or list(d.lap) != list(range(1, N_LAPS + 1)):
        print(f"WARNING: expected laps 1..{N_LAPS}, got {list(d.lap)}", file=sys.stderr)
    return d


def lap_times_to_recording(laps: pd.DataFrame, t_start_tap: float, t_end_tap: float | None = None):
    """Map stopwatch totals to recording time using the start anchor (lap 1 = 2nd start tap) and, if given, the end anchor
    (lap 20 = 2nd end tap). Decision rule fixed in advance: drift = (t_end - t_start) - (lap20 - lap1); |drift| < 0.5 s -> start
    anchor only (scale 1); otherwise linear interpolation between the two anchors."""
    tot = laps.set_index("lap").total_s
    info = dict(anchor="start only", drift_s=np.nan, scale=1.0)
    scale = 1.0
    if t_end_tap is not None and N_LAPS in tot.index:
        span_sw = tot[N_LAPS] - tot[1]
        drift = (t_end_tap - t_start_tap) - span_sw
        info["drift_s"] = float(drift)
        if abs(drift) >= DRIFT_TOL_S:
            scale = (t_end_tap - t_start_tap) / span_sw; info["anchor"] = "start+end (interpolated)"
        else:
            info["anchor"] = "start only (|drift| < 0.5 s)"
    info["scale"] = scale
    return t_start_tap + (tot - tot[1]) * scale, info


def stopwatch_only_times(laps: pd.DataFrame, stop_total_s: float, rec_duration_s: float):
    """Session-1-style mapping: lag = stopwatch total at stop - recording duration; t_rec = total - lag."""
    lag = stop_total_s - rec_duration_s
    tot = laps.set_index("lap").total_s
    return tot - lag, dict(anchor="STOPWATCH-ONLY", lag_s=float(lag), scale=1.0, drift_s=np.nan)


def hold_windows(t_lap: pd.Series, mode: str, inset_s: float | None = None) -> pd.DataFrame:
    """Per-hold primary windows from lap times (laps 2..19 = 9 holds, order A), the rest-only stretches, and the pre-hold baseline.
    Insets: primary 2 s (taps) / 5 s (stopwatch-only); also 0 s and 5 s for sensitivity (NaN where the hold is too short)."""
    primary = inset_s if inset_s is not None else (INSET_STOPWATCH_ONLY_S if mode.startswith("STOPWATCH") else INSET_S)
    banner = "SYNC=taps" if mode == "taps" else "SYNC=STOPWATCH-ONLY (not tap-synchronised; boundaries uncertain by several seconds)"
    rows = []
    for k, planned in enumerate(ORDER_A, start=1):
        l0, l1 = 2 * k, 2 * k + 1
        s, e = float(t_lap[l0]), float(t_lap[l1])
        row = dict(item=f"H{k}", kind="hold", planned_s=planned, lap_start=l0, lap_end=l1, t_start=s, t_end=e, actual_s=e - s,
                   aborted_or_short=bool((e - s) < planned - 2.0))
        for name, ins in (("primary", primary), ("inset0", 0.0), ("inset5", 5.0)):
            a, b = s + ins, e - ins
            row[f"{name}_start"], row[f"{name}_end"] = (a, b) if b - a >= 1.0 else (np.nan, np.nan)
        row["primary_len_s"] = row["primary_end"] - row["primary_start"] if not np.isnan(row["primary_start"]) else np.nan
        row["primary_window_too_short"] = bool(np.isnan(row["primary_len_s"]) or row["primary_len_s"] < 3.0)
        row["pre_start"], row["pre_end"] = s - PRE_GAP_S - PRE_LEN_S, s - PRE_GAP_S
        rows.append(row)
    rows.append(dict(item="rest_start", kind="rest_only", t_start=float(t_lap[1]), t_end=float(t_lap[2])))
    rows.append(dict(item="rest_end", kind="rest_only", t_start=float(t_lap[19]), t_end=float(t_lap[20])))
    d = pd.DataFrame(rows)
    d["sync_mode"] = mode
    d["sync_label"] = banner
    return d


# ----------------------------------------------------------------------------- CLI
def _load(path):
    rec = load_sensortile(path)
    t = (rec.timestamp_us - rec.timestamp_us[0]) / 1e6
    return rec, t


def cmd_detect(path):
    rec, t = _load(path)
    z, med, sig, pairs = analyse(rec.accel_mg / 1000.0, t, rec.fs)
    out = {}
    for which in ("start", "end"):
        m = select_marker(pairs, t, which); out[which] = m
        if m is None:
            print(f"{which.upper()} marker: NOT FOUND (no accepted pair in the {'first 60 s' if which=='start' else 'last 120 s'})")
        else:
            print(f"{which.upper()} marker: second tap at t = {m.t2:.3f} s | taps {m.z1:.0f}/{m.z2:.0f} sigma | gap {m.gap_s:.2f}s | dominance {m.dom_ratio:.1f}x")
    return rec, t, pairs, out


def cmd_align(path, laps_path, out_dir, stopwatch_only=False, stop_total=None):
    rec, t, pairs, mk = cmd_detect(path)
    laps = parse_laps(laps_path)
    if stopwatch_only or mk["start"] is None:
        if stop_total is None:
            sys.exit("STOPWATCH-ONLY alignment needs --stop-total <stopwatch total seconds at recording stop>.")
        t_lap, info = stopwatch_only_times(laps, _to_s(stop_total), rec.duration_s); mode = "STOPWATCH-ONLY"
        print("\n*** STOPWATCH-ONLY SYNC: outputs below are labelled STOPWATCH-ONLY and must stay labelled in any analysis. ***")
    else:
        t_lap, info = lap_times_to_recording(laps, float(mk["start"].t2), float(mk["end"].t2) if mk["end"] is not None else None); mode = "taps"
    W = hold_windows(t_lap, mode)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    W.to_csv(out / "aligned_windows.csv", index=False)
    pd.DataFrame(dict(lap=t_lap.index, t_rec_s=t_lap.values, sync_mode=mode)).to_csv(out / "lap_times_recording.csv", index=False)
    print(f"\nsync_mode = {mode} | {info}")
    print(W[W.kind == "hold"][["item", "planned_s", "actual_s", "t_start", "t_end", "primary_start", "primary_end", "primary_window_too_short", "aborted_or_short"]].round(2).to_string(index=False))
    print(f"\nSaved {out/'aligned_windows.csv'} and {out/'lap_times_recording.csv'}")
    return W


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check"); g.add_argument("--detect"); g.add_argument("--align")
    ap.add_argument("--laps"); ap.add_argument("--out", default="results/breath_hold/session2_alignment")
    ap.add_argument("--stopwatch-only", action="store_true"); ap.add_argument("--stop-total")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if tap_check(a.check) else 1)
    if a.detect:
        cmd_detect(a.detect); return
    if not a.laps:
        sys.exit("--align needs --laps")
    cmd_align(a.align, a.laps, a.out, a.stopwatch_only, a.stop_total)


if __name__ == "__main__":
    main()
