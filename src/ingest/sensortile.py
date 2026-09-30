"""
SensorTile.box PRO recording loader.

Parses the 16-column tab-delimited .txt format produced by the SensorTile
(see CLAUDE.md Sec.10):
    Timestamp [us]
    A_X/Y/Z [LSB], A_X/Y/Z [mg]
    G_X/Y/Z [LSB], G_X/Y/Z [dps]
    T [LSB], T [degC]
    QVAR [LSB]   -- wired as the ECG channel

QVAR is the biosignal (ECG) channel. The accelerometer (mg) channels are the
effort branch and are always kept, never dropped.

Per CLAUDE.md rule 8/9: never trust the nominal 240 Hz. This loader always
computes the empirical sample rate from the microsecond timestamp column and
warns loudly if it disagrees with nominal by more than 2%. It also detects
timestamp gaps/dropouts (non-monotonic or abnormally large jumps).

Usage:
    python src/ingest/sensortile.py <path/to/recording.txt>
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

NOMINAL_FS_HZ = 240.0
FS_WARN_PCT = 2.0  # warn if measured fs differs from nominal by more than this

EXPECTED_COLUMNS = [
    "Timestamp [us]",
    "A_X [LSB]", "A_Y [LSB]", "A_Z [LSB]",
    "A_X [mg]", "A_Y [mg]", "A_Z [mg]",
    "G_X [LSB]", "G_Y [LSB]", "G_Z [LSB]",
    "G_X [dps]", "G_Y [dps]", "G_Z [dps]",
    "T [LSB]", "T [degC]",
    "QVAR [LSB]",
]


@dataclass
class SensorTileRecord:
    """One loaded SensorTile recording. All channels kept (IMU + QVAR/ECG)."""

    name: str
    n_samples: int
    timestamp_us: np.ndarray  # raw microsecond timestamps
    accel_mg: np.ndarray  # Nx3, physical units (mg)
    gyro_dps: np.ndarray  # Nx3, physical units (dps)
    temp_degC: np.ndarray  # N, physical units
    qvar: np.ndarray  # N, raw LSB -- the ECG channel

    fs_nominal: float = NOMINAL_FS_HZ
    fs_measured: float = field(default=float("nan"))
    duration_s: float = field(default=float("nan"))
    n_gaps: int = 0
    gap_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    # Aliases so this can be used wherever ecg/rpeaks.py expects an
    # ApneaRecord-like object (it only reads .signal and .fs).
    @property
    def signal(self) -> np.ndarray:
        return self.qvar

    @property
    def fs(self) -> float:
        return self.fs_measured

    @property
    def accel_is_usable(self) -> bool:
        """True if the accelerometer channels are present and not flat/dead."""
        if self.accel_mg.size == 0:
            return False
        stds = np.nanstd(self.accel_mg, axis=0)
        return bool(np.any(stds > 1e-6))


def _find_header_row(path: Path, max_scan_lines: int = 20) -> int:
    """Return the 0-based line index of the header row (the line containing
    'Timestamp'). Robust to leading metadata lines."""
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= max_scan_lines:
                break
            if "Timestamp" in line:
                return i
    raise ValueError(f"{path}: could not find a header row containing 'Timestamp' "
                      f"in the first {max_scan_lines} lines")


def load_sensortile(path: str | Path) -> SensorTileRecord:
    """Load one SensorTile .txt recording. Robust to leading metadata lines,
    a trailing empty column from the trailing tab, and short/tiny (aborted
    capture) files."""
    path = Path(path)
    name = path.stem

    header_row = _find_header_row(path)

    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=header_row,
        header=0,
        engine="c",
        skip_blank_lines=True,
    )
    # Drop the trailing empty column produced by the trailing tab in the header.
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing expected columns {missing}; "
                          f"found columns {list(df.columns)}")

    df = df.dropna(subset=["Timestamp [us]"])
    n_samples = len(df)

    timestamp_us = df["Timestamp [us]"].to_numpy(dtype=np.float64)
    accel_mg = df[["A_X [mg]", "A_Y [mg]", "A_Z [mg]"]].to_numpy(dtype=np.float64)
    gyro_dps = df[["G_X [dps]", "G_Y [dps]", "G_Z [dps]"]].to_numpy(dtype=np.float64)
    temp_degC = df["T [degC]"].to_numpy(dtype=np.float64)
    qvar = df["QVAR [LSB]"].to_numpy(dtype=np.float64)

    rec = SensorTileRecord(
        name=name,
        n_samples=n_samples,
        timestamp_us=timestamp_us,
        accel_mg=accel_mg,
        gyro_dps=gyro_dps,
        temp_degC=temp_degC,
        qvar=qvar,
    )

    if n_samples < 2:
        print(f"WARNING: {name}: only {n_samples} sample(s) -- too short to "
              f"compute sample rate or detect gaps (likely an aborted capture).")
        return rec

    duration_s = (timestamp_us[-1] - timestamp_us[0]) / 1e6
    fs_measured = (n_samples - 1) / duration_s if duration_s > 0 else float("nan")
    rec.duration_s = duration_s
    rec.fs_measured = fs_measured

    pct_diff = 100.0 * abs(fs_measured - NOMINAL_FS_HZ) / NOMINAL_FS_HZ
    print(f"{name}: measured fs={fs_measured:.2f} Hz vs nominal {NOMINAL_FS_HZ:.0f} Hz "
          f"({pct_diff:+.2f}% diff), n_samples={n_samples}, duration={duration_s:.1f}s")
    if pct_diff > FS_WARN_PCT:
        print(f"WARNING: {name}: measured sample rate differs from nominal "
              f"{NOMINAL_FS_HZ:.0f} Hz by {pct_diff:.2f}% (> {FS_WARN_PCT:.0f}% threshold). "
              f"Do not assume 240 Hz for this recording.")

    diffs_us = np.diff(timestamp_us)
    median_dt = np.median(diffs_us) if len(diffs_us) else float("nan")
    non_monotonic = diffs_us <= 0
    large_jump = diffs_us > 3 * median_dt if median_dt > 0 else np.zeros_like(diffs_us, dtype=bool)
    gap_mask = non_monotonic | large_jump
    gap_indices = np.nonzero(gap_mask)[0]
    rec.n_gaps = int(len(gap_indices))
    rec.gap_indices = gap_indices

    if rec.n_gaps > 0:
        print(f"WARNING: {name}: {rec.n_gaps} timestamp gap/dropout(s) detected "
              f"(non-monotonic or >3x median inter-sample interval of {median_dt:.1f}us).")

    if not rec.accel_is_usable:
        print(f"WARNING: {name}: accelerometer channels are flat/absent -- "
              f"effort branch not testable on this file.")

    return rec


def _report(path: str) -> SensorTileRecord:
    rec = load_sensortile(path)
    print(f"  accel usable: {rec.accel_is_usable}")
    return rec


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        _report(p)
