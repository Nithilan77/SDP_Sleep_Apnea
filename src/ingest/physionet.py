"""
PhysioNet Apnea-ECG loader.

Loads a single-lead ECG record plus its per-minute apnea/normal labels from
the PhysioNet Apnea-ECG database (a01-a20, b01-b05, c01-c10 have .apn labels;
x01-x35 are withheld test recordings with no .apn file).

Data layout expected (relative to project root):
    data/physionet/<record>.dat/.hea/.apn/.qrs

Usage:
    python src/ingest/physionet.py            # sanity-check a01 and c01
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

DATA_DIR = Path("data/physionet")


@dataclass
class ApneaRecord:
    """One loaded Apnea-ECG record."""

    name: str
    fs: float  # sampling rate in Hz
    signal: np.ndarray  # 1-D ECG signal, physical units (mV)
    labels: np.ndarray | None  # per-minute labels, 'A' or 'N' (None for x-records)

    @property
    def n_minutes_labelled(self) -> int:
        return 0 if self.labels is None else len(self.labels)

    @property
    def n_apnea_minutes(self) -> int:
        return 0 if self.labels is None else int(np.sum(self.labels == "A"))

    @property
    def n_normal_minutes(self) -> int:
        return 0 if self.labels is None else int(np.sum(self.labels == "N"))


def load_record(name: str, data_dir: Path = DATA_DIR) -> ApneaRecord:
    """Load one Apnea-ECG record by name (e.g. 'a01', 'c01', 'x01').

    Records a01-c10 have a companion .apn annotation file with one label per
    minute of recording. x-records (withheld test set) have no .apn file, so
    labels will be None for those.
    """
    record_path = str(data_dir / name)

    record = wfdb.rdrecord(record_path)
    fs = float(record.fs)
    signal = record.p_signal[:, 0]  # single ECG channel

    labels = None
    apn_path = data_dir / f"{name}.apn"
    if apn_path.exists():
        ann = wfdb.rdann(record_path, extension="apn")
        # ann.symbol is a list of 'A'/'N' chars, one per minute
        labels = np.array(ann.symbol)

    return ApneaRecord(name=name, fs=fs, signal=signal, labels=labels)


def _report(name: str) -> ApneaRecord:
    rec = load_record(name)
    duration_min = len(rec.signal) / rec.fs / 60.0
    print(f"{name}: fs={rec.fs:.1f} Hz, {len(rec.signal)} samples, "
          f"~{duration_min:.1f} min")
    if rec.labels is not None:
        print(f"  labelled minutes: {rec.n_minutes_labelled} "
              f"(apnea={rec.n_apnea_minutes}, normal={rec.n_normal_minutes})")
    else:
        print("  no .apn labels (withheld test record)")
    return rec


if __name__ == "__main__":
    print("=== PhysioNet Apnea-ECG loader sanity check ===")
    a01 = _report("a01")
    c01 = _report("c01")

    assert a01.fs == 100.0, f"expected 100 Hz, got {a01.fs}"
    expected_a01 = (470, 19)
    got_a01 = (a01.n_apnea_minutes, a01.n_normal_minutes)
    expected_c01 = (0, 484)
    got_c01 = (c01.n_apnea_minutes, c01.n_normal_minutes)

    print()
    print(f"a01 expected (apnea, normal) = {expected_a01}, got = {got_a01} "
          f"-> {'MATCH' if got_a01 == expected_a01 else 'MISMATCH'}")
    print(f"c01 expected (apnea, normal) = {expected_c01}, got = {got_c01} "
          f"-> {'MATCH' if got_c01 == expected_c01 else 'MISMATCH'}")
