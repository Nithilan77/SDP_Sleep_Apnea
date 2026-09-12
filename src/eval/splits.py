"""
Subject-wise splitting utilities for Apnea-ECG.

Each Apnea-ECG record (a01, b03, c07, ...) is a different subject/patient, so
"subject-independent" evaluation here means leave-one-record-out: for every
fold, one whole record is held out for testing and the model never sees any
of that record's minutes during training. This is the only split type used
in this project's evaluations -- mixed-subject (minute-level) splits leak
information between train and test and inflate accuracy (see CLAUDE.md, this
project's rule 1 & 7).
"""
from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data/physionet")

# Official PhysioNet Apnea-ECG groups (from the record naming convention):
#   a01-a20 = apnea group, b01-b05 = borderline, c01-c10 = control.
APNEA_GROUP_PREFIXES = ("a", "b")
CONTROL_GROUP_PREFIX = "c"


def list_labelled_records(data_dir: Path = DATA_DIR) -> list[str]:
    """Return the 35 record names that have .apn per-minute labels, sorted."""
    names = sorted(p.stem for p in data_dir.glob("*.apn"))
    # a01r/a02r-style alternate recordings and a01er-style ones also carry .apn
    # files in this dataset dump; keep only the plain a/b/c-NN records used for
    # training (matches CLAUDE.md's a01-a20/b01-b05/c01-c10 definition).
    return [n for n in names if len(n) == 3 and n[0] in ("a", "b", "c") and n[1:].isdigit()]


def record_group(name: str) -> str:
    """Official group for a record name: 'apnea' (a/b prefix) or 'control' (c)."""
    return "control" if name.startswith(CONTROL_GROUP_PREFIX) else "apnea"


def leave_one_subject_out(records: list[str]):
    """Yield (train_records, test_record) for each record in turn.

    This is a generator over folds; each fold trains on every record except
    one and tests on the held-out one, so no subject ever appears in both
    train and test within a fold.
    """
    for i, test_name in enumerate(records):
        train_names = records[:i] + records[i + 1:]
        yield train_names, test_name
