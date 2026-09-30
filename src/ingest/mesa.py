"""MESA (NSRR) loader: PSG EDF channels + NSRR event-annotation XML -> label grid.

Track B (separate from the PhysioNet / in-house tracks). Subject key = mesaid (4-digit str).

Files (under data/mesa/, gitignored -- never commit):
  polysomnography/edfs/mesa-sleep-XXXX.edf
  polysomnography/annotations-events-nsrr/mesa-sleep-XXXX-nsrr.xml
  polysomnography/annotations-events-profusion/mesa-sleep-XXXX-profusion.xml   (not used here)

Label-grid choice: **30 s epochs** (MESA-native AASM scoring resolution; the scorer's
events, sleep stages and central/obstructive designation are all 30 s-epoch based).
A per-minute grid (continuity with the PhysioNet pipeline) is derived from it with
`to_per_minute()`, so no information is lost by choosing 30 s.

An epoch is POSITIVE for an event class when scored events of that class overlap
the epoch by >= MIN_OVERLAP_S seconds in total (edge-grazing events don't flip an epoch).

Sampling rates: every channel keeps its native fs (EKG 256, Thor/Abdo 32, SpO2 1). Epoching
is done in TIME (seconds) via `epoch_slice()`, so rates never enter label alignment;
`resample_to()` is provided for when a common grid is needed (e.g. EKG -> 100 Hz for
the PhysioNet-trained ECG branch, effort -> a single rate).
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("mesa")

MESA_ROOT = Path("data/mesa")
EDF_DIR = MESA_ROOT / "polysomnography" / "edfs"
XML_DIR = MESA_ROOT / "polysomnography" / "annotations-events-nsrr"

EPOCH_S = 30.0
MIN_OVERLAP_S = 5.0

# canonical name -> exact EDF label (matched case-insensitively, whitespace-stripped)
CHANNELS = {"ecg": "EKG", "thor": "Thor", "abdo": "Abdo", "spo2": "SpO2"}

# NSRR event concepts -> our subtype names (matched on the text before/after '|', lower-case)
_SUBTYPES = {
    "obstructive apnea": "obstructive",
    "central apnea": "central",
    "mixed apnea": "mixed",
    "hypopnea": "hypopnea",
}


def edf_path(mesaid: str) -> Path:
    return EDF_DIR / f"mesa-sleep-{mesaid}.edf"


def xml_path(mesaid: str) -> Path:
    return XML_DIR / f"mesa-sleep-{mesaid}-nsrr.xml"


def list_downloaded() -> list[str]:
    """mesaids that have BOTH an EDF and an event XML on disk."""
    ids = []
    for p in sorted(EDF_DIR.glob("mesa-sleep-*.edf")):
        m = re.search(r"mesa-sleep-(\d+)\.edf$", p.name)
        if m and xml_path(m.group(1)).exists():
            ids.append(m.group(1))
    return ids


@dataclass
class Channel:
    label: str       # exact EDF label found
    fs: float        # native sampling rate (Hz)
    x: np.ndarray    # physical units, float32


@dataclass
class PSG:
    mesaid: str
    duration_s: float
    channels: dict[str, Channel] = field(default_factory=dict)  # canonical name -> Channel
    all_labels: list[str] = field(default_factory=list)
    all_fs: list[float] = field(default_factory=list)


def load_psg(mesaid: str, channels: dict[str, str] | None = None, log_names: bool = False) -> PSG:
    """Load only the wanted channels (by exact label, auto-detected from the EDF header)."""
    import pyedflib

    channels = channels or CHANNELS
    r = pyedflib.EdfReader(str(edf_path(mesaid)))
    try:
        labels = [s.strip() for s in r.getSignalLabels()]
        fss = [float(r.getSampleFrequency(i)) for i in range(len(labels))]
        psg = PSG(mesaid, float(r.getFileDuration()), all_labels=labels, all_fs=fss)
        lower = [s.lower() for s in labels]
        for canon, want in channels.items():
            if want.lower() in lower:
                i = lower.index(want.lower())
                psg.channels[canon] = Channel(labels[i], fss[i], r.readSignal(i).astype(np.float32))
        if log_names:
            log.info("mesa %s: duration %.0fs; EDF labels/fs: %s", mesaid, psg.duration_s,
                     ", ".join(f"{l}@{f:g}" for l, f in zip(labels, fss)))
            log.info("mesa %s: matched %s", mesaid,
                     {k: (c.label, c.fs) for k, c in psg.channels.items()})
    finally:
        r.close()
    return psg


def parse_events(mesaid: str) -> pd.DataFrame:
    """NSRR event XML -> DataFrame(event_type, concept, subtype, start_s, duration_s).

    `subtype` is obstructive/central/mixed/hypopnea for respiratory events, else ''.
    Sleep-stage events are kept (event_type 'Stages|Stages') so sleep time can be derived.
    """
    root = ET.parse(xml_path(mesaid)).getroot()
    rows = []
    for ev in root.iter("ScoredEvent"):
        et = (ev.findtext("EventType") or "").strip()
        concept = (ev.findtext("EventConcept") or "").strip()
        start = float(ev.findtext("Start") or "nan")
        dur = float(ev.findtext("Duration") or "nan")
        sub = ""
        for part in concept.lower().split("|"):
            if part.strip() in _SUBTYPES:
                sub = _SUBTYPES[part.strip()]
                break
        rows.append((et, concept, sub, start, dur))
    return pd.DataFrame(rows, columns=["event_type", "concept", "subtype", "start_s", "duration_s"])


def stage_series(events: pd.DataFrame) -> pd.DataFrame:
    """Sleep-stage events -> DataFrame(start_s, duration_s, stage) with stage in {W,N1,N2,N3,R,?}.
    NSRR concepts look like 'Wake|0', 'Stage 1 sleep|1', ..., 'REM sleep|5'."""
    st = events[events.event_type.str.lower().str.startswith("stages")].copy()

    def code(c):
        n = c.split("|")[-1].strip()
        return {"0": "W", "1": "N1", "2": "N2", "3": "N3", "4": "N3", "5": "R"}.get(n, "?")

    st["stage"] = st.concept.map(code)
    return st[["start_s", "duration_s", "stage"]].reset_index(drop=True)


def label_grid(events: pd.DataFrame, duration_s: float, epoch_s: float = EPOCH_S) -> pd.DataFrame:
    """Per-epoch label grid from scored events (independent of desaturation).

    Columns: epoch, t0_s, sec_{obstructive,central,mixed,hypopnea} (event overlap seconds),
    apnea (obstructive|central|mixed overlap >= MIN_OVERLAP_S), hypopnea,
    resp_event (apnea | hypopnea), subtype (dominant by overlap; 'normal' if none), stage.
    """
    n = int(np.floor(duration_s / epoch_s))
    t0 = np.arange(n) * epoch_s
    g = pd.DataFrame({"epoch": np.arange(n), "t0_s": t0})
    for sub in ("obstructive", "central", "mixed", "hypopnea"):
        ov = np.zeros(n)
        for s, d in events.loc[events.subtype == sub, ["start_s", "duration_s"]].itertuples(index=False):
            if not (np.isfinite(s) and np.isfinite(d)) or d <= 0:
                continue
            a, b = int(max(0, s // epoch_s)), int(min(n - 1, (s + d) // epoch_s))
            for k in range(a, b + 1):
                ov[k] += max(0.0, min(s + d, t0[k] + epoch_s) - max(s, t0[k]))
        g[f"sec_{sub}"] = ov
    ap = g[["sec_obstructive", "sec_central", "sec_mixed"]].sum(axis=1)
    g["apnea"] = (ap >= MIN_OVERLAP_S).astype(int)
    g["hypopnea"] = (g.sec_hypopnea >= MIN_OVERLAP_S).astype(int)
    g["resp_event"] = ((g.apnea == 1) | (g.hypopnea == 1)).astype(int)
    secs = g[["sec_obstructive", "sec_central", "sec_mixed", "sec_hypopnea"]]
    dom = secs.idxmax(axis=1).str.replace("sec_", "", regex=False)
    g["subtype"] = np.where(g.resp_event == 1, dom, "normal")
    # sleep stage per epoch (stage events are 30 s, start at multiples of 30 s)
    stg = stage_series(events)
    g["stage"] = "?"
    for s, d, lab in stg.itertuples(index=False):
        a = int(s // epoch_s)
        b = int(np.ceil((s + d) / epoch_s))
        g.loc[(g.epoch >= a) & (g.epoch < b), "stage"] = lab
    return g


def to_per_minute(grid: pd.DataFrame) -> pd.DataFrame:
    """Collapse the 30 s grid to 60 s (PhysioNet-style): minute positive if either half is."""
    g = grid.copy()
    g["minute"] = g.epoch // 2
    return g.groupby("minute").agg(t0_s=("t0_s", "min"), apnea=("apnea", "max"),
                                   hypopnea=("hypopnea", "max"),
                                   resp_event=("resp_event", "max")).reset_index()


def epoch_slice(ch: Channel, t0_s: float, dur_s: float = EPOCH_S) -> np.ndarray:
    """Samples of `ch` covering [t0_s, t0_s+dur_s) -- alignment is by time, not by sample count."""
    a, b = int(round(t0_s * ch.fs)), int(round((t0_s + dur_s) * ch.fs))
    return ch.x[a:b]


def resample_to(x: np.ndarray, fs: float, target_fs: float) -> np.ndarray:
    """Polyphase resample to a common grid (e.g. EKG 256 -> 100 Hz, effort 32 -> 25 Hz)."""
    from fractions import Fraction
    from scipy.signal import resample_poly

    fr = Fraction(target_fs / fs).limit_denominator(1000)
    return resample_poly(x, fr.numerator, fr.denominator).astype(np.float32)


def sleep_hours(grid: pd.DataFrame) -> float:
    """Hours of scored sleep (N1-R). NaN if the XML carried no stage events."""
    if (grid.stage == "?").all():
        return float("nan")
    return float(grid.stage.isin(["N1", "N2", "N3", "R"]).sum() * EPOCH_S / 3600.0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ids = list_downloaded()
    print(f"{len(ids)} subjects with EDF+XML on disk")
    for mid in ids[:3]:
        psg = load_psg(mid, log_names=True)
        ev = parse_events(mid)
        g = label_grid(ev, psg.duration_s)
        print(mid, "epochs", len(g), "apnea", int(g.apnea.sum()), "hypopnea", int(g.hypopnea.sum()),
              "sleep_h", round(sleep_hours(g), 2))
        print(ev.concept.value_counts().head(12).to_string())
