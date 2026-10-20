"""Annotation file for the in-house breath-hold recordings (Track B).  python3 -m src.inhouse_bh.annotate

Labels (compatible mapping to earlier terminology: 'hold' / 'event' in results/breath_hold/ = voluntary_breath_hold):
  baseline_reference | voluntary_breath_hold | recovery | uncertain
Clinical labels (sleep_apnea, obstructive_apnea, central_apnea) are NEVER written here: voluntary holds are not clinical events.

SESSION 1 (recording id S1_ECGTILE; one volunteer, 2026-10-07).  Stopwatch laps 2,4,..,14 = 7 holds of ~30 s (nominal times in results/breath_hold/event_summary.csv).
  Recording-vs-stopwatch lag is 4.8 s in total (stopwatch total 1800.46 s vs recording 1795.6 s; direction/shape unknown) and the detector-estimated boundaries differ from nominal by
  up to ~7.5 s at the ends. So, with MARGIN = 5 s at each end:
    voluntary_breath_hold : [nominal_start+5, nominal_end-5]       (~20 s core)   confidence 'medium' ('high' where a visual check of the signal confirmed quiet boundaries: events 2,4,6)
    uncertain             : [nominal_start-5, nominal_start+5] and [nominal_end-5, nominal_end+5]   (boundary bands; excluded from supervised training)
    recovery              : [nominal_end+5, nominal_end+25]
    baseline_reference    : [prev recovery end+5, nominal_start-15] (for event 1: the 120 s before), i.e. normal breathing between holds
SESSION 2 (Mithin, 2026-10-11): the stopwatch timeline is reconstructed from the video (results/breath_hold_s2/mithin/laps_extracted.csv) but NO tile could be aligned to it (shuffle-null
  p = 0.49-0.95, results/breath_hold_s2/mithin/REPORT.md).  Its intervals are therefore written in the STOPWATCH time base, label 'uncertain', verification 'unaligned', and are NOT usable for
  supervised training or evaluation.  The recordings themselves remain usable as unlabelled data.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

OUT = Path("results/inhouse_bh/annotations.csv")
MARGIN, REC_S, BASE_GAP = 5.0, 20.0, 15.0
VISUAL_OK = {2, 4, 6}        # detector_visual_verdict == OK in results/breath_hold/event_summary.csv (boundaries sit on a visible quiet stretch)


def session1() -> list[dict]:
    ev = pd.read_csv("results/breath_hold/event_summary.csv"); rows = []; prev_rec_end = None
    for _, e in ev.iterrows():
        n, s, t = int(e.event), float(e.nominal_start), float(e.nominal_end); conf = "high" if n in VISUAL_OK else "medium"
        ver = "boundaries_visually_checked" if n in VISUAL_OK else "nominal_stopwatch_inset_only"
        b0 = (prev_rec_end + 5.0) if prev_rec_end else s - 120.0
        add = lambda a, b, lab, c, src, v, note="": rows.append(dict(recording_id="S1_ECGTILE", event_no=n, start_s=round(a, 2), end_s=round(b, 2), label=lab, confidence=c,
                                                                    annotation_source=src, verification_status=v, time_base="recording", notes=note))
        add(b0, s - BASE_GAP, "baseline_reference", "medium", "protocol rest between holds (stopwatch laps) + margins", "nominal_stopwatch_inset_only")
        add(s - MARGIN, s + MARGIN, "uncertain", "low", "boundary band (+-5 s: stopwatch lag 4.8 s, detector offsets up to 7.5 s)", "unresolved")
        add(s + MARGIN, t - MARGIN, "voluntary_breath_hold", conf, "stopwatch lap times inset by 5 s", ver, "voluntary hold; NOT a clinical apnea event")
        add(t - MARGIN, t + MARGIN, "uncertain", "low", "boundary band", "unresolved")
        add(t + MARGIN, t + MARGIN + REC_S, "recovery", "medium", "20 s after the end boundary band", ver)
        prev_rec_end = t + MARGIN + REC_S
    return rows


def session2() -> list[dict]:
    lt = pd.read_csv("results/breath_hold_s2/mithin/laps_extracted.csv"); rows = []
    for _, r in lt.iterrows():
        hold = "HOLD" in str(r.inferred_role)
        rows.append(dict(recording_id="S2_STOPWATCH_ONLY", event_no=(int(str(r.inferred_role).split()[1]) if hold else None), start_s=r.start_s_calculated, end_s=r.end_s_calculated,
                         label="uncertain", confidence="none", annotation_source="stopwatch video lap list (calculated cumulative times)", verification_status="unaligned_to_any_tile",
                         time_base="stopwatch", notes=("inferred voluntary hold (30 s lap)" if hold else "inferred rest/lead-in") + "; no sensor alignment established -> excluded from supervised use"))
    return rows


if __name__ == "__main__":
    df = pd.DataFrame(session1() + session2()); OUT.parent.mkdir(parents=True, exist_ok=True); df.to_csv(OUT, index=False)
    print(df.groupby(["recording_id", "label"]).agg(n=("start_s", "size"), seconds=("end_s", lambda s: 0)).join(df.assign(d=df.end_s - df.start_s).groupby(["recording_id", "label"]).d.sum().rename("total_s")).round(1))
