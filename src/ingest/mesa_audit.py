"""MESA subset audit -> results/mesa/audit/.

Per subject: signal presence / native fs / duration, scored-event counts (obstructive / central /
mixed / hypopnea), sleep time, XML-derived AHI, thoracic-vs-abdominal belt quality, class balance
on the 30 s label grid. Cross-checks against NSRR's own harmonized AHI (dataset CSV) so label
definitions are validated, not assumed.

Incremental: per-subject results are cached in results/mesa/audit/.cache/, so this can be re-run
while the download is still in progress (only new complete subjects are processed).

    python -m src.ingest.mesa_audit            # from project root
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest import mesa as M

log = logging.getLogger("mesa_audit")
OUT = Path("results/mesa/audit")
CACHE = OUT / ".cache"
DATASET_CSV = M.MESA_ROOT / "datasets" / "mesa-sleep-dataset-0.8.0.csv"

RESP_BAND = (0.1, 0.6)     # Hz, breathing
TOTAL_BAND = (0.03, 3.0)   # Hz, denominator for "is there breathing-band signal"
GOOD_BAND_RATIO = 0.5      # epoch counts as "breathing visible" if >=50% of band power is in RESP_BAND
DESAT_MIN_DROP = 3.0       # %, for the desaturation-supported AHI cross-check
DESAT_WINDOW_S = 30.0      # desat may start up to this long after the event ends

# usable-subject thresholds
MIN_SLEEP_H = 4.0
MAX_FLAT_FRAC = 0.5
MIN_GOOD_EPOCH_FRAC = 0.5


def _epoch_matrix(x: np.ndarray, fs: float) -> np.ndarray:
    n = int(len(x) // (fs * M.EPOCH_S))
    k = int(round(fs * M.EPOCH_S))
    return x[: n * k].reshape(n, k).astype(np.float64)


def belt_quality(ch: M.Channel | None) -> dict:
    """Quality of a respiratory-effort belt (Thor / Abdo)."""
    if ch is None:
        return {}
    x = ch.x.astype(np.float64)
    E = _epoch_matrix(x, ch.fs)
    nan_frac = float(np.isnan(x).mean())
    E = np.nan_to_num(E)
    ptp = np.ptp(E, axis=1)
    ref = np.percentile(np.ptp(E, axis=1), 90) if len(E) else 0.0
    flat = ptp <= max(1e-9, 0.01 * ref)                  # epoch with ~no excursion
    lo, hi = np.nanmin(x), np.nanmax(x)
    clip = float(((x <= lo) | (x >= hi)).mean()) if hi > lo else 1.0
    # per-epoch fraction of power in the breathing band
    Ed = E - E.mean(axis=1, keepdims=True)
    P = np.abs(np.fft.rfft(Ed * np.hanning(E.shape[1]), axis=1)) ** 2
    f = np.fft.rfftfreq(E.shape[1], 1 / ch.fs)
    band = (f >= RESP_BAND[0]) & (f <= RESP_BAND[1])
    tot = (f >= TOTAL_BAND[0]) & (f <= TOTAL_BAND[1])
    ratio = P[:, band].sum(1) / np.maximum(P[:, tot].sum(1), 1e-12)
    good = (~flat) & (ratio >= GOOD_BAND_RATIO)
    pk = f[band][np.argmax(P[:, band], axis=1)]
    return {
        "fs": ch.fs, "nan_frac": nan_frac, "flat_frac": float(flat.mean()) if len(E) else 1.0,
        "clip_frac": clip, "band_ratio_med": float(np.median(ratio)) if len(E) else 0.0,
        "good_epoch_frac": float(good.mean()) if len(E) else 0.0,
        "breath_rate_bpm_med": float(np.median(pk[good]) * 60) if good.any() else float("nan"),
        "amp_p90": float(ref),
    }


def belt_agreement(a: M.Channel | None, b: M.Channel | None) -> float:
    """Median per-epoch correlation between the 0.1-0.6 Hz components of the two belts."""
    if a is None or b is None or a.fs != b.fs:
        return float("nan")
    from scipy.signal import butter, sosfiltfilt

    sos = butter(2, RESP_BAND, btype="band", fs=a.fs, output="sos")
    n = min(len(a.x), len(b.x))
    xa, xb = sosfiltfilt(sos, np.nan_to_num(a.x[:n])), sosfiltfilt(sos, np.nan_to_num(b.x[:n]))
    Ea, Eb = _epoch_matrix(xa, a.fs), _epoch_matrix(xb, a.fs)
    Ea -= Ea.mean(1, keepdims=True); Eb -= Eb.mean(1, keepdims=True)
    den = np.sqrt((Ea ** 2).sum(1) * (Eb ** 2).sum(1))
    r = (Ea * Eb).sum(1) / np.where(den > 0, den, np.nan)
    return float(np.nanmedian(np.abs(r))) if np.isfinite(r).any() else float("nan")


def flat_or_bad(ch: M.Channel | None, lo=None, hi=None) -> dict:
    """Generic quality for EKG / SpO2."""
    if ch is None:
        return {}
    x = ch.x.astype(np.float64)
    out = {"fs": ch.fs, "nan_frac": float(np.isnan(x).mean())}
    E = np.nan_to_num(_epoch_matrix(x, ch.fs))
    out["flat_frac"] = float((np.ptp(E, axis=1) <= 1e-9).mean()) if len(E) else 1.0
    if lo is not None:
        out["valid_frac"] = float(((x >= lo) & (x <= hi)).mean())
    return out


def desat_supported_ahi(ev: pd.DataFrame, sleep_h: float) -> float:
    """XML-only approximation of the NSRR '3% desaturation' AHI: apneas + hypopneas that are followed
    by a scored SpO2 desaturation with baseline-nadir >= 3 within DESAT_WINDOW_S of the event end."""
    if not np.isfinite(sleep_h) or sleep_h <= 0:
        return float("nan")
    ds = ev[ev.concept.str.lower().str.startswith("spo2 desaturation")]
    d0 = ds.start_s.to_numpy()
    drop = (ds.get("drop", pd.Series(dtype=float))).to_numpy() if "drop" in ds else np.array([])
    keep = np.isfinite(drop) & (drop >= DESAT_MIN_DROP) if len(drop) else np.zeros(len(ds), bool)
    d0 = d0[keep]
    r = ev[ev.subtype != ""]
    n = 0
    for s, d in r[["start_s", "duration_s"]].itertuples(index=False):
        if ((d0 >= s) & (d0 <= s + d + DESAT_WINDOW_S)).any():
            n += 1
    return n / sleep_h


def _events_with_drop(mesaid: str) -> pd.DataFrame:
    """parse_events + SpO2 drop (baseline - nadir) for desaturation events (extra XML fields)."""
    import xml.etree.ElementTree as ET

    ev = M.parse_events(mesaid)
    drops = []
    for e in ET.parse(M.xml_path(mesaid)).getroot().iter("ScoredEvent"):
        b, n = e.findtext("SpO2Baseline"), e.findtext("SpO2Nadir")
        drops.append(float(b) - float(n) if b and n else np.nan)
    ev["drop"] = drops
    return ev


def audit_subject(mesaid: str) -> dict:
    psg = M.load_psg(mesaid)
    ev = _events_with_drop(mesaid)
    g = M.label_grid(ev, psg.duration_s)
    sleep_h = M.sleep_hours(g)
    row: dict = {"mesaid": mesaid, "duration_h": psg.duration_s / 3600.0, "sleep_h": sleep_h,
                 "n_stage_events": int((ev.event_type.str.lower().str.startswith("stages")).sum())}
    for c in M.CHANNELS:
        row[f"has_{c}"] = c in psg.channels
        row[f"fs_{c}"] = psg.channels[c].fs if c in psg.channels else np.nan
    cnt = ev.subtype.value_counts()
    for s in ("obstructive", "central", "mixed", "hypopnea"):
        row[f"n_{s}"] = int(cnt.get(s, 0))
    n_ap = row["n_obstructive"] + row["n_central"] + row["n_mixed"]
    n_all = n_ap + row["n_hypopnea"]
    row["n_apnea"] = n_ap
    row["ahi_xml_all"] = n_all / sleep_h if sleep_h and np.isfinite(sleep_h) else np.nan
    row["ai_xml"] = n_ap / sleep_h if sleep_h and np.isfinite(sleep_h) else np.nan
    row["ahi_xml_desat3"] = desat_supported_ahi(ev, sleep_h)
    row["central_frac"] = row["n_central"] / n_ap if n_ap else np.nan
    # class balance on the 30 s grid
    slp = g.stage.isin(["N1", "N2", "N3", "R"])
    row["epochs"] = len(g)
    row["sleep_epochs"] = int(slp.sum())
    for k in ("apnea", "hypopnea", "resp_event"):
        row[f"ep_{k}"] = int(g[k].sum())
        row[f"ep_{k}_sleep"] = int(g.loc[slp, k].sum())
    for s in ("obstructive", "central", "mixed"):
        row[f"ep_dom_{s}"] = int((g.subtype == s).sum())
    # signal quality
    th, ab = psg.channels.get("thor"), psg.channels.get("abdo")
    for name, ch in (("thor", th), ("abdo", ab)):
        for k, v in belt_quality(ch).items():
            row[f"{name}_{k}"] = v
    row["belt_agree_r"] = belt_agreement(th, ab)
    for k, v in flat_or_bad(psg.channels.get("ecg")).items():
        row[f"ecg_{k}"] = v
    for k, v in flat_or_bad(psg.channels.get("spo2"), 50, 100).items():
        row[f"spo2_{k}"] = v
    concepts = Counter(ev.loc[ev.event_type.str.lower().str.startswith("respiratory"), "concept"])
    return {"row": row, "concepts": {f"{k}": v for k, v in concepts.items()}}


def run_subjects() -> tuple[pd.DataFrame, Counter]:
    CACHE.mkdir(parents=True, exist_ok=True)
    rows, concepts = [], Counter()
    ids = M.list_downloaded()
    for i, mid in enumerate(ids):
        cf = CACHE / f"{mid}.json"
        if cf.exists():
            res = json.loads(cf.read_text())
        else:
            log.info("[%d/%d] auditing %s", i + 1, len(ids), mid)
            try:
                res = audit_subject(mid)
            except Exception as e:  # keep going; record the failure
                log.warning("subject %s failed: %r", mid, e)
                res = {"row": {"mesaid": mid, "error": repr(e)}, "concepts": {}}
            cf.write_text(json.dumps(res))
        rows.append(res["row"])
        concepts.update(res["concepts"])
    return pd.DataFrame(rows), concepts


def merge_official(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach NSRR-reported age/sex/AHI; also return the full-cohort table for representativeness."""
    off = pd.read_csv(DATASET_CSV, low_memory=False)
    off["mesaid"] = off.mesaid.astype(int).astype(str).str.zfill(4)
    keep = ["mesaid", "sleepage5c", "gender1", "race1c", "ahi_a0h3", "ahi_a0h3a", "ahi_a0h4", "ahi_a0h4a",
            "ahi_o0h3a", "ahi_c0h3a", "slpprdp5"]
    off = off[[c for c in keep if c in off.columns]]
    return df.merge(off, on="mesaid", how="left"), off


def usable_funnel(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    steps = []
    m = pd.Series(True, index=d.index)

    def step(name, cond):
        nonlocal m
        m = m & cond.fillna(False)
        steps.append((name, int(m.sum())))

    steps.append(("complete subjects (EDF+XML)", len(d)))
    step("ECG + SpO2 + Thor + Abdo all present", d.has_ecg & d.has_spo2 & d.has_thor & d.has_abdo)
    step("XML has sleep-stage events", d.n_stage_events > 0)
    step(f"sleep time >= {MIN_SLEEP_H:g} h", d.sleep_h >= MIN_SLEEP_H)
    step(f"both belts flat_frac < {MAX_FLAT_FRAC}", (d.thor_flat_frac < MAX_FLAT_FRAC) & (d.abdo_flat_frac < MAX_FLAT_FRAC))
    step(f"both belts breathing visible in >= {MIN_GOOD_EPOCH_FRAC:.0%} epochs",
         (d.thor_good_epoch_frac >= MIN_GOOD_EPOCH_FRAC) & (d.abdo_good_epoch_frac >= MIN_GOOD_EPOCH_FRAC))
    step("scored respiratory events present (>=1 apnea or hypopnea)", (d.n_apnea + d.n_hypopnea) > 0)
    f = pd.DataFrame(steps, columns=["criterion (cumulative)", "subjects"])
    return f, m


def _sev(x):
    return pd.cut(x, [-np.inf, 5, 15, 30, np.inf], labels=["<5 normal", "5-15 mild", "15-30 moderate", ">=30 severe"], right=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, concepts = run_subjects()
    df, off_all = merge_official(df)
    df.to_csv(OUT / "subjects.csv", index=False)
    pd.Series(concepts, name="count").sort_values(ascending=False).rename_axis("respiratory_concept").to_csv(OUT / "respiratory_concepts.csv")
    ok = df[df.get("error", pd.Series(index=df.index, dtype=object)).isna()].copy()
    funnel, umask = usable_funnel(ok)
    usable = ok[umask]
    funnel.to_csv(OUT / "usable_funnel.csv", index=False)

    S: dict = {"n_complete": len(df), "n_errors": int(len(df) - len(ok)), "n_usable": int(len(usable))}
    L: list[str] = ["# MESA subset audit", "", f"Subjects audited (EDF+XML complete): **{len(df)}** "
                    f"(errors: {S['n_errors']}). Usable under the strict definition: **{len(usable)}**.", "",
                    "## Usable-subject funnel", "", funnel.to_markdown(index=False), ""]

    # signal presence / fs
    L += ["## Signal presence and native sampling rate", ""]
    for c in M.CHANNELS:
        L.append(f"- `{c}`: present in {int(ok[f'has_{c}'].sum())}/{len(ok)}; fs values {sorted(ok[f'fs_{c}'].dropna().unique().tolist())}")
    L += [f"- recording duration h: median {ok.duration_h.median():.2f} (min {ok.duration_h.min():.2f}, max {ok.duration_h.max():.2f}); "
          f"scored sleep h: median {ok.sleep_h.median():.2f} (min {ok.sleep_h.min():.2f})", ""]

    # events
    tot = {s: int(ok[f"n_{s}"].sum()) for s in ("obstructive", "central", "mixed", "hypopnea")}
    S["event_totals"] = tot
    L += ["## Scored events", "", "Totals across audited subjects: " + ", ".join(f"{k} {v}" for k, v in tot.items()) + ".",
          f"Subjects with >=1 central apnea: {int((ok.n_central > 0).sum())}; with >=5 central: {int((ok.n_central >= 5).sum())}; "
          f"with >=1 obstructive: {int((ok.n_obstructive > 0).sum())}; with any apnea (O/C/M): {int((ok.n_apnea > 0).sum())}.", "",
          "All respiratory event concepts seen in the XMLs (anything unexpected here is NOT parsed into a subtype):", "",
          pd.Series(concepts).sort_values(ascending=False).rename_axis("concept").reset_index(name="count").to_markdown(index=False), ""]

    # class balance
    for scope, sfx in (("all epochs", ""), ("sleep epochs only", "_sleep")):
        n = ok["epochs"].sum() if sfx == "" else ok["sleep_epochs"].sum()
        L.append(f"- Epoch class balance, {scope} (30 s, n={int(n)}): " + ", ".join(
            f"{k} {ok[f'ep_{k}{sfx}'].sum() / n:.1%}" for k in ("apnea", "hypopnea", "resp_event")))
    S["epoch_balance_sleep"] = {k: float(ok[f"ep_{k}_sleep"].sum() / ok.sleep_epochs.sum()) for k in ("apnea", "hypopnea", "resp_event")}
    L += ["", "Note: 'apnea' (obstructive/central/mixed) is rare next to hypopnea in MESA; an apnea-only label is a highly imbalanced target.", ""]

    # AHI distribution + label-definition cross-check
    L += ["## AHI distribution", ""]
    for col, nm in (("ahi_xml_all", "XML-derived, all scored events / scored sleep h"),
                    ("ahi_xml_desat3", "XML-derived, desat(>=3%)-supported"),
                    ("ahi_a0h3a", "NSRR official ahi_a0h3a (3% or arousal)"), ("ahi_a0h4", "NSRR official ahi_a0h4 (4%)")):
        v = ok[col].dropna() if col in ok else pd.Series(dtype=float)
        if len(v):
            b = _sev(v).value_counts().reindex(["<5 normal", "5-15 mild", "15-30 moderate", ">=30 severe"])
            L.append(f"- {nm}: n={len(v)}, median {v.median():.1f}, IQR {v.quantile(.25):.1f}-{v.quantile(.75):.1f}, max {v.max():.1f}; "
                     + ", ".join(f"{k} {int(x)}" for k, x in b.items()))
    both = ok[["ahi_xml_all", "ahi_xml_desat3", "ahi_a0h3", "ahi_a0h3a"]].dropna()
    if len(both) > 3:
        S["ahi_corr"] = {a: float(both.ahi_xml_all.corr(both[a])) for a in ("ahi_a0h3", "ahi_a0h3a")}
        S["ahi_corr"]["desat3_vs_a0h3"] = float(both.ahi_xml_desat3.corr(both.ahi_a0h3))
        S["ahi_ratio_median"] = {"xml_all/official_a0h3a": float((both.ahi_xml_all / both.ahi_a0h3a.replace(0, np.nan)).median()),
                                 "xml_desat3/official_a0h3": float((both.ahi_xml_desat3 / both.ahi_a0h3.replace(0, np.nan)).median())}
        L += ["", f"Cross-check vs NSRR official AHI (n={len(both)}): Pearson r(xml_all, a0h3)={S['ahi_corr']['ahi_a0h3']:.3f}, "
              f"r(xml_all, a0h3a)={S['ahi_corr']['ahi_a0h3a']:.3f}, r(xml_desat3, a0h3)={S['ahi_corr']['desat3_vs_a0h3']:.3f}. "
              f"Median ratio xml_all/a0h3a = {S['ahi_ratio_median']['xml_all/official_a0h3a']:.2f}; "
              f"xml_desat3/a0h3 = {S['ahi_ratio_median']['xml_desat3/official_a0h3']:.2f}."]
    if "slpprdp5" in ok:
        d = (ok.sleep_h * 60 - ok.slpprdp5).dropna()
        if len(d):
            S["sleep_time_diff_min"] = {"median_abs": float(d.abs().median()), "max_abs": float(d.abs().max())}
            L.append(f"Sleep-time check (stage-derived vs NSRR slpprdp5): median |diff| {d.abs().median():.1f} min, max {d.abs().max():.1f} min (n={len(d)}).")
    L.append("")

    # belts
    L += ["## Thoracic vs abdominal belt quality", ""]
    bq = pd.DataFrame({b: {"flat_frac median": ok[f"{b}_flat_frac"].median(), "good_epoch_frac median": ok[f"{b}_good_epoch_frac"].median(),
                           "good_epoch_frac<0.5 (subjects)": int((ok[f"{b}_good_epoch_frac"] < 0.5).sum()),
                           "clip_frac median": ok[f"{b}_clip_frac"].median(), "breath rate median (bpm)": ok[f"{b}_breath_rate_bpm_med"].median()}
                       for b in ("thor", "abdo")}).round(3)
    L += [bq.to_markdown(), ""]
    better = np.sign(ok.thor_good_epoch_frac - ok.abdo_good_epoch_frac)
    L.append(f"Per subject, cleaner belt by breathing-visible epoch fraction: Thor {int((better > 0).sum())}, Abdo {int((better < 0).sum())}, tie {int((better == 0).sum())}. "
             f"Median |r| between belts (0.1-0.6 Hz, per epoch): {ok.belt_agree_r.median():.2f}.")
    S["belts"] = json.loads(bq.to_json())
    L.append("")

    # elderly-cohort prevalence sanity check
    L += ["## Apnea prevalence sanity check (elderly cohort)", ""]
    if "sleepage5c" in ok:
        L.append(f"Subset age at sleep exam: median {ok.sleepage5c.median():.0f} (range {ok.sleepage5c.min():.0f}-{ok.sleepage5c.max():.0f}); "
                 f"male fraction {(ok.gender1 == 1).mean():.2f}.")
    rows = []
    for nm, t in (("audited subset", ok), ("full MESA sleep cohort (NSRR)", off_all)):
        for col, rule in (("ahi_a0h3a", "3% or arousal"), ("ahi_a0h4", "4% desat")):
            v = t[col].dropna() if col in t else pd.Series(dtype=float)
            if len(v):
                rows.append({"group": nm, "hypopnea rule": rule, "n": len(v), "AHI>=5": f"{(v >= 5).mean():.0%}",
                             "AHI>=15": f"{(v >= 15).mean():.0%}", "AHI>=30": f"{(v >= 30).mean():.0%}", "median AHI": round(v.median(), 1)})
    if rows:
        L += [pd.DataFrame(rows).to_markdown(index=False), "",
              "Read-out: prevalence depends heavily on the hypopnea rule (3%-or-arousal is much more liberal than 4%), so quote the rule with any number. "
              "The subset row vs the full-cohort row (same rule) shows whether the sampled subset is representative of MESA; "
              "the full-cohort row is NSRR's own reported AHI, i.e. the reference this audit is checked against."]
        S["prevalence"] = rows
    L.append("")

    # usable subset detail
    if len(usable):
        L += ["## Usable subset", "", f"n={len(usable)}. Epoch balance (sleep epochs): " + ", ".join(
            f"{k} {usable[f'ep_{k}_sleep'].sum() / usable.sleep_epochs.sum():.1%}" for k in ("apnea", "hypopnea", "resp_event")) + ".",
            f"Usable subjects with >=5 central apneas: {int((usable.n_central >= 5).sum())}; with >=5 obstructive apneas: {int((usable.n_obstructive >= 5).sum())}.", ""]

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for col, nm in (("ahi_xml_all", "XML all events"), ("ahi_a0h3a", "NSRR official a0h3a")):
        if col in ok and ok[col].notna().any():
            ax[0].hist(ok[col].dropna().clip(upper=100), bins=np.arange(0, 105, 5), alpha=.55, label=nm)
    ax[0].set(xlabel="AHI (events/h, clipped at 100)", ylabel="subjects", title="AHI distribution"); ax[0].legend(frameon=False)
    if len(both):
        ax[1].scatter(both.ahi_a0h3a, both.ahi_xml_all, s=10, alpha=.6)
        m_ = float(max(both.ahi_a0h3a.max(), both.ahi_xml_all.max())); ax[1].plot([0, m_], [0, m_], "k--", lw=.8)
        ax[1].set(xlabel="NSRR official ahi_a0h3a", ylabel="XML-derived (all scored events)", title="Label-definition check")
    ax[2].scatter(ok.thor_good_epoch_frac, ok.abdo_good_epoch_frac, s=10, alpha=.6)
    ax[2].plot([0, 1], [0, 1], "k--", lw=.8)
    ax[2].set(xlabel="Thor: breathing-visible epoch frac", ylabel="Abdo: breathing-visible epoch frac", title="Belt quality per subject")
    fig.tight_layout(); fig.savefig(OUT / "audit_overview.png", dpi=130); plt.close(fig)

    (OUT / "summary.json").write_text(json.dumps(S, indent=2, default=float))
    (OUT / "summary.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
