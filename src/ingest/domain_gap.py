"""
Domain-gap analysis: PhysioNet (training data) vs our in-house QVAR recordings.

Quantifies how far the SensorTile's QVAR-ECG signal sits from what the
deployable ECG-branch model (results/phase3_cnn/deployable/) was trained on,
for the four channels that feed the model or its documented EDR fallback:

    - RR-interval (s)     -- time-based, expected to transfer cleanly (rule 6)
    - instantaneous HR (bpm)
    - R-peak amplitude    -- the flagged transfer risk (PhysioNet-calibrated
                              ECG vs QVAR's electrostatic-sensor raw units)
    - QRS-area (EDR proxy, method 6's fallback if amplitude doesn't transfer)

Three groups compared:
    - physionet: all 35 labelled PhysioNet Apnea-ECG records (the model's
      training distribution)
    - inhouse_plausible: the 13 in-house files flagged PLAUSIBLE in
      results/inhouse_validation/quality_report.csv
    - inhouse_noisy: user_0802-anagesh_ecg_000.txt (flagged NOISY_ECG),
      reported separately so its effect on the gap is visible but doesn't
      pollute the "healthy hardware" baseline.

Excludes user_1612bhavi_ecg_000.txt (flagged NO_BEATS_DETECTED / too short --
an aborted capture with 1 beat, no distribution to speak of).

This is a DISTRIBUTION COMPARISON ONLY. No accuracy/sensitivity claims.

Usage:
    python src/ingest/domain_gap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ecg.qrs_area import compute_qrs_area  # noqa: E402
from src.ecg.rpeaks import detect_rpeaks  # noqa: E402
from src.eval.splits import list_labelled_records  # noqa: E402
from src.ingest.physionet import load_record  # noqa: E402
from src.ingest.sensortile import load_sensortile  # noqa: E402

RESULTS_DIR = Path("results/inhouse_validation/domain_gap")
INHOUSE_DIR = Path("data/recordings/inhouse_ecg")
QUALITY_CSV = Path("results/inhouse_validation/quality_report.csv")

METRICS = ["rr_interval_s", "hr_inst_bpm", "r_amplitude", "qrs_area"]
METRIC_LABELS = {
    "rr_interval_s": "RR-interval (s)",
    "hr_inst_bpm": "Instantaneous HR (bpm)",
    "r_amplitude": "R-peak amplitude (cleaned-signal units)",
    "qrs_area": "QRS-area (cleaned-signal-units * s)",
}


def _per_beat_metrics(signal: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """Run R-peak detection + QRS-area on a raw signal, return per-beat arrays
    aligned the same way rr_amp_sequence.py aligns RR and amplitude: RR/HR/
    QRS-area all indexed to beats 1..n-1 (amplitude at the peak ENDING each
    RR interval), so all four metrics have matching length."""
    cleaned = nk.ecg_clean(signal, sampling_rate=int(round(fs)))
    _, info = nk.ecg_peaks(cleaned, sampling_rate=int(round(fs)))
    r_peaks = np.asarray(info["ECG_R_Peaks"])

    if len(r_peaks) < 2:
        empty = np.array([], dtype=np.float64)
        return {"rr_interval_s": empty, "hr_inst_bpm": empty,
                "r_amplitude": empty, "qrs_area": empty}

    rr_s = np.diff(r_peaks) / fs
    amp_all = cleaned[r_peaks]
    qrs_all = compute_qrs_area(cleaned, r_peaks, fs)

    return {
        "rr_interval_s": rr_s,
        "hr_inst_bpm": 60.0 / rr_s,
        "r_amplitude": amp_all[1:],   # aligned to the peak ending each RR interval
        "qrs_area": qrs_all[1:],
    }


def _collect_physionet() -> dict[str, np.ndarray]:
    names = list_labelled_records()
    print(f"PhysioNet: {len(names)} labelled records")
    per_metric: dict[str, list[np.ndarray]] = {m: [] for m in METRICS}
    for name in names:
        rec = load_record(name)
        vals = _per_beat_metrics(rec.signal, rec.fs)
        for m in METRICS:
            per_metric[m].append(vals[m])
        print(f"  {name}: {len(vals['rr_interval_s'])} beats")
    return {m: np.concatenate(v) for m, v in per_metric.items()}


def _collect_inhouse(files: list[str], label: str) -> dict[str, np.ndarray]:
    print(f"In-house ({label}): {len(files)} file(s)")
    per_metric: dict[str, list[np.ndarray]] = {m: [] for m in METRICS}
    for fname in files:
        rec = load_sensortile(INHOUSE_DIR / fname)
        vals = _per_beat_metrics(rec.qvar, rec.fs_measured)
        for m in METRICS:
            per_metric[m].append(vals[m])
        print(f"  {fname}: {len(vals['rr_interval_s'])} beats")
    return {m: np.concatenate(v) if v else np.array([]) for m, v in per_metric.items()}


def _summary_row(group: str, metric: str, values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"group": group, "metric": metric, "n": 0, "mean": np.nan, "std": np.nan,
                "median": np.nan, "q25": np.nan, "q75": np.nan, "iqr": np.nan}
    return {
        "group": group, "metric": metric, "n": len(values),
        "mean": float(np.mean(values)), "std": float(np.std(values)),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)), "q75": float(np.percentile(values, 75)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
    }


def _plot_metric(metric: str, groups: dict[str, np.ndarray], out_path: Path, logx: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"physionet": "#4C72B0", "inhouse_plausible": "#DD8452", "inhouse_noisy": "#55A868"}
    for group, values in groups.items():
        if len(values) == 0:
            continue
        v = values[np.isfinite(values)]
        if logx:
            v = v[v > 0]
            bins = np.logspace(np.log10(max(v.min(), 1e-9)), np.log10(v.max()), 60)
        else:
            bins = 60
        ax.hist(v, bins=bins, density=True, alpha=0.5, label=f"{group} (n={len(v)})",
                color=colors.get(group))
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(METRIC_LABELS[metric])
    ax.set_ylabel("density")
    ax.set_title(f"{METRIC_LABELS[metric]}: PhysioNet vs in-house QVAR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run_domain_gap() -> pd.DataFrame:
    qdf = pd.read_csv(QUALITY_CSV)
    plausible_files = qdf.loc[qdf["plausibility_flag"] == "PLAUSIBLE", "file"].tolist()
    noisy_files = qdf.loc[qdf["plausibility_flag"] == "NOISY_ECG", "file"].tolist()
    print(f"Plausible in-house files ({len(plausible_files)}): {plausible_files}")
    print(f"Noisy in-house files ({len(noisy_files)}): {noisy_files}")

    physionet = _collect_physionet()
    inhouse_plausible = _collect_inhouse(plausible_files, "plausible")
    inhouse_noisy = _collect_inhouse(noisy_files, "noisy") if noisy_files else {m: np.array([]) for m in METRICS}

    groups = {"physionet": physionet, "inhouse_plausible": inhouse_plausible, "inhouse_noisy": inhouse_noisy}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for metric in METRICS:
        for group_name, group_data in groups.items():
            rows.append(_summary_row(group_name, metric, group_data[metric]))
    summary = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "domain_gap_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    for metric in METRICS:
        logx = metric in ("r_amplitude", "qrs_area")
        out_path = RESULTS_DIR / f"hist_{metric}.png"
        _plot_metric(metric, {g: groups[g][metric] for g in groups}, out_path, logx=logx)
        print(f"Saved {out_path}")

    _write_readme(summary, RESULTS_DIR / "README.md")
    return summary


def _ratio_str(pn_val: float, ih_val: float) -> str:
    if not np.isfinite(pn_val) or not np.isfinite(ih_val) or pn_val == 0:
        return "n/a"
    return f"{ih_val / pn_val:.2f}x"


def _write_readme(summary: pd.DataFrame, path: Path) -> None:
    def get(group, metric, col):
        row = summary[(summary["group"] == group) & (summary["metric"] == metric)]
        return float(row[col].iloc[0]) if len(row) else float("nan")

    lines = [
        "# Domain-gap analysis: PhysioNet vs in-house QVAR",
        "",
        "**Distribution comparison only -- no accuracy/sensitivity claims.** Groups:",
        "`physionet` = all 35 labelled Apnea-ECG records (the deployable model's training",
        "distribution). `inhouse_plausible` = the 13 in-house files flagged PLAUSIBLE in",
        "the quality report. `inhouse_noisy` = user_0802-anagesh_ecg_000.txt (flagged",
        "NOISY_ECG), reported separately. user_1612bhavi_ecg_000.txt (too short, 1 beat)",
        "is excluded entirely.",
        "",
        "## Summary table",
        "",
        summary.to_markdown(index=False),
        "",
        "## Reading the gap",
        "",
    ]

    for metric in METRICS:
        pn_mean = get("physionet", metric, "mean")
        pn_std = get("physionet", metric, "std")
        ih_mean = get("inhouse_plausible", metric, "mean")
        ih_std = get("inhouse_plausible", metric, "std")
        mean_ratio = _ratio_str(pn_mean, ih_mean)
        std_ratio = _ratio_str(pn_std, ih_std)
        lines.append(
            f"- **{METRIC_LABELS[metric]}**: PhysioNet mean={pn_mean:.4g} (std={pn_std:.4g}) "
            f"vs in-house-plausible mean={ih_mean:.4g} (std={ih_std:.4g}) "
            f"-> mean ratio {mean_ratio}, std ratio {std_ratio}."
        )

    rr_mean_ratio = float(get("inhouse_plausible", "rr_interval_s", "mean")) / \
        float(get("physionet", "rr_interval_s", "mean"))
    amp_mean_ratio = float(get("inhouse_plausible", "r_amplitude", "mean")) / \
        float(get("physionet", "r_amplitude", "mean")) if get("physionet", "r_amplitude", "mean") != 0 else float("nan")

    lines += [
        "",
        "## Plain-language verdict",
        "",
        f"- **RR-interval**: mean ratio {rr_mean_ratio:.2f}x (PhysioNet subjects are older with "
        "clinical apnea/borderline/control mix; our subjects are healthy 19-20yo, so some HR "
        "difference is expected and NOT itself a hardware artifact). Distribution shape "
        "(std/IQR relative to mean) is the more informative check -- see the histogram "
        "(`hist_rr_interval_s.png`) and table above for whether the spread is comparable.",
        f"- **R-peak amplitude**: mean ratio {amp_mean_ratio:.2f}x. This is the flagged transfer "
        "risk from the deployable model's README -- PhysioNet amplitude is on a calibrated-ECG "
        "scale, QVAR is an uncalibrated electrostatic-sensor scale after the same `neurokit2` "
        "cleaning. A large ratio here (order-of-magnitude or more) means the frozen model's "
        "`norm_stats.json` (fit on PhysioNet's amplitude scale) will NOT put QVAR amplitudes in "
        "the range the model was trained on -- expect the amplitude channel to look like an "
        "extreme, out-of-distribution outlier to the model regardless of any true respiration "
        "modulation in it.",
        "- **QRS-area**: same PhysioNet-vs-QVAR scale question as amplitude, since it's built "
        "from the same cleaned-signal units. If this ratio is similarly large, QRS-area is NOT "
        "an automatic fix for the amplitude problem -- it would need its own re-normalization "
        "against QVAR's own scale, not a drop-in replacement using PhysioNet's stats.",
        "- **Bottom line**: RR-interval, being time-based (seconds), is the one channel this "
        "project's rule 6 expects to transfer without rescaling. Amplitude and QRS-area are both "
        "raw-signal-unit metrics and inherit whatever scale mismatch exists between a calibrated "
        "ECG amplifier and QVAR's electrostatic front-end -- treat any large ratio above as "
        "confirming, not just flagging, that risk.",
        "- **Noisy-file effect**: compare `inhouse_noisy` (anagesh) against `inhouse_plausible` "
        "in the table -- if noisy-file std/IQR is visibly wider for the same metric, that's the "
        "1.09% implausible-RR translating into distribution spread, isolated from the plausible "
        "group's own numbers.",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    df = run_domain_gap()
    print("\n=== SUMMARY ===")
    print(df.to_string(index=False))
