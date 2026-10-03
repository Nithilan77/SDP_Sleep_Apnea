"""Regenerates the handoff figures from results CSVs. Run from project root.
Outputs only aggregate metrics -- no subject identifiers, no raw data."""
import json, shutil
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/presentation_handoff/figures/"
C = {"base": "#8a8f98", "main": "#1f6feb", "good": "#2da44e", "bad": "#d1242f", "warn": "#d4a72c"}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

def mm(path):
    d = pd.read_csv(path); y, p = d.y_true, d.y_pred
    return (y == p).mean(), p[y == 1].mean(), 1 - p[y == 0].mean()

# 1. Phase 3 progression (per-minute LOSO) -- recomputed from prediction CSVs
steps = [("Phase 2\nclassical", "results/phase2_cvhr_baseline/per_minute_predictions.csv"),
         ("v1 CNN\nRR only", "results/phase3_cnn/per_minute_predictions.csv"),
         ("v2 CNN\nRR+amp, 1.5x wt", "results/phase3_cnn_v2/per_minute_predictions.csv"),
         ("FINAL\nRR+amp", "results/phase3_cnn/final_per_minute_predictions.csv")]
vals = np.array([mm(p) for _, p in steps])
fig, ax = plt.subplots(figsize=(9, 5)); w = 0.26; x = np.arange(len(steps))
for i, (lab, col) in enumerate([("Accuracy", C["main"]), ("Sensitivity", C["good"]), ("Specificity", C["base"])]):
    b = ax.bar(x + (i - 1) * w, vals[:, i], w, label=lab, color=col)
    ax.bar_label(b, fmt="%.3f", fontsize=8, padding=2)
ax.set_xticks(x); ax.set_xticklabels([s for s, _ in steps]); ax.set_ylim(0.5, 0.95)
ax.set_ylabel("per-minute, leave-one-subject-out"); ax.legend(ncol=3, loc="upper left", frameon=False)
ax.set_title("ECG branch: classical baseline to 1D-CNN (PhysioNet Apnea-ECG, 35 records)")
fig.tight_layout(); fig.savefig(OUT + "phase3_progression.png", dpi=200); plt.close()

# 2. Feature study (8 methods)
m = pd.read_csv("results/feature_study/master_results.csv")
short = ["1 RR", "2 RR+amp\n(adopted)", "3 Time\nHRV", "4 Freq\nHRV", "5 Non-\nlinear", "6 RR+QRS\narea", "7 CWT\nscalogram", "8 ECG\nspectrogram"]
cols = [C["good"] if i == 1 else C["main"] for i in range(8)]
fig, ax = plt.subplots(figsize=(10, 5))
b = ax.bar(range(8), m.accuracy, color=cols); ax.bar_label(b, fmt="%.3f", fontsize=9, padding=2)
ax.axhline(0.711, color=C["bad"], ls="--", lw=1); ax.text(7.45, 0.716, "Phase 2 floor 0.711", color=C["bad"], ha="right", fontsize=9)
ax.set_xticks(range(8)); ax.set_xticklabels(short, fontsize=9); ax.set_ylim(0.4, 0.9)
ax.set_ylabel("per-minute accuracy (LOSO)"); ax.set_title("Eight ECG feature representations, identical LOSO protocol")
fig.tight_layout(); fig.savefig(OUT + "feature_study_comparison.png", dpi=200); plt.close()

# 3. Head-to-head (AUROC / AUPRC) incl. ablations + ceiling
h = pd.read_csv("results/mesa/canet/head_to_head.csv")
lab = ["ECG-only\nbaseline", "CANet\n(ECG+effort)", "+SpO2\nceiling*", "Effort\nonly", "ECG+effort\nconcat"]
col = [C["base"], C["main"], C["warn"], C["base"], C["base"]]
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
for ax, k, t, chance in [(axs[0], "auroc", "AUROC (chance 0.5)", 0.5), (axs[1], "auprc", "AUPRC (chance 0.254)", 0.254)]:
    b = ax.bar(range(5), h[k], color=col); ax.bar_label(b, fmt="%.3f", fontsize=9, padding=2)
    ax.axhline(chance, color="k", ls=":", lw=1); ax.set_xticks(range(5)); ax.set_xticklabels(lab, fontsize=9)
    ax.set_title(t); ax.set_ylim(0, h[k].max() * 1.15)
fig.suptitle("MESA, 220 subjects, subject-independent 5-fold CV   (*SpO2 not available on our wearable)", fontsize=11)
fig.tight_layout(); fig.savefig(OUT + "canet_vs_baseline.png", dpi=200); plt.close()

# 4. Per-class sensitivity gain
r = h.iloc[[0, 1]]; cls = ["sens_obstructive", "sens_central", "sens_hypopnea", "spec_normal"]
nm = ["Obstructive\napnea", "Central\napnea", "Hypopnea", "Normal epochs\n(specificity)"]
fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(4); w = 0.36
b1 = ax.bar(x - w/2, r.iloc[0][cls].astype(float), w, color=C["base"], label="ECG-only baseline")
b2 = ax.bar(x + w/2, r.iloc[1][cls].astype(float), w, color=C["main"], label="CANet (ECG + effort)")
ax.bar_label(b1, fmt="%.2f", fontsize=9, padding=2); ax.bar_label(b2, fmt="%.2f", fontsize=9, padding=2)
ax.set_xticks(x); ax.set_xticklabels(nm); ax.set_ylim(0, 1.15); ax.set_ylabel("fraction correctly flagged / kept")
ax.legend(frameon=False, loc="upper right", ncol=2); ax.set_title("Per-class gain (each model at its own validation-chosen threshold)")
fig.tight_layout(); fig.savefig(OUT + "per_class_gain.png", dpi=200); plt.close()

# 5. Negative-control summary (model level)
s = pd.read_csv("results/inhouse_canet/negative_control/summary_by_model.csv")
ref = pd.read_csv("results/inhouse_canet/negative_control/mesa_out_of_fold_reference.csv")
def g(model, src): return s[(s.model == model) & (s.effort_source == src)].frac_flagged_median.iloc[0]
rows = [("ECG-only", g("ecg_only", "-"), C["base"]), ("Effort-only\n(accelerometer)", g("resp_only", "mag"), C["bad"]),
        ("CANet\n(accelerometer)", g("headline", "mag"), C["bad"]),
        ("Effort-only\n(white noise)", g("resp_only", "diag: effort=white noise"), C["good"])]
mesa = ref[(ref.model == "headline") & ref.reference.str.contains("low-AHI")].frac_flagged_median.iloc[0]
fig, ax = plt.subplots(figsize=(9, 5))
b = ax.bar(range(4), [r_[1] for r_ in rows], color=[r_[2] for r_ in rows]); ax.bar_label(b, fmt="%.2f", padding=2)
ax.axhline(mesa, color="k", ls="--", lw=1, label=f"MESA low-AHI median, CANet: {mesa:.2f}"); ax.legend(frameon=False, loc="upper right")
ax.set_xticks(range(4)); ax.set_xticklabels([r_[0] for r_ in rows]); ax.set_ylim(0, 1.1)
ax.set_ylabel("fraction of epochs flagged (median of 10 nights)")
ax.set_title("Healthy in-house nights: flag rate (a transfer failure, NOT apnea accuracy)")
fig.tight_layout(); fig.savefig(OUT + "inhouse_negative_control_summary.png", dpi=200); plt.close()

# 6. Domain-gap summary: classifier AUROC
d = pd.read_csv("results/inhouse_canet/domain_gap/domain_classifier_auroc.csv")
d = d[d.features.str.contains("handcrafted")]
lab = ["Control:\nMESA vs MESA", "MESA belt vs\naccel magnitude", "MESA belt vs\naccel principal axis"]
fig, ax = plt.subplots(figsize=(8, 5))
b = ax.bar(range(3), d.auroc, color=[C["good"], C["bad"], C["bad"]]); ax.bar_label(b, fmt="%.2f", padding=2)
ax.axhline(0.5, color="k", ls=":", lw=1); ax.set_xticks(range(3)); ax.set_xticklabels(lab); ax.set_ylim(0, 1.1)
ax.set_ylabel("AUROC of belt-vs-accelerometer classifier"); ax.set_title("Can a classifier tell belt from accelerometer? (0.5 = identical)")
fig.tight_layout(); fig.savefig(OUT + "inhouse_domain_gap_separability.png", dpi=200); plt.close()

# existing, already-anonymised PNGs
for src, dst in [("results/mesa/audit/audit_overview.png", "mesa_audit_overview.png"),
                 ("results/mesa/ecg_baseline/ecg_baseline_overview.png", "mesa_ecg_baseline_overview.png"),
                 ("results/inhouse_canet/domain_gap/spectrum_and_breathing_features.png", "inhouse_domain_gap_spectrum.png"),
                 ("results/inhouse_canet/domain_gap/amplitude_and_shape.png", "inhouse_domain_gap_amplitude_shape.png"),
                 ("results/inhouse_canet/negative_control/healthy_flagged_fraction.png", "inhouse_negative_control_per_night.png")]:
    shutil.copy(src, OUT + dst)

# funnel chart
f = pd.read_csv("results/mesa/audit/usable_funnel.csv")
fig, ax = plt.subplots(figsize=(10, 4.5))
lbl = ["Complete (EDF+XML)", "ECG+SpO2+Thor+Abdo present", "Sleep stages in XML", "Sleep >= 4 h", "Both belts not flat", "Breathing visible >=50% epochs", "Has scored event"]
b = ax.barh(range(len(f))[::-1], f.subjects, color=C["main"]); ax.bar_label(b, padding=3)
ax.set_yticks(range(len(f))[::-1]); ax.set_yticklabels(lbl, fontsize=10); ax.set_xlim(0, 285)
ax.set_xlabel("subjects remaining (cumulative)"); ax.set_title("MESA usable-subject funnel: 250 downloaded -> 220 usable")
fig.tight_layout(); fig.savefig(OUT + "mesa_usable_funnel.png", dpi=200); plt.close()
print("ok")
