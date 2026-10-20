"""Train the deployable Track-B breath-hold models on ALL labelled windows of session 1 (+ augmentation + synthetic, per --config) and write the dashboard bundle.
Session-1 outputs of these models are IN-SAMPLE (training demonstrations); out-of-sample evidence is only the leave-one-event-out table in results/inhouse_bh/.
    python3 -m src.inhouse_bh.train_final [--config aug+synth] [--seed 0]"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np, joblib, torch
os.environ.setdefault("SYNTH_SPEC_DIR", "results/synth/spec_noS1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.inhouse_bh import common as Cm, synthetic as Syn
from src.inhouse_bh import run_experiments as R

OUT = Path("results/inhouse_bh/deployed")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="real"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n-synth-rec", type=int, default=100); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); x, dur, fs = Cm.trace32(Cm.PATHS["S1_ECGTILE"]); w, _ = Cm.labelled_windows("S1_ECGTILE", x)
    X = Cm.window_array(x, w.i0.values, w.scale.values); y = (w.label == "voluntary_breath_hold").astype(int).values
    syn_all = Syn.synthetic_windows(a.n_synth_rec, 4242, return_stats=True); syn = syn_all[:2]; syn_stats = syn_all[3]; seed = a.seed; use_aug = "aug" in a.config; use_syn = "synth" in a.config
    Xr, yr = R.expand(X, y, seed) if use_aug else (X, y); n_aug = len(Xr) - len(X)
    Xt, yt = (np.concatenate([Xr, syn[0]]), np.concatenate([yr, syn[1]])) if use_syn else (Xr, yr)
    sk = R.sk_models(seed)
    for m in sk.values(): m.fit(R.F(Xt), yt)
    cnn = R.fit_cnn(np.concatenate([X, syn[0]]) if use_syn else X, np.concatenate([y, syn[1]]) if use_syn else y, seed, aug=use_aug)
    joblib.dump(sk, OUT / "sklearn_models.joblib"); torch.save(cnn.state_dict(), OUT / "cnn.pt")
    meta = dict(task="voluntary breath-hold-like low-effort windows (NOT sleep apnea)", config=a.config, seed=seed, trained=time.strftime("%Y-%m-%d %H:%M"),
                training_data=dict(source_recording="S1_ECGTILE (one volunteer, 2026-10-07; sha256 " + hashlib.sha256(Cm.PATHS["S1_ECGTILE"].read_bytes()).hexdigest()[:16] + ")",
                                   original_windows=int(len(X)), original_hold=int(y.sum()), original_nonhold=int((1 - y).sum()), augmented_windows=int(n_aug if use_aug else 0),
                                   synthetic_windows=int(len(syn[0]) if use_syn else 0), synthetic_qc=syn_stats, rejected_windows="uncertain-boundary windows are never created (see annotations.csv)", events=7),
                window_s=Cm.WIN_S, stride_s=Cm.STRIDE_S, thr=0.5, in_sample_recordings=["user_MukuBreathHold_ECG_07_10_2026.csv"],
                caveat="Trained on session 1: its outputs are in-sample. Out-of-sample estimate = leave-one-event-out (results/inhouse_bh/loeo_summary.csv), one volunteer, 7 events.")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1)); print(json.dumps(meta, indent=1))
    # pin in docs/checkpoints.json
    ck = json.loads(Path("docs/checkpoints.json").read_text())
    for name, f, rf in (("bh_sklearn", "sklearn_models.joblib", "breath-hold detector (LR/HGB)"), ("bh_cnn", "cnn.pt", "breath-hold detector (CNN)")):
        p = OUT / f; ck[name] = dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size, required_for=rf, fix="python3 -m src.inhouse_bh.train_final (needs data/recordings/breath_hold/… session-1 file)")
    Path("docs/checkpoints.json").write_text(json.dumps(ck, indent=1))


if __name__ == "__main__":
    main()
