"""Phase 3b-v2: full 35-fold LOSO for the per-architecture config chosen in src.ecg.rnn_lstm_select (validation-selected;
see that file's caveat). Also writes deployable checkpoints trained on all 35 records. Does NOT overwrite the primary results."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ecg.rnn_lstm import *  # noqa
OUT2 = Path("results/phase3_rnn_lstm_v2"); OUT2.mkdir(parents=True, exist_ok=True)
best = json.load(open("results/phase3_rnn_lstm/selection.json"))["best"]
ds = load_or_build_sequences(); mask = restrict_to_phase2_minutes(ds); summary = {}
for arch, c in best.items():
    kw = dict(epochs=c["epochs"], hidden=c["hidden"], pool=c["pool"], bidir=c["bidir"])
    pr = run_loso(arch, ds, mask, **kw); pr.to_csv(OUT2 / f"{arch}_per_minute_predictions.csv", index=False)
    m = binary_metrics(pr["y_true"], pr["y_pred"]); rec = per_recording_evaluation(pr)
    print(f"\n=== {arch.upper()} v2 LOSO {kw} ===\n{format_metrics(m)}\nper-recording {int(rec.correct.sum())}/{len(rec)}", flush=True)
    mm = RecurrentApnea(arch, hidden=c["hidden"], pool=c["pool"], bidir=c["bidir"])
    summary[arch] = dict(config=kw, metrics={k: float(v) for k, v in m.items()}, per_recording_correct=int(rec.correct.sum()), params=n_params(mm))
    json.dump(summary, open(OUT2 / "summary.json", "w"), indent=1)
    seq, lab = ds["sequence"][mask], (ds["label"][mask] == "A").astype(np.float32)
    mu = seq.mean(axis=(0, 2), keepdims=True); sd = seq.std(axis=(0, 2), keepdims=True)
    mdl = train(arch, (seq - mu) / sd, lab, **kw); d = OUT2 / f"deployable_{arch}"; d.mkdir(exist_ok=True)
    torch.save(mdl.cpu().state_dict(), d / "model_state_dict.pt")
    json.dump(dict(mean=mu.flatten().tolist(), std=sd.flatten().tolist(), arch=arch, hidden=c["hidden"], pool=c["pool"], bidir=c["bidir"]), open(d / "norm_stats.json", "w"))
