"""Reproducible synthetic SensorTile-format dataset generator.

    python3 -m src.synth.gen --size pilot  --out data/synthetic/pilot  [--workers 8]
    python3 -m src.synth.gen --n 100 --minutes 30 --out data/synthetic/custom --fmt csv|npz --base-seed 5000

Sizes (configurable): pilot 20 x 10 min, medium 100 x 30 min, large 400 x 30 min.  Output goes under data/synthetic/ (gitignored).
Each recording: <id>.csv|npz, <id>.truth.json (ground-truth event boundaries from the schedule), and a line in manifest.jsonl.
Recording i uses seed = base_seed + i, so any recording can be regenerated alone and is bit-identical.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.synth.render import sample_profile, render, write_csv, COLUMNS, GEN_VERSION, MG_PER_LSB, DPS_PER_LSB
from src.synth.validate import validate

SIZES = {"pilot": (20, 10), "medium": (100, 30), "large": (400, 30)}
HOLD_D = [5, 10, 15, 20, 30, 45, 60]


def schedule(rng, dur, scenario):
    ev = []
    if scenario == "holds":                                  # awake voluntary holds (protocol-like)
        t = rng.uniform(60, 300)
        while True:
            d = float(rng.choice(HOLD_D) + rng.uniform(-0.5, 0.5))
            if t + d > dur - 150: break
            ev.append((t, t + d, 1)); t += d + max(60.0, 3 * d) * rng.uniform(1.0, 1.4)
    elif scenario == "apnea":                                 # simulated sleep-apnea events (central/obstructive)
        lam = rng.uniform(15, 60) / 3600; t = rng.uniform(30, 120)
        while t < dur - 60:
            d = rng.uniform(10, 45); ev.append((t, t + d, int(rng.choice([2, 3])))); t += d + max(10, rng.exponential(1 / lam - 27))
    return ev


def make(i: int, base_seed: int, minutes: float, fmt: str, out: Path):
    seed = base_seed + i; rng = np.random.default_rng(seed); dur = minutes * 60
    scenario = str(rng.choice(["holds", "apnea", "baseline"], p=[.4, .3, .3])); tile = str(rng.choice(["ECG", "EMG", "EOG"], p=[.7, .15, .15]))
    if scenario == "holds" and dur < 600: scenario = "baseline"
    rid = f"SYN-{seed:08d}"
    p = sample_profile(rid, seed, dur, tile, holds=schedule(rng, dur, scenario), taps=bool(rng.random() < 0.5 and scenario == "holds"))
    df, truth = render(p); truth["scenario"] = scenario
    v = validate(df, truth, tile)
    path = out / (rid + (".csv" if fmt == "csv" else ".npz"))
    if fmt == "csv": write_csv(df, path)
    else:
        np.savez_compressed(path, **{c: df[c].to_numpy() for c in COLUMNS if "[LSB]" in c or c == "time[us]"})
    (out / f"{rid}.truth.json").write_text(json.dumps(truth))
    h = hashlib.sha256(df.iloc[:2000].to_csv(index=False).encode()).hexdigest()[:16]
    fails = [k for k, c in v["checks"].items() if c["required"] and not c["passed"]]
    return dict(id=rid, synthetic=True, not_a_human_recording=True, generator=GEN_VERSION, seed=seed, scenario=scenario, tile=tile, profile=truth["profile"],
                fs_hz=v["fs"], channels=COLUMNS, n_samples=len(df), duration_s=dur, events=truth["events"], sync_taps_s=truth["sync_taps_s"], file=path.name, head_sha256_16=h,
                spec="results/synth/spec/real_data_spec.json", validation=dict(valid=v["valid"], failed_required=fails,
                optional_failed=[k for k, c in v["checks"].items() if not c["required"] and not c["passed"]]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--size", choices=list(SIZES)); ap.add_argument("--n", type=int); ap.add_argument("--minutes", type=float)
    ap.add_argument("--out", required=True); ap.add_argument("--fmt", default="csv"); ap.add_argument("--base-seed", type=int, default=100000); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); n, m = SIZES[a.size] if a.size else (a.n, a.minutes); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    spec_sha = hashlib.sha256(Path("results/synth/spec/real_data_spec.json").read_bytes()).hexdigest()[:16]; t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex, open(out / "manifest.jsonl", "w") as f:
        for k, rec in enumerate(ex.map(make, range(n), [a.base_seed] * n, [m] * n, [a.fmt] * n, [out] * n), 1):
            rec.update(spec_sha256_16=spec_sha, generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds")); f.write(json.dumps(rec) + "\n")
            if k % 10 == 0 or k == n: print(f"{k}/{n} {time.time()-t0:.0f}s", flush=True)
    man = [json.loads(l) for l in open(out / "manifest.jsonl")]
    print("valid:", sum(r["validation"]["valid"] for r in man), "/", len(man))


if __name__ == "__main__":
    main()
