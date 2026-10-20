"""Pinned, CWD-independent access to the model artifacts the dashboard and deployment wrapper need.

Root cause of `FileNotFoundError: data\\mesa\\cache\\final_models\\headline.pt` (2026-10): the loaders used a path RELATIVE to the working
directory, and `data/mesa/` is gitignored, so a fresh clone (or a clone made before commit a184d56, which force-added the file) has no such file.
Nothing is substituted when an artifact is missing or does not match its pinned hash: the caller gets a CheckpointError whose message says what
is wrong and how to fix it.   python3 -m src.deploy.checkpoints   prints the status table (exit 1 if anything required is bad).
"""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(os.environ.get("APNEA_CHECKPOINT_MANIFEST", ROOT / "docs/checkpoints.json"))   # override is used by the e2e tests


class CheckpointError(FileNotFoundError):
    """Required model artifact missing, unreadable or not the pinned version."""


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def status(name: str, check_hash: bool = True) -> dict:
    m = manifest()[name]; p = ROOT / m["path"]
    s = dict(name=name, path=m["path"], required_for=m["required_for"], exists=p.exists(), ok=False, problem=None)
    if not p.exists():
        s["problem"] = f"file not found: {p}"
    elif p.stat().st_size == 0:
        s["problem"] = "file is empty (0 bytes)"
    elif check_hash and _sha(p) != m["sha256"]:
        s["problem"] = f"sha256 differs from the pinned version ({m['sha256'][:12]}…): not the validated artifact"
    else:
        s["ok"] = True
    s["fix"] = m["fix"]; return s


def require(name: str, check_hash: bool = True) -> Path:
    s = status(name, check_hash)
    if not s["ok"]:
        raise CheckpointError(f"Required model artifact '{name}' is unavailable: {s['problem']}. Needed for: {s['required_for']}. To fix: {s['fix']} "
                              f"(resolved against the project root {ROOT}, not the working directory).")
    return ROOT / s["path"]


def all_status(check_hash: bool = True) -> list[dict]:
    return [status(k, check_hash) for k in manifest()]


if __name__ == "__main__":
    rows = all_status()
    for r in rows: print(("OK   " if r["ok"] else "FAIL ") + r["name"], "-", r["problem"] or r["path"])
    sys.exit(0 if all(r["ok"] for r in rows) else 1)
