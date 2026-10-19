"""Research dashboard.   python3 -m src.webapp.app   ->  http://127.0.0.1:5000

Pages: / (overview), /inhouse, /detect (upload + real inference), /models, /mesa, /methods, /report/<id> (downloadable HTML report).
Every number shown is read from a result file (src/webapp/artifacts.py) or computed from the uploaded file (src/webapp/pipeline.py)."""
from __future__ import annotations

import datetime, hashlib, json, sys, tempfile, traceback, uuid
from collections import OrderedDict
from pathlib import Path

from flask import Flask, abort, make_response, render_template, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.webapp import artifacts as A, charts as C, pipeline as P  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024   # 1 GiB (overnight .txt files reach ~650 MB)
DEMO = ROOT / "data/recordings/breath_hold/user_MukuBreathHold_ECG_07_10_2026.csv"
ART_DIRS = ("inhouse_audit", "breath_hold", "breath_hold_s2", "mesa", "expA_multiseed", "phase3_cnn", "inhouse_validation")
RESULTS: "OrderedDict[str, dict]" = OrderedDict()      # last analyses, for report download
NAV = [("overview", "/", "Overview"), ("inhouse", "/inhouse", "In-house recordings"), ("detect", "/detect", "Apnea detection"),
       ("models", "/models", "Model comparison"), ("mesa", "/mesa", "MESA multimodal"), ("methods", "/methods", "Methodology")]


def _ckpt_versions() -> list[dict]:
    out = []
    for name, d in (("1D-CNN (adopted)", P.CNN_DIR), *P.REC_DIRS.items()):
        f = d / "model_state_dict.pt"
        out.append(dict(model=name, path=str(d.relative_to(ROOT)), sha256_12=hashlib.sha256(f.read_bytes()).hexdigest()[:12] if f.exists() else "missing"))
    return out


@app.context_processor
def _ctx():
    return dict(NAV=NAV, C=C, now=datetime.date.today().isoformat())


@app.get("/")
def overview():
    hist = {r["label"]: r for r in A.historical()}; mesa = {r["label"]: r for r in A.mesa().get("rows", [])}
    inh = A.inhouse(); ev = inh.get("events") or []
    return render_template("overview.html", page="overview", hist=hist, mesa=mesa, ctrl=A.controlled(), inh=inh, n_ev=len(ev), n_ev_lt1=sum(1 for e in ev if e["effort_ratio_fixedpad"] < 0.8),
                           mesa_cfg=A.mesa().get("cfg", {}).get("headline", {}))


@app.get("/inhouse")
def inhouse():
    prov = ROOT / "results/inhouse_audit/provenance.json"
    return render_template("inhouse.html", page="inhouse", inh=A.inhouse(), prov=json.loads(prov.read_text()) if prov.exists() else None, gate=P.CARDIAC_PROM_MIN)


@app.get("/models")
def models():
    return render_template("models.html", page="models", hist=A.historical(), ctrl=A.controlled())


@app.get("/mesa")
def mesa():
    return render_template("mesa.html", page="mesa", m=A.mesa())


@app.get("/methods")
def methods():
    return render_template("methods.html", page="methods", ckpts=_ckpt_versions())


@app.get("/detect")
def detect():
    return render_template("detect.html", page="detect", r=None, error=None, rid=None)


def _run(path: Path, shown_name: str):
    try:
        r = P.analyze(path)
    except P.InputError as e:
        return render_template("detect.html", page="detect", r=None, error=str(e), rid=None), 400
    except Exception as e:   # unexpected: show a message, log the trace to the console; never substitute a prediction
        traceback.print_exc()
        return render_template("detect.html", page="detect", r=None, error=f"Internal error while processing: {type(e).__name__}: {str(e)[:200]}", rid=None), 500
    rid = uuid.uuid4().hex[:12]; r["analysed"] = datetime.datetime.now().isoformat(timespec="seconds"); r["source_name"] = shown_name
    RESULTS[rid] = r
    while len(RESULTS) > 10: RESULTS.popitem(last=False)
    return render_template("detect.html", page="detect", r=r, error=None, rid=rid)


@app.post("/analyze")
def analyze():
    f = request.files.get("recording")
    if f is None or not f.filename:
        return render_template("detect.html", page="detect", r=None, error="No file uploaded.", rid=None), 400
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / Path(f.filename).name; f.save(p)
        return _run(p, Path(f.filename).name)


@app.post("/demo")
def demo():
    if not DEMO.exists():
        return render_template("detect.html", page="detect", r=None, error=f"Demo recording missing: {DEMO.name}", rid=None), 404
    return _run(DEMO, DEMO.name + " (bundled demo)")


@app.get("/report/<rid>")
def report(rid):
    r = RESULTS.get(rid)
    if r is None:
        abort(404, "Report expired or unknown (reports are kept in memory for the last 10 analyses; re-run the analysis).")
    html = render_template("report.html", r=r, ckpts=_ckpt_versions(), hist={x["label"]: x for x in A.historical()})
    resp = make_response(html); resp.headers["Content-Disposition"] = f'attachment; filename="apnea_screening_report_{rid}.html"'
    return resp


@app.get("/artifact/<path:p>")
def artifact(p):
    parts = Path(p).parts
    if not parts or parts[0] not in ART_DIRS or Path(p).suffix.lower() != ".png" or ".." in parts:
        abort(404)
    return send_from_directory(ROOT / "results", p)


@app.errorhandler(404)
def _404(e):
    return render_template("message.html", page="", title="Not found", msg=getattr(e, "description", "Page not found.")), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
