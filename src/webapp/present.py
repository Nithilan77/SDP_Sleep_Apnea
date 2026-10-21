"""Presentation dashboard: a short storyline + an upload page. Registered on the existing app at /present; `python3 -m src.webapp.present` serves it at /.
   python3 -m src.webapp.present   ->  http://127.0.0.1:5000   (detailed technical dashboard stays at /full)"""
from __future__ import annotations
import sys, tempfile, traceback, uuid
from collections import OrderedDict
from pathlib import Path
from flask import Blueprint, abort, make_response, render_template, request
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.webapp import present_data as D, pipeline as P

bp = Blueprint("present", __name__, url_prefix="/present")
NAV = [("story", "/present/#project", "Project"), ("p1", "/present/#physionet", "PhysioNet ECG"), ("p2", "/present/#why-mesa", "Why MESA"), ("p3", "/present/#mesa", "MESA results"),
       ("p4", "/present/#data", "Our data"), ("p5", "/present/#transfer", "Transfer learning"), ("try", "/present/try", "Try it")]
REPORTS: "OrderedDict[str, dict]" = OrderedDict()


@bp.get("/")
def story():
    return render_template("present/story.html", NAV=NAV, page="story", pn=D.physionet(), mesa=D.mesa(), dg=D.domain_gap(), tr=D.transfer())


@bp.get("/try")
def try_page():
    return render_template("present/try.html", NAV=NAV, page="try", r=None, error=None, rid=None)


@bp.post("/analyze")
def analyze():
    f = request.files.get("recording")
    if f is None or not f.filename:
        return render_template("present/try.html", NAV=NAV, page="try", r=None, error="No file uploaded.", rid=None), 400
    from src.webapp import pipeline_present as PP
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / Path(f.filename).name; f.save(p)
        try:
            r = PP.analyze(p)
        except P.InputError as e:
            return render_template("present/try.html", NAV=NAV, page="try", r=None, error=str(e), rid=None), 400
        except Exception as e:
            traceback.print_exc()
            return render_template("present/try.html", NAV=NAV, page="try", r=None, error=f"Internal error while processing: {type(e).__name__}: {str(e)[:200]}", rid=None), 500
    rid = uuid.uuid4().hex[:12]; REPORTS[rid] = r
    while len(REPORTS) > 10: REPORTS.popitem(last=False)
    return render_template("present/try.html", NAV=NAV, page="try", r=r, error=None, rid=rid)


@bp.get("/report/<rid>")
def report(rid):
    r = REPORTS.get(rid)
    if r is None: abort(404, "Report expired or unknown (the last 10 analyses are kept in memory; run the analysis again).")
    resp = make_response(render_template("present/report.html", r=r)); resp.headers["Content-Disposition"] = f'attachment; filename="recording_report_{rid}.html"'
    return resp


def register(app):
    app.register_blueprint(bp)


if __name__ == "__main__":
    from flask import redirect
    from src.webapp.app import app
    orig = app.view_functions["overview"]
    app.view_functions["overview"] = lambda: redirect("/present/")      # '/' opens the presentation view
    app.add_url_rule("/full", "full_overview", orig)                      # detailed technical overview
    app.run(host="127.0.0.1", port=5000, debug=False)
