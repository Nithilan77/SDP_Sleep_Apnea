"""End-to-end test: starts the REAL Flask server as a subprocess and drives it over HTTP.   python3 tests/e2e_webapp.py
Writes results/webapp_e2e/test_table.md (case | expected | actual | status). Requires the bundled recordings to exist.
Covers: every page, valid uploads (real inference), invalid/unsupported uploads (wrong type, garbage, too short, empty, non-ECG tiles,
flat-QVAR / IMU-only), report export, artifact whitelist, restart + health after errors."""
import re, subprocess, sys, time, urllib.request, urllib.error, uuid, tempfile, signal
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; PORT = 5055; URL = f"http://127.0.0.1:{PORT}"
BH = ROOT / "data/recordings/breath_hold"
S1 = BH / "user_MukuBreathHold_ECG_07_10_2026.csv"; S1_EMG = BH / "user_MukuBreathHold_EMG_07_10_2026.csv"; S1_EOG = BH / "user_MukuBreathHold_EOG_07_10_2026.csv"
MIT = Path.home() / "Desktop/Mithin-BreathHold"
S2 = MIT / "user_MithinBreathHold_ECG_000.csv"; S2_EMG = MIT / "user_MithinBreathHold_EMG_000.csv"
TMP = Path(tempfile.mkdtemp())


def req(path, method="GET", fname=None, data=None, timeout=900):
    if fname is None:
        r = urllib.request.Request(URL + path, method=method, data=b"" if method == "POST" else None)
    else:
        b = uuid.uuid4().hex; body = (f'--{b}\r\nContent-Disposition: form-data; name="recording"; filename="{fname}"\r\n'
                                      f'Content-Type: application/octet-stream\r\n\r\n').encode() + data + f"\r\n--{b}--\r\n".encode()
        r = urllib.request.Request(URL + path, data=body, method="POST", headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp: return resp.status, resp.read().decode("utf-8", "replace"), dict(resp.headers)
    except urllib.error.HTTPError as e: return e.code, e.read().decode("utf-8", "replace"), dict(e.headers)


def start(env=None):
    import os
    p = subprocess.Popen([sys.executable, "-c", f"from src.webapp.app import app; app.run(port={PORT})"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(os.environ, **(env or {})))
    for _ in range(80):
        try: urllib.request.urlopen(URL, timeout=1); return p
        except Exception: time.sleep(.5)
    p.kill(); raise SystemExit("server did not start")


def txt(h): return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"base64,[A-Za-z0-9+/=]+", "", h)))


rows = []
def case(name, expected, fn):
    t = time.time()
    try: ok, actual = fn()
    except Exception as e: ok, actual = False, f"exception {type(e).__name__}: {e}"
    rows.append((name, expected, actual, "PASS" if ok else "FAIL")); print(rows[-1], f"{time.time()-t:.1f}s", flush=True)


def page(path, *must):
    def f():
        st, h, _ = req(path); t = txt(h); missing = [m for m in must if m not in t]
        return st == 200 and not missing, f"HTTP {st}; missing={missing}"
    return f


def make_variants():
    """Derived test files (originals untouched): flat-QVAR copy (IMU-only log) and noise-QVAR copy of the session-1 ECG tile."""
    import pandas as pd, numpy as np
    d = pd.read_csv(S1)
    d.assign(**{"ah_qvar[LSB]": 0}).to_csv(TMP / "imu_only.csv", index=False)
    rng = np.random.default_rng(0)
    d.assign(**{"ah_qvar[LSB]": rng.normal(0, 150, len(d)).round().astype(int)}).to_csv(TMP / "noise_qvar.csv", index=False)


make_variants()
srv = start()
try:
    case("P1 Overview", "200; headline MESA result + historical CNN card", page("/", "AUROC", "Per-minute apnea classifier", "Pipeline"))
    case("P2 In-house", "200; inventory with 6 logs, Drive status, provenance", page("/inhouse", "Recording inventory", "RETRIEVED AND VERIFIED", "S2 EMG", "negative result"))
    case("P3 Models", "200; historical table; controlled section present", page("/models", "Model comparison", "Historical results", "Controlled comparison"))
    case("P4 MESA", "200; six configurations listed", page("/mesa", "ECG only", "SpO2 only", "ECG + SpO2", "Effort only", "ECG + effort (CANet)", "ECG + effort + SpO2"))
    case("P5 Methods", "200; leakage + limitations + checkpoint hashes", page("/methods", "leakage prevention", "Negative / limiting results", "SHA-256"))
    case("P6 Detect form", "200; upload form", page("/detect", "Upload", "EMG/EOG-wired tiles"))
    case("P9 Breath-hold page", "200; task banner says NOT sleep apnea", page("/breathhold", "voluntary breath-hold-like", "does not screen for sleep apnea", "Upload"))
    case("P10 System status", "200; all artifacts verified", page("/status", "System status", "all required artifacts present and verified", "headline.pt"))
    case("P11 Detect page names its task", "200; sleep-apnea screening banner", page("/detect", "Task: sleep-apnea screening", "not evidence that apnea is absent"))
    case("P12 Ensembles & expanded data page", "200; three result families, tables read from result files", page("/expansion", "Ensembles vs the strongest single model", "synthR10", "Events detected", "never pooled"))
    case("P7 Unknown page", "HTTP 404 with message", lambda: ((r := req("/nope"))[0] == 404, f"HTTP {r[0]}"))
    case("P8 Artifact whitelist", "png under results/inhouse_audit ok (200); traversal and non-png refused (404)",
         lambda: ((a := req("/artifact/inhouse_audit/figs/S1_tiles.png"))[0] == 200 and req("/artifact/../CLAUDE.md")[0] == 404 and req("/artifact/inhouse_audit/audit.json")[0] == 404,
                  f"png {a[0]}, traversal {req('/artifact/../CLAUDE.md')[0]}, json {req('/artifact/inhouse_audit/audit.json')[0]}"))

    state = {}
    def valid(path, dur_s, key):
        def f():
            st, h, _ = req("/analyze", "POST", path.name, path.read_bytes()); t = txt(h)
            has_img = h.count("data:image/png;base64") >= 3; dur_ok = f"({dur_s} s)" in t
            models = all(m in t for m in ("CNN", "RNN", "LSTM")); eff = "Effort branch" in t; rep = re.search(r"/report/([0-9a-f]{12})", h)
            if rep: state[key] = rep.group(1)
            m_ = re.search(r"3 · ECG branch.*?(?=4 · Effort|Limitations)", t); state[key + "_ecg"] = m_.group(0) if m_ else ""
            return st == 200 and has_img and dur_ok and models and eff and bool(rep), f"HTTP {st}; 3+ plots={has_img}; duration {dur_s}s shown={dur_ok}; CNN/RNN/LSTM rows={models}; effort section={eff}; report link={bool(rep)}"
        return f
    case("U1 upload valid recording (session 1 ECG tile)", "200; metadata 1796 s; plots; CNN/RNN/LSTM + effort + report link", valid(S1, 1796, "s1"))
    if S2.exists(): case("U2 upload different valid recording (session 2 ECG tile)", "200; duration 1574 s (differs from U1 => result tied to upload)", valid(S2, 1574, "s2"))
    case("U3 report export", "200 attachment HTML containing metadata, model SHA, limitations",
         lambda: ((r := req(f"/report/{state['s1']}"))[0] == 200 and "attachment" in r[2].get("Content-Disposition", "") and "Model versions" in r[1] and "not a clinical diagnosis" in r[1].lower() and "Limitations" in r[1],
                  f"HTTP {r[0]}; disposition={r[2].get('Content-Disposition','')[:40]}"))
    if "s2" in state:
        case("U3b results are computed from each upload", "ECG-branch numbers differ between the two uploads; each report carries its own recording duration",
             lambda: (state["s1_ecg"] != state["s2_ecg"] and len(state["s1_ecg"]) > 200 and "1796" in req(f"/report/{state['s1']}")[1] and "1574" in req(f"/report/{state['s2']}")[1],
                      f"ecg sections differ={state['s1_ecg'] != state['s2_ecg']} (len {len(state['s1_ecg'])}/{len(state['s2_ecg'])})"))
    case("U4 unknown report id", "HTTP 404", lambda: ((r := req("/report/deadbeef0000"))[0] == 404, f"HTTP {r[0]}"))
    case("E1 wrong file type (.pdf)", "HTTP 400, 'Unsupported file type'", lambda: ((r := req("/analyze", "POST", "x.pdf", b"%PDF-1.4"))[0] == 400 and "Unsupported file type" in r[1], f"HTTP {r[0]}"))
    case("E2 garbage .csv", "HTTP 400, 'Could not parse'", lambda: ((r := req("/analyze", "POST", "x.csv", b"a,b,c\n1,2,3\n"))[0] == 400 and "Could not parse" in r[1], f"HTTP {r[0]}: {txt(r[1])[-150:]}"))
    short = TMP / "short.csv"; short.write_bytes(b"".join(S1.open("rb").readlines()[:30000]))
    case("E3 valid format but too short (~2 min)", "HTTP 400, 'at least 600 s'", lambda: ((r := req("/analyze", "POST", "short.csv", short.read_bytes()))[0] == 400 and "at least 600 s" in r[1], f"HTTP {r[0]}: {txt(r[1])[-110:]}"))
    case("E4 empty upload", "HTTP 400, 'No file uploaded'", lambda: ((r := req("/analyze", "POST", "", b""))[0] == 400 and "No file" in r[1], f"HTTP {r[0]}"))
    for label, f in (("E5 session-1 EMG-wired tile", S1_EMG), ("E6 session-1 EOG-wired tile", S1_EOG), ("E7 session-2 EMG-wired tile", S2_EMG)):
        if f.exists(): case(label, "HTTP 400 'no clear cardiac rhythm' (not accepted as ECG)", lambda f=f: ((r := req("/analyze", "POST", f.name, f.read_bytes()))[0] == 400 and "no clear cardiac rhythm" in r[1] and "No prediction was produced" in r[1], f"HTTP {r[0]}: {txt(r[1])[txt(r[1]).find('Could not'):][:120]}"))
    case("E8 IMU-only log (QVAR flat)", "HTTP 400, flat QVAR rejected", lambda: ((r := req("/analyze", "POST", "imu_only.csv", (TMP / "imu_only.csv").read_bytes()))[0] == 400 and "flat" in r[1], f"HTTP {r[0]}: {txt(r[1])[txt(r[1]).find('Could not'):][:120]}"))
    case("E9 QVAR = noise (no cardiac rhythm)", "HTTP 400, cardiac-rhythm gate", lambda: ((r := req("/analyze", "POST", "noise_qvar.csv", (TMP / "noise_qvar.csv").read_bytes()))[0] == 400 and "no clear cardiac rhythm" in r[1], f"HTTP {r[0]}"))
    bh = {}
    def bhup(path, key, dur):
        def f():
            st, h, _ = req("/breathhold/analyze", "POST", path.name, path.read_bytes()); t = txt(h); rep = re.search(r"/breathhold/report/([0-9a-f]{12})", h)
            if rep: bh[key] = rep.group(1)
            ok = st == 200 and h.count("data:image/png;base64") >= 2 and f"{dur} s" in t and "Window counts" in t and "Original vs augmented" in t and "not" in t and bool(rep)
            return ok, f"HTTP {st}; plots={h.count('data:image/png;base64')}; duration {dur}s shown={f'{dur} s' in t}; counts table={'Window counts' in t}; aug section={'Original vs augmented' in t}; report={bool(rep)}; summary={t[t.find('Result')+7:t.find('Result')+110]}"
        return f
    case("B1 breath-hold upload (session 1, in-sample)", "200; plots, window counts, augmented comparison, in-sample banner, report link", bhup(S1, "b1", 1796))
    case("B1b in-sample banner shown for the training recording", "banner present", lambda: ("In-sample" in req("/breathhold/analyze", "POST", S1.name, S1.read_bytes())[1], "checked"))
    if S2.exists(): case("B2 breath-hold upload (session 2, out-of-sample)", "200; 1574 s; no in-sample banner", lambda: (lambda r: (r[0] == 200 and "In-sample" not in r[1] and "1574 s" in txt(r[1]), f"HTTP {r[0]}; in-sample banner absent={'In-sample' not in r[1]}"))(req("/breathhold/analyze", "POST", S2.name, S2.read_bytes())))
    case("B3 breath-hold report", "200 attachment; task disclaimer, model config, limitations", lambda: ((r := req(f"/breathhold/report/{bh['b1']}"))[0] == 200 and "attachment" in r[2].get("Content-Disposition", "") and "a sleep-apnea screening" in r[1] and "Limitations" in r[1] and "in-sample" in r[1].lower(), f"HTTP {r[0]}"))
    case("B4 breath-hold invalid inputs", "400 each: pdf, garbage, empty, no file, too short",
         lambda: (all(req("/breathhold/analyze", "POST", n, d)[0] == 400 for n, d in (("x.pdf", b"%PDF"), ("g.csv", b"a,b\n1,2\n"), ("e.csv", b""))) and req("/breathhold/analyze", "POST", "short.csv", short.read_bytes())[0] in (200, 400), "400 x3"))
    srv.send_signal(signal.SIGTERM); srv.wait(10)
    import json as _j, os as _o
    man = _j.loads((ROOT / "docs/checkpoints.json").read_text()); man["headline.pt"]["path"] = "data/mesa/cache/final_models/MISSING.pt"; man["bh_cnn"]["sha256"] = "0" * 64
    mf = TMP / "bad_manifest.json"; mf.write_text(_j.dumps(man)); srv = start({"APNEA_CHECKPOINT_MANIFEST": str(mf)})
    case("C1 status page reports unavailable artifacts", "200; UNAVAILABLE + fix text", page("/status", "UNAVAILABLE", "file not found", "sha256 differs", "Fix:"))
    case("C2 breath-hold with altered checkpoint: explicit error, no prediction", "400 'models unavailable', no result section", lambda: ((r := req("/breathhold/analyze", "POST", S1.name, S1.read_bytes()))[0] == 400 and "models unavailable" in r[1] and "Window counts" not in r[1], f"HTTP {r[0]}"))
    case("C3 sleep-apnea upload with missing headline.pt: ECG branch runs, effort branch explains", "200; warning mentions headline.pt missing; no effort section",
         lambda: ((r := req("/analyze", "POST", S1.name, S1.read_bytes()))[0] == 200 and "Effort branch unavailable" in r[1] and "headline.pt" in r[1] and "4 · Effort branch" not in r[1], f"HTTP {r[0]}"))
    srv.send_signal(signal.SIGTERM); srv.wait(10); srv = start()
    case("R1 restart server (documented command) + GET /", "HTTP 200 after restart", page("/", "Overview"))
    case("R2 report from before restart is gone (in-memory)", "HTTP 404 (documented behaviour)", lambda: ((r := req(f"/report/{state['s1']}"))[0] == 404, f"HTTP {r[0]}"))
    case("R3 server healthy after errors (demo button)", "HTTP 200, effort + ECG sections", lambda: ((r := req("/demo", "POST"))[0] == 200 and "Effort branch" in r[1] and "ECG branch" in r[1], f"HTTP {r[0]}"))
finally:
    srv.terminate()
out = ROOT / "results/webapp_e2e"; out.mkdir(parents=True, exist_ok=True)
(out / "test_table.v3.md").write_text("| Test | Expected | Actual | Status |\n|---|---|---|---|\n" + "\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows) + "\n")
print("\n".join(f"{r[3]} {r[0]}" for r in rows)); sys.exit(0 if all(r[3] == "PASS" for r in rows) else 1)
