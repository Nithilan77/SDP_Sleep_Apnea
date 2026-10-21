"""Presentation-dashboard tests (unittest):  python3 -m unittest tests.test_present -v"""
import io, os, re, sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.webapp.app import app
from src.webapp import present_data as D
OVERNIGHT = ROOT / "data/recordings/inhouse_ecg/user_1612bhavi_ecg_001.txt"; EMG = ROOT / "data/recordings/breath_hold/user_MukuBreathHold_EMG_07_10_2026.csv"


def txt(h): return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"base64,[A-Za-z0-9+/=]+", "", h)))


class Story(unittest.TestCase):
    def setUp(self): self.c = app.test_client(); self.h = self.c.get("/present/").get_data(as_text=True); self.t = txt(self.h)

    def test_all_sections_present(self):
        for s in ("What is this project", "Starting point: ECG only", "Why we moved to MESA", "What we did with MESA", "Our own data", "Transfer learning to our sensor", "Try it on a recording"):
            self.assertIn(s, self.t)

    def test_no_single_result_is_called_best(self):
        self.assertNotIn("best", self.t.lower()); self.assertNotIn("winner", self.t.lower())

    def test_numbers_come_from_result_files(self):
        pn, tr, dg = D.physionet(), D.transfer(), D.domain_gap()
        self.assertIn("%.2f" % pn["cvhr"], self.t); self.assertIn("%.2f" % tr["frozen"]["auroc"], self.t); self.assertIn("%.0f%%" % (100 * dg["ours"]), self.t)
        self.assertAlmostEqual(tr["pooled_frozen"], 0.031, places=3); self.assertAlmostEqual(tr["pooled_hand"], 0.046, places=3)

    def test_unverified_numbers_not_shown(self):
        for bad in ("AHI", "83.7", "81.4", "3.7 s", "64-dim"): self.assertNotIn(bad, self.t)

    def test_spo2_models_tagged_and_limits_stated(self):
        self.assertGreaterEqual(self.t.count("needs SpO2"), 5); self.assertIn("not clinical apnea", self.t)

    def test_old_dashboard_still_available(self):
        self.assertEqual(self.c.get("/").status_code, 200); self.assertEqual(self.c.get("/present/try").status_code, 200)


class Upload(unittest.TestCase):
    def setUp(self): self.c = app.test_client()

    def post(self, name, data): return self.c.post("/present/analyze", data={"recording": (io.BytesIO(data), name)}, content_type="multipart/form-data")

    def test_invalid_inputs_rejected_with_reason(self):
        for name, data in (("x.pdf", b"%PDF"), ("g.csv", b"a,b\n1,2\n"), ("e.csv", b"")):
            r = self.post(name, data); self.assertEqual(r.status_code, 400); self.assertIn("could not be analysed", r.get_data(as_text=True))
        self.assertEqual(self.c.post("/present/analyze", data={}, content_type="multipart/form-data").status_code, 400)

    @unittest.skipUnless(EMG.exists(), "recording missing")
    def test_non_ecg_tile_rejected(self):
        r = self.post("emg.csv", EMG.read_bytes()); self.assertEqual(r.status_code, 400); self.assertIn("no clear cardiac rhythm", r.get_data(as_text=True))

    @unittest.skipUnless(OVERNIGHT.exists(), "recording missing")
    def test_overnight_upload_report(self):
        r = self.post("night.txt", OVERNIGHT.read_bytes()); h = r.get_data(as_text=True); self.assertEqual(r.status_code, 200)
        t = txt(h); self.assertIn("What the two heads flagged", t); self.assertIn("2.8 h", t); self.assertGreaterEqual(h.count("data:image/png;base64"), 1); self.assertNotIn("AHI", t)
        rid = re.search(r"/present/report/([0-9a-f]{12})", h).group(1); rep = self.c.get(f"/present/report/{rid}")
        self.assertEqual(rep.status_code, 200); self.assertIn("attachment", rep.headers["Content-Disposition"]); self.assertIn("Limitations", rep.get_data(as_text=True))
        self.assertEqual(self.c.get("/present/report/deadbeef0000").status_code, 404)


if __name__ == "__main__":
    unittest.main()
