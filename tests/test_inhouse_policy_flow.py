"""Policy + data-flow tests (stdlib unittest):  python3 -m unittest tests.test_inhouse_policy_flow -v
Checks that (1) the old 'never train on in-house recordings' rule is gone from the documents/code, (2) in-house annotations keep label semantics,
(3) in-house recordings actually flow through windowing -> augmentation -> synthetic -> LR/HGB/CNN training -> inference, (4) checkpoint errors are explicit."""
import json, os, re, subprocess, sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT); sys.path.insert(0, str(ROOT))
os.environ.setdefault("SYNTH_SPEC_DIR", "results/synth/spec_noS1")
S1 = ROOT / "data/recordings/breath_hold/user_MukuBreathHold_ECG_07_10_2026.csv"
OLD = [r"never train on (them|our own)", r"VALIDATION ONLY\. Never", r"wrong by design\.?\s*$"]


class PolicyText(unittest.TestCase):
    def test_old_rule_removed_from_claude_md(self):
        s = (ROOT / "CLAUDE.md").read_text()
        body = s.split("> **History", 1)[0] + s.split("## 5. Hardware", 1)[1]          # exclude the quoted-history block
        for pat in OLD: self.assertIsNone(re.search(pat, body, re.I | re.M), pat)
        self.assertIn("IN-HOUSE DATA-USE POLICY", s); self.assertIn("voluntary_breath_hold", s); self.assertIn("Track A", s)

    def test_no_code_enforces_the_old_restriction(self):
        for p in list((ROOT / "src").rglob("*.py")):
            t = p.read_text()
            for pat in (r"never train on our own", r"validation[- ]only"):
                self.assertIsNone(re.search(pat, t, re.I), f"{p}: {pat}")

    def test_webapp_text_updated(self):
        self.assertNotIn("validation only; raw files are never committed or trained on", (ROOT / "src/webapp/templates/methods.html").read_text())


class Annotations(unittest.TestCase):
    def setUp(self):
        import pandas as pd; self.a = pd.read_csv(ROOT / "results/inhouse_bh/annotations.csv")

    def test_no_clinical_labels(self):
        self.assertTrue(set(self.a.label) <= {"baseline_reference", "voluntary_breath_hold", "recovery", "uncertain"})

    def test_session2_is_unaligned_and_unusable(self):
        s2 = self.a[self.a.recording_id.str.startswith("S2")]; self.assertTrue((s2.label == "uncertain").all()); self.assertTrue((s2.time_base == "stopwatch").all())

    def test_columns(self):
        for c in ("recording_id", "start_s", "end_s", "label", "confidence", "annotation_source", "verification_status"): self.assertIn(c, self.a.columns)


@unittest.skipUnless(S1.exists(), "session-1 recording not present")
class Flow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from src.inhouse_bh import common as Cm
        x, d, fs = Cm.trace32(S1); cls.x = x; cls.w, _ = Cm.labelled_windows("S1_ECGTILE", x)
        cls.X = Cm.window_array(x, cls.w.i0.values, cls.w.scale.values); cls.y = (cls.w.label == "voluntary_breath_hold").astype(int).values

    def test_windows_avoid_uncertain(self):
        import pandas as pd
        a = pd.read_csv(ROOT / "results/inhouse_bh/annotations.csv"); u = a[(a.recording_id == "S1_ECGTILE") & (a.label == "uncertain")]
        for t0 in self.w.t0:
            self.assertFalse(((u.start_s < t0 + 10) & (u.end_s > t0)).any())
        self.assertGreater(self.y.sum(), 20)

    def test_augment_preserves_shape_and_is_deterministic(self):
        import numpy as np
        from src.inhouse_bh.augment import augment_window
        a = augment_window(self.X[0], np.random.default_rng(1)); b = augment_window(self.X[0], np.random.default_rng(1))
        self.assertEqual(a.shape, self.X[0].shape); self.assertTrue(np.allclose(a, b)); self.assertFalse(np.allclose(a, self.X[0]))

    def test_training_flow_with_augmentation_and_synthetic_runs(self):
        import numpy as np
        from src.inhouse_bh import run_experiments as R, synthetic as Syn
        syn = Syn.synthetic_windows(2, 7)[:2]; ev = self.w.event_no.values; tr, te = ev != 1, ev == 1
        R.torch.set_num_threads(1)
        Xr, yr = R.expand(self.X[tr], self.y[tr], 0, k=1); self.assertEqual(len(Xr), 2 * tr.sum())
        sk = R.sk_models(0)
        for m in sk.values():
            m.fit(R.F(np.concatenate([Xr, syn[0]])), np.concatenate([yr, syn[1]])); p = m.predict_proba(R.F(self.X[te]))[:, 1]; self.assertTrue(np.isfinite(p).all())
        cnn = R.fit_cnn(np.concatenate([self.X[tr], syn[0]]), np.concatenate([self.y[tr], syn[1]]), 0, epochs=2, aug=True); p = R.prob_cnn(cnn, self.X[te]); self.assertEqual(len(p), te.sum())

    def test_pipeline_end_to_end_inference_if_bundle_exists(self):
        from src.inhouse_bh import predict as BH
        if not (BH.DEPLOY / "cnn.pt").exists(): self.skipTest("deployed bundle not trained yet")
        r = BH.predict_recording(S1, n_aug=1); self.assertGreater(r["n_scored"], 500); self.assertIn("ENS(mean)", r["probs"])


class Checkpoints(unittest.TestCase):
    def _manifest(self, mutate):
        m = json.loads((ROOT / "docs/checkpoints.json").read_text()); mutate(m); f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); json.dump(m, f); f.close(); return f.name

    def _run(self, manifest):
        env = dict(os.environ, APNEA_CHECKPOINT_MANIFEST=manifest)
        return subprocess.run([sys.executable, "-c", "from src.deploy import checkpoints as C; C.require('headline.pt')"], cwd=ROOT, env=env, capture_output=True, text=True)

    def test_ok(self):
        self.assertEqual(self._run(str(ROOT / "docs/checkpoints.json")).returncode, 0)

    def test_missing_is_explicit_not_substituted(self):
        r = self._run(self._manifest(lambda m: m["headline.pt"].update(path="data/mesa/cache/final_models/does_not_exist.pt")))
        self.assertNotEqual(r.returncode, 0); self.assertIn("unavailable", r.stderr); self.assertIn("To fix", r.stderr)

    def test_hash_mismatch_detected(self):
        r = self._run(self._manifest(lambda m: m["headline.pt"].update(sha256="0" * 64)))
        self.assertNotEqual(r.returncode, 0); self.assertIn("sha256 differs", r.stderr)

    def test_cwd_independent(self):
        r = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, '{ROOT}'); from src.deploy import checkpoints as C; print(C.require('headline.pt'))"], cwd="/tmp", capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


class SyntheticValidator(unittest.TestCase):
    def test_generated_recording_validates_and_is_deterministic(self):
        import numpy as np
        from src.synth.render import sample_profile, render
        from src.synth.validate import validate
        p = sample_profile("t", 11, 300, "ECG", holds=[(100, 130, 1)], quality="good"); d1, t1 = render(p); d2, _ = render(p)
        self.assertTrue(d1.equals(d2)); v = validate(d1, t1, "ECG"); self.assertTrue(v["valid"], [k for k, c in v["checks"].items() if c["required"] and not c["passed"]])


if __name__ == "__main__":
    unittest.main()
