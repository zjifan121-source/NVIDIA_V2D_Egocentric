import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("fallback", Path(__file__).parents[1] / "asset_fallback.py")
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


class FallbackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        proof = self.root / "proof.json"
        proof.write_text('{"input_only": true}')
        self.proof = [{"path": "proof.json", "sha256": f.sha256(proof)}]
        root = self.root / "bundle"
        root.mkdir()
        (root / "mesh.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        T = np.repeat(np.eye(4)[None], 3, axis=0)
        np.savez(root / "result.npz", object_names=np.array(["object"]),
                 camera_to_world_transform=T, camera_is_valid=np.ones(3, bool),
                 object_to_camera_transform=T, object_to_world_transform=T,
                 object_is_valid=np.ones(3, bool))
        files = {p.name: f.sha256(p) for p in root.iterdir()}
        self.arm = dict(state="COMPLETE", input_signature="same-observations-and-settings",
                        bundle=dict(path="bundle", files=files),
                        binding=dict(convention=f.CONVENTION, bundle_sha256=f.fingerprint(files), evidence=self.proof),
                        gate=dict(state="PASS", GT_used=False, evidence=self.proof))
        self.case = dict(id="arbitrary_name", frames=3, objects=["object"],
                         baseline=copy.deepcopy(self.arm), candidate=copy.deepcopy(self.arm))

    def fail_baseline(self):
        self.case["baseline"] = dict(state="FAIL_ANCHOR_ALIGNMENT", input_signature=self.arm["input_signature"],
                                     gate=dict(state="FAIL", GT_used=False,
                                               reason="ASYMMETRIC_STRUCTURE_DIRECTION", evidence=self.proof))

    def test_complete_baseline_always_retained(self):
        self.case["candidate"]["auc"] = 1
        self.case["baseline"]["auc"] = 0
        self.assertEqual(f.choose(self.case, self.root)[0], "baseline")

    def test_failure_recovery_independent_of_case_id(self):
        self.fail_baseline()
        for ident in ["unknown", "episode_000005", "banana"]:
            self.case["id"] = ident
            self.assertEqual(f.choose(self.case, self.root)[0], "candidate")

    def test_mismatched_inputs_rejected(self):
        self.fail_baseline()
        self.case["candidate"]["input_signature"] = "different"
        with self.assertRaisesRegex(ValueError, "input signature"):
            f.choose(self.case, self.root)

    def test_missing_structure_review_rejected(self):
        self.fail_baseline()
        del self.case["candidate"]["gate"]
        with self.assertRaisesRegex(ValueError, "candidate review"):
            f.choose(self.case, self.root)

    def test_failed_candidate_stays_unresolved(self):
        self.fail_baseline()
        self.case["candidate"]["gate"]["state"] = "FAIL"
        self.assertIsNone(f.choose(self.case, self.root)[0])

    def test_partial_pose_file_rejected_even_with_updated_hash(self):
        self.case["frames"] = 4
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            f.choose(self.case, self.root)

    def test_tampered_mesh_rejected(self):
        (self.root / "bundle/mesh.obj").write_text("v 42 42 42\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            f.choose(self.case, self.root)

    def test_new_mesh_old_binding_rejected(self):
        self.case["baseline"]["bundle"]["files"]["mesh.obj"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            f.choose(self.case, self.root)

    def test_evidence_tamper_rejected(self):
        (self.root / "proof.json").write_text("tamper")
        with self.assertRaisesRegex(ValueError, "receipt hash mismatch"):
            f.choose(self.case, self.root)

    def test_atomic_export_and_no_overwrite(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(dict(schema_version=1, cases=[self.case])))
        output = self.root / "export"
        result = f.run(manifest, output)
        self.assertTrue(result["complete"])
        self.assertEqual(f.sha256(output / "arbitrary_name/result.npz"),
                         f.sha256(self.root / "bundle/result.npz"))
        with self.assertRaisesRegex(ValueError, "never overwrite"):
            f.run(manifest, output)


if __name__ == "__main__":
    unittest.main()
