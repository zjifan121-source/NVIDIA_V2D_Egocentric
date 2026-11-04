"""CPU-only contracts, not evidence of an executed CUDA tracking branch."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np

spec=importlib.util.spec_from_file_location("cohort_runner",Path(__file__).with_name("run.py"))
r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


class Contracts(unittest.TestCase):
    def test_cpu_import_never_loads_torch_and_wrong_runner_pin_fails(self):
        self.assertNotIn("torch",sys.modules)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/"manifest.json").write_text("{}")
            (p/"protocol.json").write_text(json.dumps({"runner_sha256":"wrong"}))
            with self.assertRaisesRegex(ValueError,"Runner is not pinned"):
                r.load_package(p/"manifest.json",p/"protocol.json")

    def test_variable_length_anchor_schedules_are_complete_without_repeat(self):
        for count in [1,166,173,235,263,313,407]:
          for anchor in range(count):
            s=r.schedule(anchor,count)
            self.assertEqual(len(s["forward"])+len(s["backward"]),count-1)
            self.assertEqual(sorted([anchor]+s["forward"]+s["backward"]),list(range(count)))
            self.assertEqual(s["forward"],list(range(anchor+1,count)))
            self.assertEqual(s["backward"],list(range(anchor-1,-1,-1)))

    def test_nonzero_mesh_center_and_rotated_pose_use_right_multiplication(self):
        c=np.array([.031,-.087,.046]); A=np.eye(4);A[:3,3]=-c
        public=np.eye(4);public[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];public[:3,3]=[.2,.1,.7]
        vertices=np.array([[.01,.02,.04],[.08,-.1,.06],[.07,.04,-.01]])
        internal=r.public_to_internal(public,A)
        v_h=np.c_[vertices,np.ones(3)]
        np.testing.assert_allclose((public@v_h.T).T,(internal@(A@v_h.T)).T,atol=1e-14)
        self.assertLess(r.assert_pair(internal,A,public),1e-14)
        with self.assertRaises(ValueError):r.assert_pair(public,A,public)

    def test_reset_each_direction_recovers_same_anchor_not_last_forward_pose(self):
        A=np.eye(4);A[:3,3]=[-.05,.02,0]
        anchor=np.eye(4);anchor[:3,3]=[0,0,.5]
        states=[]
        for _ in ("forward","backward"):
            state=r.public_to_internal(anchor.copy(),A)
            states.append(state.copy());state[:3,3]+=[100,20,30]
        np.testing.assert_array_equal(states[0],states[1])
        np.testing.assert_array_equal(anchor[:3,3],[0,0,.5])

    def test_nonrigid_or_reflected_pose_rejected(self):
        for diagonal in ([2,1,1],[-1,1,1]):
            T=np.eye(4);T[:3,:3]=np.diag(diagonal)
            with self.assertRaises(ValueError):r.validate_transform(T)

    def test_runtime_model_change_rejected_but_mount_path_can_change(self):
        expected=dict(backend="nvidia_tensorrt",model_registry="r",model_version="v",
                      tensorrt_version="10",gpu_compute_capability="8.6",models={"score":{"sha256":"abc"}})
        r.validate_runtime({**expected,"model_dir":"/different-mount"},expected)
        with self.assertRaises(ValueError):r.validate_runtime({**expected,"models":{}},expected)

    def test_native_tuple_model_spec_equals_json_array_but_still_pins_values(self):
        expected=dict(backend="nvidia_tensorrt",model_registry="r",model_version="v",tensorrt_version="10",gpu_compute_capability="8.6",models={"score":{"sha256":"abc","input_names":["inputA","inputB"]}})
        actual={**expected,"models":{"score":{"sha256":"abc","input_names":("inputA","inputB")}}}
        r.validate_runtime(actual,expected)
        actual["models"]["score"]["sha256"]="different"
        with self.assertRaises(ValueError):r.validate_runtime(actual,expected)

    def test_paths_cannot_escape_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in ("../mesh.obj","/tmp/mesh.obj"):
                with self.assertRaises(ValueError):r.checked_path(tmp,bad)

    def test_depth_inverse_codec_invalid_values_and_mask(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            rgb=np.zeros((2,3,3),np.uint8);raw=np.array([[0,65535,32768],[16384,32768,32768]],np.uint16)
            mask=np.array([[255,255,255],[255,0,255]],np.uint8)
            for name,arr in [("rgb.png",rgb),("depth.png",raw),("mask.png",mask)]:cv2.imwrite(str(p/name),arr)
            _,d,_=r.decode_frame(p,{"frame_id":0,"rgb":"rgb.png","depth":"depth.png","mask":"mask.png"},{"width":3,"height":2})
            self.assertEqual(d[0,0],0);self.assertEqual(d[0,1],0);self.assertEqual(d[1,1],0)
            self.assertAlmostEqual(float(d[0,2]),65535/32768-1,places=6)
            self.assertAlmostEqual(float(d[1,0]),65535/16384-1,places=6)

    def test_track_gate_needs_exact_mesh_pose_inputs_and_human_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"anchor.npy";np.save(p,np.eye(4))
            review=dict(proceed_to_fixed_tracking=True,visual_target_identity_checked=True,
                visible_structure_alignment_checked=True,no_gross_target_miss=True,rank0_only=True,
                protocol_sha256="p",input_manifest_sha256="i",mesh_sha256="m",
                anchor_public_sha256=r.sha(p),anchor_frame=42,reviewer="human",reviewed_evidence=["anchor_overlay.png"])
            r.validate_review(review,"p","i","m",p,42)
            for change in ({"mesh_sha256":"oldmesh"},{"no_gross_target_miss":False},{"rank0_only":False}):
                with self.assertRaises(ValueError):r.validate_review({**review,**change},"p","i","m",p,42)


if __name__=="__main__":unittest.main(verbosity=2)
