# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only orchestration tests; no Docker, model assets, or GPU imports.

Compile the actual orchestration functions in isolation so tests do not need
every container's optional Python dependencies installed on the host.
"""
import argparse
import ast
import glob
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

MODULES = Path(__file__).resolve().parents[2]
WILOR = MODULES / "v2d_pipelines/run_ego_wilor.py"
UNIFIED = MODULES / "v2d_pipelines/run_ego_reconstruction.py"
WRAPPER = MODULES / "v2d_gsplat_refinement/docker/run_refine_simple.py"
LIBRARY = MODULES / "v2d_gsplat_refinement/lib/refine_simple.py"


def functions(path, names, **extra):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.Assign)) and
                (getattr(node, "name", None) in names or
                 (isinstance(node, ast.Assign) and any(
                     isinstance(t, ast.Name) and t.id == "GSPLAT_DEFAULTS"
                     for t in node.targets)))]
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
                       names=[ast.alias(name="annotations")], level=0), *selected],
                        type_ignores=[])
    env = dict(argparse=argparse, glob=glob, hashlib=hashlib, json=json, math=math,
               os=os, Path=Path, tempfile=tempfile, np=np, __doc__="test")
    env.update(extra)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    return env


class DiagnosticFlagTests(unittest.TestCase):
    def test_direct_cli_default_and_flag_are_forwarded(self):
        env = functions(WILOR, {"parse_args", "run_from_args"})
        for extra, expected in [([], False), (["--skip_diagnostic_renders"], True)]:
            with patch.object(sys, "argv", ["run", "--video_path", "clip.mp4",
                                         "--output_dir", "output", *extra]):
                args = env["parse_args"]()
            calls = []
            env["run_ego_wilor"] = lambda **kwargs: calls.append(kwargs)
            env["run_from_args"](args)
            self.assertEqual(calls[0]["skip_diagnostic_renders"], expected)

    def test_legacy_refinement_rejected_before_work(self):
        env = functions(WILOR, {"run_ego_wilor"})
        with self.assertRaisesRegex(ValueError, "legacy"):
            env["run_ego_wilor"]("clip.mp4", "output", "sam2",
                                run_refinement=True, skip_diagnostic_renders=True)

    def test_unified_dynhamr_rejects_unsupported_flag(self):
        env = functions(UNIFIED, {"_validate_args"})
        args = argparse.Namespace(depth_source=None, object_mesh=None,
                                  skip_object_scale_estimation=False,
                                  hand_tracking="dynhamr",
                                  run_gsplat_refinement=False,
                                  skip_diagnostic_renders=True)
        with self.assertRaisesRegex(ValueError, "--skip_diagnostic_renders"):
            env["_validate_args"](args)
        args.skip_diagnostic_renders = False
        env["_validate_args"](args)

    def test_unified_hamer_passes_flag(self):
        env = functions(UNIFIED, {"parse_args", "_add_weight_args", "_add_gsplat_args",
                                 "_run_hamer"})
        with patch.object(sys, "argv", ["run", "--video", "clip.mp4", "--output_dir",
                                     "output", "--object_prompt", "cup",
                                     "--hand_tracking", "hamer",
                                     "--skip_diagnostic_renders"]):
            args = env["parse_args"]()
        calls = []
        env["_import_run_ego_wilor"] = lambda: lambda **kwargs: calls.append(kwargs)
        env["_finalize_result_bundle"] = lambda *a, **kw: None
        env["_run_hamer"](args)
        self.assertTrue(calls[0]["skip_diagnostic_renders"])

    def test_wrapper_flag_suppresses_only_diagnostic_frequency(self):
        calls = []
        env = functions(WRAPPER, {"run_refine_simple"}, IMAGE_NAME="test-image",
                        MODULES_DIR="test-modules",
                        run_in_container=lambda **kwargs: calls.append(kwargs))
        common = dict(frames_dir="frames", depth_dir="depth", intrinsics_path="K.json",
                      object_mesh_path="mesh.obj", object_poses_dir="poses",
                      object_mask_dir="masks", refined_object_poses_dir="out",
                      overlay_path="overlay.mp4")
        env["run_refine_simple"](**common)
        env["run_refine_simple"](**common, skip_overlay=True)
        normal, skipped = [c["extra_args"] for c in calls]
        self.assertEqual(normal["render_every"], 25)
        self.assertFalse(normal["skip_overlay"])
        self.assertEqual(skipped["render_every"], 0)
        self.assertTrue(skipped["skip_overlay"])
        self.assertEqual({k: v for k, v in normal.items()
                          if k not in {"render_every", "skip_overlay"}},
                         {k: v for k, v in skipped.items()
                          if k not in {"render_every", "skip_overlay"}})

    def test_new_argument_preserves_existing_positional_tail(self):
        for path, name, tail in [(WRAPPER, "run_refine_simple", "dev"),
                                 (LIBRARY, "refine_simple", "device")]:
            env = functions(path, {name})
            parameters = list(inspect.signature(env[name]).parameters)
            self.assertEqual(parameters[-2:], [tail, "skip_overlay"])
            self.assertFalse(inspect.signature(env[name]).parameters["skip_overlay"].default)
            self.assertEqual(parameters[parameters.index("render_every") + 1], "progress_dir")


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = functions(WILOR, {"_frame_paths", "_contains_nonfinite_json_value",
                                     "_simple_refinement_completion",
                                     "_write_simple_refinement_completion"})
        self.transform = {"rotation": [1, 0, 0, 0], "translation": [0, 0, 0],
                          "scale": [1, 1, 1]}
        self.hand = {"mano": {"global_orient": [0] * 3, "hand_pose": [0] * 45,
                             "betas": [0] * 10}, "camera": {"cam_t": [0] * 3}}
        for name in ["frames", "input_object", "output_object", "input_hand",
                     "output_hand", "output_camera"]:
            (self.root / name).mkdir()
        # Nonconsecutive frames and a hand observed on only a subset.
        for f in [2, 5, 8]:
            (self.root / "frames" / f"{f:06d}.png").write_bytes(b"unused")
            for name in ["input_object", "output_object", "output_camera"]:
                self.write(name, f, self.transform)
        self.write("input_hand", 5, self.hand)
        self.write("output_hand", 5, self.hand)
        (self.root / "scale.json").write_text('{"scale":1.0}')
        self.kwargs = dict(
            frames_dir=str(self.root / "frames"),
            object_input_dir=str(self.root / "input_object"),
            object_output_dir=str(self.root / "output_object"),
            hand_dirs=[(str(self.root / "input_hand"), str(self.root / "output_hand")),
                       (None, None)],
            camera_output_dir=str(self.root / "output_camera"),
            scale_path=str(self.root / "scale.json"))

    def write(self, folder, frame, data):
        (self.root / folder / f"{frame:06d}.json").write_text(json.dumps(data))

    def check(self):
        return self.env["_simple_refinement_completion"](**self.kwargs)

    def test_legacy_complete_without_overlay_can_receive_receipt(self):
        receipt = self.check()
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt["object_frames"], [2, 5, 8])
        self.assertEqual(len(receipt["sha256"]), 8)
        path = self.root / "complete.json"
        self.env["_write_simple_refinement_completion"](
            str(path), receipt, "validated_existing_outputs")
        saved = json.loads(path.read_text())
        self.assertEqual(saved["origin"], "validated_existing_outputs")
        self.assertEqual(saved["sha256"], receipt["sha256"])
        self.assertEqual(list(self.root.glob(".refinement-complete-*")), [])

    def test_partial_camera_is_not_completion(self):
        (self.root / "output_camera/000008.json").unlink()
        self.assertIsNone(self.check())

    def test_partial_or_stale_hand_is_not_completion(self):
        (self.root / "output_hand/000005.json").unlink()
        self.assertIsNone(self.check())
        self.write("output_hand", 5, self.hand)
        self.write("output_hand", 2, self.hand)
        self.assertIsNone(self.check())

    def test_nonfinite_or_wrong_schema_is_not_completion(self):
        self.write("output_object", 2, {"rotation": [float("nan")] * 4,
                                     "translation": [0] * 3, "scale": [1] * 3})
        self.assertIsNone(self.check())
        self.write("output_object", 2, {"rotation": [1, 0, 0, 0]})
        self.assertIsNone(self.check())

    def test_invalid_scale_is_not_completion(self):
        (self.root / "scale.json").write_text('{"scale":0}')
        self.assertIsNone(self.check())

    def test_masked_background_and_provided_mesh_need_neither_output(self):
        self.kwargs["camera_output_dir"] = None
        self.kwargs["scale_path"] = None
        self.assertIsNotNone(self.check())

    def test_input_outside_rgb_does_not_require_extra_output(self):
        self.write("input_object", 99, self.transform)
        self.write("input_hand", 99, self.hand)
        self.assertIsNotNone(self.check())


if __name__ == "__main__":
    unittest.main()

