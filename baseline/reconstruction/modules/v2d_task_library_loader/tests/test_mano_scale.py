"""CPU regression tests for reconstruction's scaled-MANO consumer contract.

Run with ``python -m unittest discover -s <this directory> -p test_mano_scale.py``.
Requires numpy, scipy, torch, trimesh, and pyarrow, but no licensed MANO assets,
GPU, viewer, or IK runtime. Synthetic layers deliberately have a nonzero wrist
and unequal native-left / mirrored-right centroids.
"""

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[5]
LOADER_LIB = Path(__file__).resolve().parents[1] / "lib"
GROUNDING_SOURCE = REPO_ROOT / "baseline/robotic_grounding/source/robotic_grounding"
SCALE_FIELDS = (
    "mano_right_scale", "mano_left_scale",
    "mano_right_scale_center", "mano_left_scale_center",
)


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


@contextmanager
def _consumer_modules(real_layers=False):
    """Load production math; stub only unavailable assets and unrelated UI code."""
    base = _module(
        "v2d.task_library_loader.lib.dataset_loader_base",
        DatasetLoaderBase=object,
        SequenceInfo=types.SimpleNamespace,
        load_meshes_to_device=None,
        poses_to_root_position_and_axis_angle=None,
    )
    stubs = {base.__name__: base}
    if not real_layers:
        stubs.update({
            "manotorch": _module("manotorch", __path__=[]),
            "manotorch.axislayer": _module("manotorch.axislayer", AxisLayerFK=object),
            "manotorch.manolayer": _module("manotorch.manolayer", ManoLayer=object),
        })
    # Restore only our injected names. Clearing all modules imported within this
    # context can unload native-extension wrappers and make their next import fail.
    absent = object()
    names = [*stubs, "v2d.task_library_loader.lib.read_mano",
             "_mano_scale_ego_loader", "_mano_scale_data_logger"]
    saved = {name: sys.modules.get(name, absent) for name in names}
    sys.modules.update(stubs)
    try:
        with patch.object(sys, "path", [str(GROUNDING_SOURCE), *sys.path]):
            mano = _load_file(
                "v2d.task_library_loader.lib.read_mano", LOADER_LIB / "read_mano.py"
            )
            loader = _load_file("_mano_scale_ego_loader", LOADER_LIB / "ego_recon_loader.py")
            logger = _load_file(
                "_mano_scale_data_logger",
                GROUNDING_SOURCE / "robotic_grounding/retarget/data_logger.py",
            )
            yield mano, loader, logger
    finally:
        for name, original in saved.items():
            if original is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class _SyntheticLayer:
    """Simple articulated FK fixture, independent of the consumer's scale math."""

    def __init__(self, side):
        self.side = side
        mirror = np.array([-1.0, 1.0, 1.0]) if side == "left" else np.ones(3)
        self.wrist = np.array([0.025, -0.012, 0.008]) * mirror
        x = np.linspace(-0.03, 0.065, 21)
        self.joints = np.column_stack((x, x * x + 0.02, x * 0.15)) * mirror
        self.joints[0] = self.wrist
        self.vertices = np.array([
            [-0.02, 0.01, 0.005], [0.09, 0.02, -0.01],
            [0.03, 0.13, 0.02], [0.06, -0.02, 0.015],
        ]) * mirror
        if side == "left":
            # The native-left model need not have exactly the mirrored-right
            # centroid. A fixture with perfect symmetry would hide this bug.
            self.vertices += np.array([0.004, -0.002, 0.003])

    def __call__(self, pose_coeffs, betas):
        poses = pose_coeffs.detach().cpu().numpy()
        shape = betas.detach().cpu().numpy()
        rotation = Rotation.from_rotvec(poses[:, :3]).as_matrix()
        batch = len(poses)
        joints = np.broadcast_to(self.joints, (batch, 21, 3)).copy()
        vertices = np.broadcast_to(self.vertices, (batch, 4, 3)).copy()
        # A pose/shape dependent change prevents a rest-pose-only test from
        # accidentally accepting a centroid cached for the wrong hand pose.
        deformation = 0.01 * poses[:, 3] + 0.002 * shape[:, 0]
        joints[:, 1:, 1] += deformation[:, None]
        vertices[:, :, 1] += deformation[:, None]
        joints = np.einsum("bij,bnj->bni", rotation, joints - self.wrist) + self.wrist
        vertices = np.einsum("bij,bnj->bni", rotation, vertices - self.wrist) + self.wrist
        transforms = np.broadcast_to(np.eye(4), (batch, 16, 4, 4)).copy()
        transforms[:, :, :3, :3] = rotation[:, None]
        transforms[:, :, :3, 3] = joints[:, :16]
        kw = {"dtype": pose_coeffs.dtype, "device": pose_coeffs.device}
        return types.SimpleNamespace(
            joints=torch.tensor(joints, **kw), verts=torch.tensor(vertices, **kw),
            transforms_abs=torch.tensor(transforms, **kw),
        )


def _synthetic_mano(mano_module):
    mano = mano_module.MANO.__new__(mano_module.MANO)
    mano.device = torch.device("cpu")
    mano.right_mano_layer = _SyntheticLayer("right")
    mano.left_mano_layer = _SyntheticLayer("left")
    mano.right_axis_layer = mano.left_axis_layer = lambda transforms: (transforms, None, None)
    mano.right_faces = mano.left_faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    return mano


class TestManoScale(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module_context = _consumer_modules()
        cls.mano_module, cls.loader_module, cls.logger_module = cls.module_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.module_context.__exit__(None, None, None)

    def setUp(self):
        self.mano = _synthetic_mano(self.mano_module)
        self.arguments = dict(
            side="right", betas=torch.tensor([0.2] + [0.0] * 9),
            global_orient=torch.tensor([[0.15, -0.2, 0.1], [-0.1, 0.3, 0.2]]),
            transl=torch.tensor([[0.3, -0.1, 0.7], [-0.4, 0.25, 0.8]]),
            finger_pose=torch.full((2, 45), 0.07),
        )

    def test_missing_scale_and_unit_scale_preserve_legacy_geometry(self):
        legacy = self.mano.forward(**self.arguments)
        unit = self.mano.forward(**self.arguments, scale=torch.ones(2))
        pose = torch.cat((self.arguments["global_orient"], self.arguments["finger_pose"]), -1)
        layer = self.mano.right_mano_layer(pose, self.arguments["betas"].expand(2, -1))
        for key, raw in (("joints", layer.joints), ("vertices", layer.verts)):
            expected = raw + self.arguments["transl"][:, None]
            torch.testing.assert_close(legacy[key], expected)
            torch.testing.assert_close(unit[key], legacy[key])
        torch.testing.assert_close(unit["joints_wxyz"], legacy["joints_wxyz"])

    def test_nonunit_scale_changes_distances_but_preserves_centroid_and_rotation(self):
        scale = torch.tensor([0.6, 1.4])
        original = self.mano.forward(**self.arguments)
        scaled = self.mano.forward(**self.arguments, scale=scale)
        torch.testing.assert_close(scaled["vertices"].mean(1), original["vertices"].mean(1))
        for key in ("vertices", "joints"):
            old_distances = torch.cdist(original[key], original[key])
            new_distances = torch.cdist(scaled[key], scaled[key])
            torch.testing.assert_close(new_distances, old_distances * scale[:, None, None])
        torch.testing.assert_close(scaled["joints_wxyz"], original["joints_wxyz"])

    def test_explicit_scale_center_is_used_before_translation(self):
        center = torch.tensor([[0.02, 0.04, -0.03], [-0.05, 0.06, 0.01]])
        scale = torch.tensor([0.5, 1.6])
        original = self.mano.forward(**self.arguments)
        actual = self.mano.forward(**self.arguments, scale=scale, scale_center=center)
        world_center = center + self.arguments["transl"]
        for key in ("vertices", "joints"):
            expected = world_center[:, None] + scale[:, None, None] * (original[key] - world_center[:, None])
            torch.testing.assert_close(actual[key], expected)

    def test_left_bundle_centroid_uses_right_layer_then_mirror(self):
        args = self.arguments
        center = self.mano.bundle_centroid(
            args["global_orient"], args["finger_pose"], args["betas"].expand(2, -1), "left"
        )
        pose = torch.cat((args["global_orient"], args["finger_pose"]), -1)
        right = self.mano.right_mano_layer(pose, args["betas"].expand(2, -1))
        expected = (right.verts * torch.tensor([-1.0, 1.0, 1.0])).mean(1)
        torch.testing.assert_close(center, expected)
        native_pose = pose.reshape(2, 16, 3).clone()
        native_pose[:, :, 1:] *= -1
        native = self.mano.left_mano_layer(native_pose.reshape(2, 48), args["betas"].expand(2, -1))
        self.assertGreater(float(torch.max(torch.abs(center - native.verts.mean(1)))), 0.001)

    def test_invalid_scales_and_centers_are_rejected(self):
        for invalid in (0.0, -0.2, float("nan"), float("inf")):
            with self.subTest(scale=invalid), self.assertRaises(ValueError):
                self.mano.forward(**self.arguments, scale=torch.tensor([1.0, invalid]))
        for invalid in (float("nan"), float("inf")):
            with self.subTest(center=invalid), self.assertRaises(ValueError):
                self.mano.forward(**self.arguments, scale=torch.ones(2), scale_center=torch.full((2, 3), invalid))

    def _bundle_fixture(self):
        rotations = Rotation.from_euler("xyz", [[20, -15, 30], [-10, 35, 15], [5, 10, -25]], degrees=True).as_matrix()
        cameras = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
        cameras[:, :3, :3] = rotations
        cameras[:, :3, 3] = [[0.3, -0.2, 0.1], [0.4, -0.15, 0.2], [-0.2, 0.3, 0.4]]
        data = {"camera_to_world_transform": cameras}
        for side in ("left", "right"):
            data.update({
                f"hand_{side}_betas": np.array([0.2] + [0.0] * 9),
                f"hand_{side}_wrist_orient_in_camera": np.array([[0.2, -0.1, 0.3], [-0.2, 0.4, 0.1], [0.3, 0.1, -0.2]]),
                f"hand_{side}_wrist_trans_in_camera": np.array([[0.1, 0.2, 0.6], [0.2, -0.1, 0.7], [0.15, 0.05, 0.8]]),
                f"hand_{side}_finger_pose": np.full((3, 45), 0.08),
            })
        return data

    def _load_fixture(self, root, data):
        npz_path = Path(root) / "result.npz"
        np.savez(npz_path, **data)
        loader = self.loader_module.EgoReconDatasetLoader()
        loader._mano = self.mano
        align_r = Rotation.from_euler("xyz", [0.1, -0.2, 0.15]).as_matrix()
        align_t = np.array([0.02, -0.03, 0.1])
        loader._scene_align = lambda _: (align_r, align_t)
        info = types.SimpleNamespace(
            sequence_id="synthetic", source=types.SimpleNamespace(npz_path=npz_path, object_scale=1.0)
        )
        output = loader.load_mano_data(info, torch.device("cpu"))
        return output, align_r, align_t

    def test_loader_world_closure_with_rotations_translation_and_left_convention(self):
        data = self._bundle_fixture()
        data["hand_right_scale"] = np.array([0.6, 0.75, 1.3])
        data["hand_left_scale"] = np.array([1.2, 0.8, 0.65])
        with tempfile.TemporaryDirectory() as root:
            output, align_r, align_t = self._load_fixture(root, data)
        for side in ("right", "left"):
            orient = torch.tensor(data[f"hand_{side}_wrist_orient_in_camera"], dtype=torch.float32)
            fingers = torch.tensor(data[f"hand_{side}_finger_pose"], dtype=torch.float32)
            betas = torch.tensor(data[f"hand_{side}_betas"], dtype=torch.float32).expand(3, -1)
            # Build camera-space reference directly from raw synthetic model
            # outputs, rather than reusing MANO.forward's scale implementation.
            right_raw = self.mano.right_mano_layer(torch.cat((orient, fingers), -1), betas)
            centroid = right_raw.verts.mean(1).numpy()
            if side == "left":
                centroid *= np.array([-1, 1, 1])
                orient[:, 1:] *= -1
                fingers = fingers.reshape(3, 15, 3)
                fingers[:, :, 1:] *= -1
                fingers = fingers.reshape(3, 45)
            layer = getattr(self.mano, f"{side}_mano_layer")
            native = layer(torch.cat((orient, fingers), -1), betas)
            actual = self.mano.forward(
                side=side, betas=output[f"{side}_betas"], global_orient=output[f"{side}_global_orient"],
                transl=output[f"{side}_trans"], finger_pose=output[f"{side}_finger_pose"],
                scale=output[f"{side}_scale"], scale_center=output[f"{side}_scale_center"],
            )
            cameras = data["camera_to_world_transform"]
            rotation = align_r @ cameras[:, :3, :3]
            translation = (align_r @ cameras[:, :3, 3].T).T + align_t
            for key, raw in (("joints", native.joints.numpy()), ("vertices", native.verts.numpy())):
                camera = centroid[:, None] + data[f"hand_{side}_scale"][:, None, None] * (raw - centroid[:, None])
                camera += data[f"hand_{side}_wrist_trans_in_camera"][:, None]
                expected = np.einsum("bij,bnj->bni", rotation, camera) + translation[:, None]
                np.testing.assert_allclose(actual[key].numpy(), expected, atol=2e-7, rtol=1e-6)

    def test_loader_legacy_missing_scale_defaults_to_one(self):
        data = self._bundle_fixture()
        with tempfile.TemporaryDirectory() as root:
            legacy, _, _ = self._load_fixture(root, data)
            for side in ("left", "right"):
                data[f"hand_{side}_scale"] = 1.0
            unit, _, _ = self._load_fixture(root, data)
        for side in ("left", "right"):
            torch.testing.assert_close(legacy[f"{side}_scale"], torch.ones(3))
            for field in ("global_orient", "finger_pose", "trans", "scale", "scale_center"):
                torch.testing.assert_close(legacy[f"{side}_{field}"], unit[f"{side}_{field}"])

    def test_loader_rejects_nonpositive_or_nonfinite_bundle_scale(self):
        for side in ("left", "right"):
            for invalid in (0.0, -1.0, float("nan"), float("inf")):
                data = self._bundle_fixture()
                data[f"hand_{side}_scale"] = np.array([1.0, invalid, 1.0])
                with self.subTest(side=side, scale=invalid), tempfile.TemporaryDirectory() as root:
                    with self.assertRaisesRegex(ValueError, "hand scale"):
                        self._load_fixture(root, data)

    def _logger(self):
        cls = self.logger_module.ManoSharpaData
        metadata = {name: None for name, _, _, time_series in cls._field_specs if not time_series}
        metadata.update(sequence_id="synthetic", raw_motion_file="result.npz", robot_name="test", fps=30.0)
        logger = cls(**metadata)
        logger.log_timestep(mano_right_joints=np.zeros((21, 3)).tolist())
        return logger

    def test_new_parquet_roundtrip_preserves_scale_and_center(self):
        logger = self._logger()
        logger.log_timestep(
            mano_right_scale=0.75, mano_left_scale=1.25,
            mano_right_scale_center=[0.02, -0.01, 0.03],
            mano_left_scale_center=[-0.025, -0.01, 0.035],
        )
        with tempfile.TemporaryDirectory() as root:
            logger.save_to_parquet(root)
            restored = self.logger_module.ManoSharpaData.from_parquet(root)
        for field in SCALE_FIELDS:
            np.testing.assert_allclose(getattr(restored, field), getattr(logger, field), atol=1e-7)
        np.testing.assert_array_equal(restored.mano_right_joints, logger.mano_right_joints)

    def test_legacy_parquet_without_scale_columns_still_loads(self):
        logger = self._logger()
        old_schema = pa.schema([field for field in logger._schema if field.name not in SCALE_FIELDS])
        old_row = {key: value for key, value in logger.to_dict().items() if key not in SCALE_FIELDS}
        with tempfile.TemporaryDirectory() as root:
            pq.write_table(pa.Table.from_pylist([old_row], schema=old_schema), Path(root) / "legacy.parquet")
            restored = self.logger_module.ManoSharpaData.from_parquet(root)
        for field in SCALE_FIELDS:
            self.assertIsNone(getattr(restored, field))
        self.assertEqual(restored.sequence_id, "synthetic")
        np.testing.assert_array_equal(restored.mano_right_joints, logger.mano_right_joints)

    def test_mixed_parquet_root_preserves_new_fields_when_legacy_sorts_first(self):
        old = self._logger()
        old.sequence_id = "legacy"
        new = self._logger()
        new.sequence_id = "scaled"
        new.log_timestep(
            mano_right_scale=0.65, mano_left_scale=1.15,
            mano_right_scale_center=[0.02, -0.01, 0.03],
            mano_left_scale_center=[-0.025, -0.01, 0.035],
        )
        old_schema = pa.schema([field for field in old._schema if field.name not in SCALE_FIELDS])
        old_row = {key: value for key, value in old.to_dict().items() if key not in SCALE_FIELDS}
        with tempfile.TemporaryDirectory() as root:
            # Explicit names reproduce schema inference from the older fragment.
            pq.write_table(pa.Table.from_pylist([old_row], schema=old_schema), Path(root) / "a_legacy.parquet")
            pq.write_table(pa.Table.from_pylist([new.to_dict()], schema=new._schema), Path(root) / "b_new.parquet")
            cls = self.logger_module.ManoSharpaData
            restored_new = cls.from_parquet(root, filters=[("sequence_id", "=", "scaled")])
            restored_old = cls.from_parquet(root, filters=[("sequence_id", "=", "legacy")])
        self.assertEqual(restored_new.sequence_id, "scaled")
        self.assertEqual(restored_old.sequence_id, "legacy")
        for field in SCALE_FIELDS:
            with self.subTest(field=field):
                self.assertIsNotNone(getattr(restored_new, field))
                np.testing.assert_allclose(getattr(restored_new, field), getattr(new, field), atol=1e-7)
                self.assertIsNone(getattr(restored_old, field))
        np.testing.assert_array_equal(restored_new.mano_right_joints, new.mano_right_joints)
        np.testing.assert_array_equal(restored_old.mano_right_joints, old.mano_right_joints)

    def test_reading_inside_hive_partition_recovers_keys_and_scale(self):
        logger = self._logger()
        logger.log_timestep(
            mano_right_scale=0.75, mano_left_scale=1.25,
            mano_right_scale_center=[0.02, -0.01, 0.03],
            mano_left_scale_center=[-0.025, -0.01, 0.035],
        )
        with tempfile.TemporaryDirectory() as root:
            logger.save_to_parquet(root, partition_cols=["sequence_id", "robot_name"])
            partition = Path(root) / "sequence_id=synthetic" / "robot_name=test"
            restored = self.logger_module.ManoSharpaData.from_parquet(str(partition))
        self.assertEqual(restored.sequence_id, logger.sequence_id)
        self.assertEqual(restored.robot_name, logger.robot_name)
        for field in SCALE_FIELDS:
            np.testing.assert_allclose(getattr(restored, field), getattr(logger, field), atol=1e-7)


@unittest.skipUnless(os.environ.get("MANO_ASSETS_ROOT"), "optional licensed MANO assets not supplied")
class TestRealManoScale(unittest.TestCase):
    """Optional smoke test with user-provided models; assets are never downloaded."""

    def test_licensed_fk_camera_to_world_closure(self):
        with _consumer_modules(real_layers=True) as (mano_module, loader_module, _):
            self.loader_module = loader_module
            self.mano = mano_module.MANO(os.environ["MANO_ASSETS_ROOT"], device=torch.device("cpu"))
            data = TestManoScale._bundle_fixture(self)
            for side in ("left", "right"):
                data[f"hand_{side}_scale"] = np.array([0.6, 0.9, 1.2])
            with tempfile.TemporaryDirectory() as root:
                output, align_r, align_t = TestManoScale._load_fixture(self, root, data)
            cameras = data["camera_to_world_transform"]
            rotation = align_r @ cameras[:, :3, :3]
            translation = (align_r @ cameras[:, :3, 3].T).T + align_t
            for side in ("left", "right"):
                orient = torch.tensor(data[f"hand_{side}_wrist_orient_in_camera"], dtype=torch.float32)
                fingers = torch.tensor(data[f"hand_{side}_finger_pose"], dtype=torch.float32)
                betas = torch.tensor(data[f"hand_{side}_betas"], dtype=torch.float32).expand(3, -1)
                center = self.mano.bundle_centroid(orient, fingers, betas, side)
                if side == "left":
                    orient[:, 1:] *= -1
                    fingers = fingers.reshape(3, 15, 3)
                    fingers[:, :, 1:] *= -1
                    fingers = fingers.reshape(3, 45)
                camera = self.mano.forward(
                    side=side, betas=betas, global_orient=orient, finger_pose=fingers,
                    transl=torch.tensor(data[f"hand_{side}_wrist_trans_in_camera"], dtype=torch.float32),
                    scale=output[f"{side}_scale"], scale_center=center,
                )
                world = self.mano.forward(
                    side=side, betas=output[f"{side}_betas"], global_orient=output[f"{side}_global_orient"],
                    finger_pose=output[f"{side}_finger_pose"], transl=output[f"{side}_trans"],
                    scale=output[f"{side}_scale"], scale_center=output[f"{side}_scale_center"],
                )
                for key in ("joints", "vertices"):
                    expected = np.einsum("bij,bnj->bni", rotation, camera[key].numpy()) + translation[:, None]
                    np.testing.assert_allclose(world[key].numpy(), expected, atol=2e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
