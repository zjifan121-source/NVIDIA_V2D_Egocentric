# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for V2D Track 3 by Edmond Wang, 2026-10: clip frame rate overridable via V2D_EGO_FPS;
# several rigid objects per bundle (object_names, object_<i>_to_world_transform, object_<i>_scale,
# mesh_<i>.obj for i >= 1). Single-object bundles load exactly as before.

"""Load an ego hand-object reconstruction (``result.npz``) into ManoSharpaData.

Input is the portable ``result/`` bundle written by ``v2d_common.result_bundle``
(produced by the ego e2e / WiLoR pipelines):

  <clip_dir>/
    outputs/result/
      result.npz          # camera + object + both-hand trajectories
      mesh.obj            # textured object mesh (metric, scaled by object_scale)
      material.mtl, *.png
      manifest.json

``result.npz`` carries raw MANO params (betas, wrist orient/trans, 15x3 finger
pose) in the per-frame CAMERA frame, plus per-frame camera-to-world transforms and
the object pose in the world frame (world = camera frame 0). The dual-hand retarget
IK consumes ``mano_*_joints`` / ``mano_*_joints_wxyz`` produced by MANO forward
kinematics, so this loader runs FK and writes the ManoSharpaData ``_loaded`` Parquet.

Coordinate frame:
  Everything is emitted in the static WORLD frame (= camera frame 0) so hands and
  object share one frame across the trajectory. The hand params in ``result.npz``
  are per-frame camera frame, so they are re-posed to world. With
  ``center_idx=None`` the MANO root sits at ``J0_template(betas) + transl`` (transl
  added post-FK), so a world re-pose is, per frame t:

    global_orient_world = rotvec( R_c2w @ R(orient_cam) )    # = rot block of wrist_to_world
    trans_world         = R_c2w @ (J0_template + cam_t) + t_c2w - J0_template

  (mirrors hot3d_loader's J0 handling, generalized to a per-frame SE(3)). A numeric
  gate asserts the resulting world joints equal R_c2w @ joints_cam + t_c2w.

MANO format:
  ``hand_*_finger_pose`` is full 45-DOF axis-angle (15x3), matching the MANO layer
  used to produce it (``use_pca=False, center_idx=None``; manotorch default
  ``flat_hand_mean=True``). No PCA expansion.

Runs stage 1 of the two-stage pipeline:
  1. python -m v2d.task_library_loader.lib.run_loader --dataset ego_recon --save
  2. python scripts/retarget/ego_recon_to_sharpa.py --save  (robotic_grounding)
"""

import argparse
import logging
import os
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

warnings.filterwarnings("ignore", category=DeprecationWarning, module="mano")

from robotic_grounding.retarget import HUMAN_MOTION_DATA_DIR  # noqa: E402
from robotic_grounding.retarget.dataset_registry import get_dataset_config  # noqa: E402
from robotic_grounding.retarget.naming import make_usd_safe  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from v2d.task_library_loader.lib.dataset_loader_base import (  # noqa: E402
    DatasetLoaderBase,
    SequenceInfo,
    load_meshes_to_device,
    poses_to_root_position_and_axis_angle,
)
from v2d.task_library_loader.lib.read_mano import MANO  # noqa: E402

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="manotorch")

DEFAULT_DATASET_ROOT = Path("data")
# Unlike the other datasets, ego_recon keeps every per-sequence artifact in ONE
# directory: the motion Parquet, the object mesh and material, the generated
# collision STL and the rigid URDF all live under ``ego_recon/processed``. A
# clip's assets are self-contained and move as a unit, and
# ``generate_rigid_urdfs.py --dataset ego_recon`` discovers objects from there.
EGO_DATASET_DIR = HUMAN_MOTION_DATA_DIR / "ego_recon"
EGO_MESH_DIR = EGO_DATASET_DIR / "processed"
EGO_URDF_DIR = EGO_DATASET_DIR / "processed"
# The loaded Parquet is a regenerable intermediate, so it is written outside the
# committed asset tree (see DatasetConfig.loaded_in_intermediate).
LOADED_SAVE_DIR = get_dataset_config("ego_recon").loaded_data_dir
# Track 3 clips are 20.25411910142744 fps (dataset meta/info.json), not 30; set V2D_EGO_FPS.
EGO_FPS = float(os.environ.get("V2D_EGO_FPS", "30.0"))

# result/ bundle layout relative to a clip directory.
_RESULT_SUBPATH = Path("outputs") / "result"
_RESULT_NPZ = "result.npz"
_RESULT_MESH = "mesh.obj"

# World-frame re-posing numeric gate: max abs joint error (meters) allowed between
# the FK'd world joints and the directly camera->world transformed camera joints.
_WORLD_REPOSE_TOL = 1e-4


@dataclass
class EgoReconSequenceSource:
    """Dataset-specific metadata stored in SequenceInfo.source."""

    result_dir: Path  # <clip>/outputs/result
    npz_path: Path
    mesh_path: Path
    object_scale: float
    # Further objects of a multi-object bundle: (pose key in result.npz, mesh path, scale).
    extra_objects: tuple[tuple[str, Path, float], ...] = ()

    def objects(self) -> list[tuple[str, Path, float]]:
        """(pose key, mesh path, scale) for every object, primary first."""
        return [("object_to_world_transform", self.mesh_path, self.object_scale), *self.extra_objects]


class EgoReconDatasetLoader(DatasetLoaderBase):
    """Loader for ego reconstruction ``result.npz`` bundles."""

    def __init__(self) -> None:
        """Initialize caches; the MANO wrapper is built lazily from --mano_model_dir."""
        super().__init__()
        self._mano: MANO | None = None
        # J0_template (rest wrist position) per side, keyed by side; depends on betas
        # so it is recomputed per sequence in load_mano_data.
        self._object_scale_cache: dict[str, float] = {}
        # Per-sequence world ground-alignment SE(3): (R_align (3,3), t_align (3,)).
        self._align_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------ helpers
    def _ensure_mano(self) -> MANO:
        """Build the MANO wrapper from --mano_model_dir (once), matching get_mano_kwargs."""
        if self._mano is None:
            self._mano = MANO(
                mano_assets_root=str(self._args.mano_model_dir),
                gender="neutral",
                device=torch.device("cpu"),
                **self.get_mano_kwargs(),
            )
        return self._mano

    def _j0_template(self, side: str, betas: np.ndarray) -> np.ndarray:
        """Rest wrist (joint 0) world position for this shape, transl/pose = 0.

        With center_idx=None the root joint is invariant to global_orient and pose,
        and transl is added post-FK, so joints[0] = J0_template + transl.
        """
        mano = self._ensure_mano()
        zero = torch.zeros(1, 3)
        out = mano.forward(
            side=side,
            betas=torch.from_numpy(betas).float().reshape(1, 10),
            global_orient=zero,
            transl=zero,
            finger_pose=torch.zeros(1, 45),
        )
        return out["joints"][0, 0].cpu().numpy().astype(np.float64)

    def _scene_align(
        self, sequence_info: SequenceInfo
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ground-align the world so frame 0 has the object flat on a z=0 table.

        The reconstruction world frame is camera-frame-0 (tilted: the head camera
        looks down at the table), so nothing is gravity-aligned. We estimate "up"
        from the object's oriented bounding box at frame 0 — for an object resting
        on a table its thinnest OBB axis is the table normal — then build a global
        SE(3) that rotates that axis to +z and places the object's bottom at z=0,
        centered in xy. Applied to every *_to_world transform (camera, object,
        hands) it yields a gravity-aligned scene suitable for sim.

        Returns (R_align, t_align): world' = R_align @ world + t_align.
        """
        sid = sequence_info.sequence_id
        if sid in self._align_cache:
            return self._align_cache[sid]

        src: EgoReconSequenceSource = sequence_info.source
        with np.load(src.npz_path) as npz:
            T0 = npz["object_to_world_transform"][0].astype(np.float64)
        mesh = trimesh.load(str(src.mesh_path))
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        verts = np.asarray(mesh.vertices, dtype=np.float64) * src.object_scale
        vw0 = (T0[:3, :3] @ verts.T).T + T0[
            :3, 3
        ]  # object verts, world(=cam0), frame 0

        if bool(getattr(self._args, "no_ground_align", False)):
            # Bundle is already gravity-aligned upstream (e.g. GeoCalib
            # --gravity_align --gravity_align_target z_up). Keep that world
            # rotation as-is; only normalize position (center xy, bottom -> z_offset).
            R_align = np.eye(3)
            vw0_rot = vw0
        else:
            center = vw0.mean(axis=0)
            centered = vw0 - center
            # PCA axes; pick the axis with the smallest physical extent as "up".
            _evals, evecs = np.linalg.eigh(centered.T @ centered)  # columns = axes
            extents = [float(np.ptp(centered @ evecs[:, i])) for i in range(3)]
            up = evecs[:, int(np.argmin(extents))]
            # Disambiguate sign: "up" (table normal) points toward the camera at the
            # origin, i.e. away from the object center -> up . center < 0.
            if float(np.dot(up, center)) > 0.0:
                up = -up

            z = np.array([0.0, 0.0, 1.0])
            R_align = Rotation.align_vectors(z[None, :], up[None, :])[
                0
            ].as_matrix()  # R_align @ up ~= +z
            vw0_rot = (R_align @ vw0.T).T
        # Extra heading correction: rotate the whole scene about world +z.
        # Gravity alignment fixes pitch/roll but leaves yaw arbitrary, so the scene
        # can come out rotated (e.g. -90 deg) about the vertical axis.
        yaw_deg = float(getattr(self._args, "ground_yaw_offset_deg", 0.0))
        if yaw_deg != 0.0:
            Rz = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
            R_align = Rz @ R_align
            vw0_rot = (R_align @ vw0.T).T
        z_offset = float(getattr(self._args, "ground_z_offset", 0.0))
        t_align = np.array(
            [
                -float(vw0_rot[:, 0].mean()),  # center object in x
                -float(vw0_rot[:, 1].mean()),  # center object in y
                -float(vw0_rot[:, 2].min()) + z_offset,  # object bottom -> z = z_offset
            ]
        )
        self._align_cache[sid] = (R_align, t_align)
        return R_align, t_align

    # ------------------------------------------------------------ abstract API
    def list_sequences(self, args: Any) -> list[SequenceInfo]:
        """Discover result bundles under --dataset_root.

        A clip directory is any dir containing ``outputs/result/result.npz``.
        ``--dataset_root`` may itself be a single clip directory. Pass
        ``--sequence_name`` to override the (single) discovered sequence id.
        """
        root = Path(args.dataset_root)
        object_name = getattr(args, "object_name", None) or "object"
        safe_object = make_usd_safe(object_name)
        # Where the result/ bundle sits under a clip dir. Override e.g. to
        # "outputs_geocalib/result" to load a gravity-aligned (GeoCalib) bundle.
        result_subpath = Path(getattr(args, "result_subpath", None) or _RESULT_SUBPATH)

        clip_dirs: list[Path] = []
        if (root / result_subpath / _RESULT_NPZ).exists():
            clip_dirs = [root]
        else:
            clip_dirs = sorted(
                p
                for p in root.iterdir()
                if p.is_dir() and (p / result_subpath / _RESULT_NPZ).exists()
            )

        sequence_name = getattr(args, "sequence_name", None)
        if sequence_name and len(clip_dirs) != 1:
            print(
                f"Warning: --sequence_name set but {len(clip_dirs)} clips discovered; "
                "ignoring override and using per-clip directory names."
            )
            sequence_name = None

        out: list[SequenceInfo] = []
        for clip_dir in clip_dirs:
            result_dir = clip_dir / result_subpath
            npz_path = result_dir / _RESULT_NPZ
            mesh_path = result_dir / _RESULT_MESH
            object_scale = 1.0
            names: list[str] = []
            extra: list[tuple[str, Path, float]] = []
            try:
                with np.load(npz_path) as npz:
                    object_scale = float(npz["object_scale"])
                    if "object_names" in npz.files:
                        names = [str(n) for n in npz["object_names"]]
                        extra = [(f"object_{i}_to_world_transform", result_dir / f"mesh_{i}.obj",
                                  float(npz[f"object_{i}_scale"])) for i in range(1, len(names))]
            except (KeyError, ValueError, OSError):
                pass
            body_names = ([make_usd_safe(f"{object_name}_{n}") for n in names]
                          if len(names) > 1 else [safe_object])
            sequence_id = sequence_name if sequence_name else clip_dir.name
            out.append(
                SequenceInfo(
                    sequence_id=sequence_id,
                    raw_motion_file=str(npz_path),
                    object_name=object_name,
                    object_body_names=body_names,
                    source=EgoReconSequenceSource(
                        result_dir=result_dir,
                        npz_path=npz_path,
                        mesh_path=mesh_path,
                        object_scale=object_scale,
                        extra_objects=tuple(extra),
                    ),
                )
            )
        return out

    def load_mano_data(
        self, sequence_info: SequenceInfo, device: torch.device
    ) -> dict[str, Any]:
        """Read result.npz hand params and re-pose them to the world frame."""
        src: EgoReconSequenceSource = sequence_info.source
        self._object_scale_cache[sequence_info.sequence_id] = src.object_scale

        npz = np.load(src.npz_path)
        cam2world = npz["camera_to_world_transform"].astype(np.float64)  # (N,4,4)
        H = cam2world.shape[0]
        # Ground-align the world (world' = R_align @ world + t_align) so frame 0 has
        # the object flat on a z=0 table. Applied to camera_to_world here, it carries
        # through to the hand world joints (joints_world = camera_to_world o FK_cam).
        R_align, t_align = self._scene_align(sequence_info)
        Rc2w = np.einsum("ij,njk->nik", R_align, cam2world[:, :3, :3])  # (N,3,3)
        tc2w = (R_align @ cam2world[:, :3, 3].T).T + t_align  # (N,3)

        out: dict[str, Any] = {"H": H}
        for side in ("right", "left"):
            betas = npz[f"hand_{side}_betas"].astype(np.float32).reshape(10)
            orient_cam = npz[f"hand_{side}_wrist_orient_in_camera"].astype(np.float64)
            cam_t = npz[f"hand_{side}_wrist_trans_in_camera"].astype(np.float64)
            finger_pose = (
                npz[f"hand_{side}_finger_pose"].astype(np.float32).reshape(H, 45)
            )

            hand_scale = np.broadcast_to(
                np.asarray(npz.get(f"hand_{side}_scale", 1.0), dtype=np.float64), (H,)
            ).copy()
            if not np.isfinite(hand_scale).all() or not (hand_scale > 0).all():
                raise ValueError(f"Invalid {side} hand scale")
            # Preserve reconstruction's centroid convention, including right-MANO
            # plus mirror-X for LEFT. Native-left conversion below is unchanged.
            with torch.no_grad():
                center_cam = self._ensure_mano().bundle_centroid(
                    torch.from_numpy(orient_cam).float(),
                    torch.from_numpy(finger_pose).float(),
                    torch.from_numpy(betas).float().reshape(1, 10).expand(H, 10), side
                ).cpu().numpy().astype(np.float64)

            # WiLoR/HaMeR store LEFT-hand MANO params in RIGHT-hand parameter space
            # (HaMeR flips left->right and only mirrors the OUTPUT mesh). manotorch's
            # native left layer (used by read_mano) expects native-left params, so
            # apply the standard MANO mirror: negate the y,z axis-angle components of
            # global_orient and every finger joint. Without this the left hand is
            # mirrored (~0.20 m vertex error; native-left+flip matches the recon
            # right-layer+mirror-X mesh to ~5e-4 m).
            if side == "left":
                orient_cam = orient_cam.copy()
                orient_cam[:, 1:] *= -1.0
                fp = finger_pose.reshape(H, 15, 3).copy()
                fp[:, :, 1:] *= -1.0
                finger_pose = fp.reshape(H, 45)

            j0 = self._j0_template(side, betas)  # (3,) native-side rest wrist

            # global_orient_world = rotvec(R_c2w @ R(orient_cam)), computed from the
            # (mirror-corrected) camera-frame orientation rather than the stored
            # wrist_to_world block (which was built from the uncorrected left params).
            Rcam = Rotation.from_rotvec(orient_cam).as_matrix()  # (H,3,3)
            global_orient_world = (
                Rotation.from_matrix(np.einsum("nij,njk->nik", Rc2w, Rcam))
                .as_rotvec()
                .astype(np.float32)
            )
            # trans_world = R_c2w @ (J0 + cam_t) + t_c2w - J0
            trans_world = (
                np.einsum("nij,nj->ni", Rc2w, cam_t + j0) + tc2w - j0
            ).astype(np.float32)

            # In zero-translation world-MANO coordinates: v_w = R(v_c-J0)+J0.
            center_world = np.einsum("nij,nj->ni", Rc2w, center_cam - j0) + j0
            self._validate_world_repose(
                side,
                betas,
                orient_cam,
                cam_t,
                finger_pose,
                Rc2w,
                tc2w,
                global_orient_world,
                trans_world,
                hand_scale,
                center_cam,
                center_world,
            )

            out[f"{side}_global_orient"] = torch.from_numpy(global_orient_world).to(
                device
            )
            out[f"{side}_finger_pose"] = torch.from_numpy(finger_pose).to(device)
            out[f"{side}_trans"] = torch.from_numpy(trans_world).to(device)
            out[f"{side}_betas"] = torch.from_numpy(betas).to(device)
            out[f"{side}_fitting_err"] = torch.zeros(H, device=device)
            out[f"{side}_scale"] = torch.from_numpy(hand_scale).float().to(device)
            out[f"{side}_scale_center"] = torch.from_numpy(center_world).float().to(device)

        npz.close()
        return out

    def _validate_world_repose(
        self,
        side: str,
        betas: np.ndarray,
        orient_cam: np.ndarray,
        cam_t: np.ndarray,
        finger_pose: np.ndarray,
        Rc2w: np.ndarray,
        tc2w: np.ndarray,
        global_orient_world: np.ndarray,
        trans_world: np.ndarray,
        hand_scale: np.ndarray,
        center_cam: np.ndarray,
        center_world: np.ndarray,
    ) -> None:
        """HARD GATE: FK'd world joints must equal R_c2w @ joints_cam + t_c2w.

        Checks a sparse subset of frames so the cost stays negligible.
        """
        H = len(cam_t)
        idx = np.unique(np.linspace(0, H - 1, num=min(H, 8)).astype(int))
        mano = self._ensure_mano()
        betas_t = torch.from_numpy(betas).float().reshape(1, 10).expand(len(idx), 10)
        finger_t = torch.from_numpy(finger_pose[idx]).float()

        joints_cam = (
            mano.forward(
                side=side,
                betas=betas_t,
                global_orient=torch.from_numpy(orient_cam[idx]).float(),
                transl=torch.from_numpy(cam_t[idx]).float(),
                finger_pose=finger_t,
                scale=torch.from_numpy(hand_scale[idx]).float(),
                scale_center=torch.from_numpy(center_cam[idx]).float(),
            )["joints"]
            .cpu()
            .numpy()
            .astype(np.float64)
        )  # (k,21,3)
        joints_world_expected = (
            np.einsum("kij,knj->kni", Rc2w[idx], joints_cam) + tc2w[idx][:, None, :]
        )

        joints_world_actual = (
            mano.forward(
                side=side,
                betas=betas_t,
                global_orient=torch.from_numpy(global_orient_world[idx]).float(),
                transl=torch.from_numpy(trans_world[idx]).float(),
                finger_pose=finger_t,
                scale=torch.from_numpy(hand_scale[idx]).float(),
                scale_center=torch.from_numpy(center_world[idx]).float(),
            )["joints"]
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        max_err = float(np.abs(joints_world_actual - joints_world_expected).max())
        if max_err > _WORLD_REPOSE_TOL:
            raise ValueError(
                f"ego_recon world re-pose gate failed for {side} hand: max joint "
                f"error {max_err:.6e} m > tol {_WORLD_REPOSE_TOL:.0e}. The "
                "camera->world MANO composition is wrong."
            )
        print(
            f"[ego_recon] {side} world-repose gate OK (max joint err {max_err:.2e} m)"
        )

    def load_object_data(self, sequence_info: SequenceInfo) -> dict[str, Any]:
        """Rigid object poses: object_to_world_transform (+ object_<i>_to_world_transform)."""
        src: EgoReconSequenceSource = sequence_info.source
        # Same ground-alignment as the hands (world' = R_align @ world + t_align).
        R_align, t_align = self._scene_align(sequence_info)
        out: dict[str, Any] = {}
        with np.load(src.npz_path) as npz:
            for body_name, (key, _mesh, _scale) in zip(sequence_info.object_body_names, src.objects()):
                poses = npz[key].astype(np.float64)  # (N,4,4)
                aligned = poses.copy()
                aligned[:, :3, :3] = np.einsum("ij,njk->nik", R_align, poses[:, :3, :3])
                aligned[:, :3, 3] = (R_align @ poses[:, :3, 3].T).T + t_align
                root_pos, root_aa = poses_to_root_position_and_axis_angle(aligned)
                out[body_name] = (aligned, root_pos, root_aa, None)
        return out

    def load_object_meshes(
        self,
        sequence_info: SequenceInfo,
        device: torch.device,
    ) -> tuple[
        dict[str, Any],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        bool,
    ]:
        """Load the textured object mesh(es) from the result bundle (metric)."""
        src: EgoReconSequenceSource = sequence_info.source
        merged: list[Any] = [{}, {}, {}, {}, {}, True]
        for body_name, (_key, mesh_path, scale) in zip(sequence_info.object_body_names, src.objects()):
            part = load_meshes_to_device({body_name: str(mesh_path)}, device, vertex_scale=scale)
            for i in range(5):
                merged[i].update(part[i])
            merged[5] = merged[5] and part[5]
        return tuple(merged)

    def get_mano_kwargs(self) -> dict[str, Any]:
        """Ego/WiLoR hand params use use_pca=False, center_idx=None, flat_hand_mean=True.

        These match the MANO layers that produced result.npz finger poses
        (tracks_from_wilor_masks._mano_layer and gsplat pose_fields), so feeding the
        stored finger pose back through FK reproduces the same hand.
        """
        return {"flat_hand_mean": True, "center_idx": None}

    def get_fps(self) -> float:
        """Ego clip frame rate (30 fps unless V2D_EGO_FPS is set)."""
        return EGO_FPS

    @staticmethod
    def _mesh_signature(path: Path, scale: float) -> tuple[int, tuple[float, ...]]:
        """(vertex count, rounded xyz extents) — a cheap geometry fingerprint."""
        mesh = trimesh.load(str(path))
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        verts = np.asarray(mesh.vertices, dtype=np.float64) * float(scale)
        extents = np.round(verts.max(axis=0) - verts.min(axis=0), 4)
        return len(mesh.vertices), tuple(extents.tolist())

    def _write_object_asset(
        self, src_obj: Path, target: Path, scale: float
    ) -> None:
        """Copy the clip's OBJ (+ material) to *target*, scaled.

        Material file refs are renamed to ``{stem}_*`` so no other object's shared
        material is clobbered.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        stem = target.stem
        src_obj = Path(src_obj)
        mtl_name: str | None = None
        obj_out: list[str] = []
        for line in src_obj.read_text().splitlines():
            parts = line.split()
            if parts and parts[0] == "mtllib":
                mtl_name = " ".join(parts[1:])
                obj_out.append(f"mtllib {stem}.mtl")
            elif parts and parts[0] == "v" and len(parts) >= 4 and scale != 1.0:
                x, y, z = (float(parts[i]) * scale for i in (1, 2, 3))
                obj_out.append(f"v {x} {y} {z}")
            else:
                obj_out.append(line)
        target.write_text("\n".join(obj_out) + "\n")
        src_mtl = src_obj.parent / mtl_name if mtl_name else None
        if src_mtl is None or not src_mtl.exists():
            return
        mtl_out: list[str] = []
        for line in src_mtl.read_text().splitlines():
            parts = line.split()
            if parts and parts[0].startswith("map_") and len(parts) >= 2:
                tex = parts[-1]
                new_tex = f"{stem}_{Path(tex).name}"
                if (src_mtl.parent / tex).exists():
                    shutil.copy(src_mtl.parent / tex, target.parent / new_tex)
                mtl_out.append(" ".join(parts[:-1] + [new_tex]))
            else:
                mtl_out.append(line)
        (target.parent / f"{stem}.mtl").write_text("\n".join(mtl_out) + "\n")

    def _ensure_object_asset(self, sequence_info: SequenceInfo, index: int = 0) -> Path:
        """Install this clip's mesh as the object asset, guarding against name reuse.

        Installs the reconstructed mesh (B1) and rejects an ``object_name`` already
        used by a different reconstruction (B2).

        ``get_object_mesh_paths`` references ``<mesh_dir>/<body>.obj`` *by name*, but
        alignment (``_scene_align``) and object FK use the clip's *own* ``mesh.obj``.
        If the asset at that name is a different reconstruction, sim renders a mesh
        whose local frame differs from the one the poses were aligned to, tilting the
        object by exactly that inter-mesh angle. So copy the clip mesh into the asset
        path here. If an asset already exists but differs, raise: reusing a name across
        distinct reconstructions silently mis-aligns the object -- use per-clip names.
        """
        src: EgoReconSequenceSource = sequence_info.source
        _key, mesh_path, scale = src.objects()[index]
        target = Path(self._args.mesh_dir) / f"{sequence_info.object_body_names[index]}.obj"
        scale = float(scale)
        clip_sig = self._mesh_signature(mesh_path, scale)
        if target.exists():
            tgt_sig = self._mesh_signature(target, 1.0)
            if tgt_sig[0] != clip_sig[0] or not np.allclose(
                tgt_sig[1], clip_sig[1], atol=1e-3
            ):
                raise ValueError(
                    f"Object mesh asset already exists but DIFFERS from this clip:\n"
                    f"  asset: {target} (verts={tgt_sig[0]}, extents={tgt_sig[1]})\n"
                    f"  clip : {mesh_path} (verts={clip_sig[0]}, extents={clip_sig[1]})\n"
                    f"Reusing --object_name {sequence_info.object_name!r} across "
                    f"different reconstructions mis-aligns the object (alignment uses "
                    f"the clip mesh; sim renders this asset). Use a distinct "
                    f"--object_name (one per reconstruction)."
                )
            return target  # already installed and consistent
        self._write_object_asset(mesh_path, target, scale)
        print(f"[ego_recon] installed object mesh asset -> {target}")
        return target

    def get_object_mesh_paths(self, sequence_info: SequenceInfo) -> list[str]:
        """Mesh path baked into the Parquet, under --mesh_dir (like hot3d).

        Installs this clip's own reconstructed mesh at that path (B1) so the mesh
        used for alignment == the mesh rendered in sim, and errors if a different
        reconstruction already occupies the name (B2). Run
        scripts/generate_rigid_urdfs.py --dataset ego_recon afterward for the URDF.
        """
        return [str(self._ensure_object_asset(sequence_info, i))
                for i in range(len(sequence_info.object_body_names))]

    def get_object_urdf_paths(self, sequence_info: SequenceInfo) -> list[str]:
        """Rigid URDF path baked into the Parquet, under --object_model_root.

        URDFs are generated by scripts/generate_rigid_urdfs.py --dataset ego_recon.
        """
        return [str(Path(self._args.object_model_root) / f"{body_name}_rigid.urdf")
                for body_name in sequence_info.object_body_names]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ego_recon loader script."""
    parser = argparse.ArgumentParser(
        description="Load ego result.npz bundles into ManoSharpaData (MANO + object)."
    )
    DatasetLoaderBase.add_common_args(
        parser,
        dataset_root=DEFAULT_DATASET_ROOT,
        object_model_root=EGO_URDF_DIR,
        mesh_dir=EGO_MESH_DIR,
        output_dir=LOADED_SAVE_DIR,
    )
    parser.add_argument(
        "--object_name",
        type=str,
        default="object",
        help="Object name (also the single body name). E.g. 'box'.",
    )
    parser.add_argument(
        "--sequence_name",
        type=str,
        default=None,
        help="Override the discovered sequence id (only when one clip is found).",
    )
    parser.add_argument(
        "--ground_z_offset",
        type=float,
        default=0.0,
        help="Constant z offset (m) added to the ground-aligned scene, lifting the "
        "object/hands (and the reconstructed support surface) above the world floor. "
        "Use e.g. 1.0 to place the object on a table-height support above the ground "
        "plane instead of resting on the floor (z=0).",
    )
    parser.add_argument(
        "--result_subpath",
        type=str,
        default="outputs/result",
        help="Path to the result/ bundle under a clip dir. Set to "
        "'outputs_geocalib/result' to load a GeoCalib gravity-aligned bundle.",
    )
    parser.add_argument(
        "--no_ground_align",
        action="store_true",
        default=False,
        help="Skip the OBB ground-alignment rotation and keep the bundle's existing "
        "world frame (use for GeoCalib-gravity-aligned bundles to avoid double "
        "rotation). Position is still normalized (center xy, bottom -> ground_z_offset).",
    )
    parser.add_argument(
        "--ground_yaw_offset_deg",
        type=float,
        default=0.0,
        help="Extra rotation (degrees) about world +z applied to the aligned scene "
        "(object + hands + support). Use to correct an arbitrary-yaw heading left by "
        "gravity alignment, e.g. -90 to undo a -90 deg offset.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run the ego_recon loader and save ManoSharpaData Parquet files."""
    loader = EgoReconDatasetLoader()
    loader.run(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
