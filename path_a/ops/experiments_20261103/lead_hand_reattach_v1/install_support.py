#!/usr/bin/env python3
"""Install an episode's explicit support surfaces beside a matching retargeted reference.

CPU only. This does not modify a mesh, pose, hand, or Parquet. A changed reference
or a non-Z-up consumer transform fails rather than silently moving the table.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def common_transform(raw, record):
    keys = ['object_to_world_transform'] + sorted((k for k in raw.files if k.startswith('object_') and k.endswith('_to_world_transform') and k != 'object_to_world_transform'), key=lambda k: int(k.split('_')[1]))
    source = np.stack([raw[k] for k in keys], axis=1)
    target = np.tile(np.eye(4), (*source.shape[:2], 1, 1))
    p = np.asarray(record["object_body_position"], dtype=float)
    q = np.asarray(record["object_body_wxyz"], dtype=float)
    if p.shape != source.shape[:2] + (3,) or q.shape != source.shape[:2] + (4,):
        raise ValueError("Expected all matching object frames")
    target[..., :3, 3] = p
    target[..., :3, :3] = Rotation.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(*source.shape[:2], 3, 3)
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite object transforms")
    # Fit only a common coordinate change; never fit separate objects or frames.
    transform = target[0, 0] @ np.linalg.inv(source[0, 0])
    predicted = transform @ source
    pos_error = float(np.max(np.linalg.norm(predicted[..., :3, 3] - target[..., :3, 3], axis=-1)))
    rot_error = float(np.max(np.abs(predicted[..., :3, :3] - target[..., :3, :3])))
    if pos_error > 1e-5 or rot_error > 1e-5:
        raise ValueError(f"Reference is not one common rigid transform of the pinned bundle: {pos_error=}, {rot_error=}")
    if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5) or np.linalg.det(transform[:3, :3]) < 0.99999:
        raise ValueError("Reference coordinate transform contains scale or reflection")
    # The current FlashCHORD support reader consumes Z cylinders, not rotations.
    if not np.allclose(transform[:3, :3] @ [0, 0, 1], [0, 0, 1], atol=1e-5):
        raise ValueError("Tilted ground alignment is unsupported; load with --no_ground_align")
    return transform, pos_error, rot_error


def default_output(parquet):
    seq = next((p for p in parquet.parents if p.name.startswith("sequence_id=")), None)
    robot = next((p.name.split("=", 1)[1] for p in parquet.parents if p.name.startswith("robot_name=")), None)
    if seq is None or robot is None:
        raise ValueError("Use native sequence_id/robot_name directories or pass --output")
    return seq.parent.parent / "reconstructed_stage" / f"{seq.name.split('=', 1)[1]}_{robot}_support.usda"


def write_stage(path, cylinders, transform):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite support: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    for i, cylinder in enumerate(cylinders):
        center = transform @ np.r_[cylinder["center_bundle_world_m"], 1.0]
        geom = UsdGeom.Cylinder.Define(stage, f"/support_{i}")
        geom.CreateAxisAttr("Z")
        geom.CreateRadiusAttr(cylinder["radius_m"])
        geom.CreateHeightAttr(cylinder["height_m"])
        UsdGeom.Xformable(geom).AddTranslateOp().Set(Gf.Vec3d(*center[:3]))
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    stage.GetRootLayer().Save()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--parquet", type=Path, required=True, help="Full native retargeted Parquet")
    ap.add_argument("--output", type=Path, help="Default: sibling reconstructed_stage path resolved by FlashCHORD")
    ap.add_argument("--spec", type=Path, default=Path(__file__).with_name("support_in_bundle_world.json"))
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text())
    for name, expected in spec["bundle_sha256"].items():
        if sha(args.bundle / name) != expected:
            raise ValueError(f"Bundle mismatch: {name}")
    table = pq.read_table(args.parquet)
    if table.num_rows != 1:
        raise ValueError("Expected one complete sequence row")
    record = {k: v[0] for k, v in table.to_pydict().items()}
    with np.load(args.bundle / "result.npz", allow_pickle=False) as raw:
        transform, pos_error, rot_error = common_transform(raw, record)
    output = (args.output or default_output(args.parquet.resolve())).resolve()
    write_stage(output, spec["cylinders"], transform)
    print(json.dumps({"state": "SUPPORT_INSTALLED_WITH_GEOMETRY_WARNING", "frames": int(len(record["object_body_position"])),
        "objects": int(len(record["object_body_names"])), "collision_shapes": len(spec["cylinders"]),
        "world_to_consumer": transform.tolist(), "max_pose_position_difference_m": pos_error,
        "max_pose_rotation_matrix_difference": rot_error, "support_sha256": sha(output),
        "output": str(output), "source_bundle_modified": False, "reference_modified": False,
        "warning": spec["warning"], "physical_rollout_validated": False}, indent=2))


if __name__ == "__main__":
    main()
