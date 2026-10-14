"""Rebuild a reconstruction bundle from NVIDIA's committed tissue_box_simple example.

The repo ships no result.npz, only the finished retarget of one clip (ego_recon
processed/sequence_id=tissue_box_simple). This inverts the loader on it: world-frame MANO
parameters become camera-frame ones under an identity camera, the left hand goes back to the
right-hand parameter space HaMeR/WiLoR use, and the object poses and mesh are copied. Loading
and retargeting the result should reproduce NVIDIA's output, so run_load_retarget.py can be
tested (and compared against a known answer) without running reconstruction.

    $RETARGET_VENV/bin/python tools/track3/make_sample_bundle.py --out <dir>
    $RETARGET_VENV/bin/python tools/track3/run_load_retarget.py --bundle <dir> \
        --sequence-id tissue_box_rt --fps 30 --out <run dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
SAMPLE = (REPO / "baseline/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/"
          "human_motion_data/ego_recon/processed")
SEQ = "tissue_box_simple"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    parquet = next((SAMPLE / f"sequence_id={SEQ}/robot_name=sharpa_wave").glob("*.parquet"))
    table = pq.read_table(parquet)
    d = {name: table.column(name)[0].as_py() for name in table.column_names}
    n = len(d["mano_right_trans"])

    object_T = np.tile(np.eye(4), (n, 1, 1))
    object_T[:, :3, :3] = Rotation.from_quat(
        np.asarray(d["object_body_wxyz"])[:, 0][:, [1, 2, 3, 0]]).as_matrix()
    object_T[:, :3, 3] = np.asarray(d["object_body_position"])[:, 0]
    arrays = {
        "camera_to_world_transform": np.tile(np.eye(4), (n, 1, 1)),
        "camera_is_valid": np.ones(n, bool),
        "object_to_world_transform": object_T,
        "object_scale": np.float32(1.0),
        "object_is_valid": np.ones(n, bool),
        "gravity_alignment_applied": np.array(True),
    }
    for side in ("right", "left"):
        orient = np.asarray(d[f"mano_{side}_global_orient"], dtype=np.float64)
        fingers = np.asarray(d[f"mano_{side}_finger_pose"], dtype=np.float64).reshape(n, 15, 3)
        if side == "left":  # undo the loader's native-left mirror (it is its own inverse)
            orient = orient.copy()
            orient[:, 1:] *= -1
            fingers = fingers.copy()
            fingers[:, :, 1:] *= -1
        arrays |= {
            f"hand_{side}_betas": np.asarray(d[f"mano_{side}_betas"], dtype=np.float32),
            f"hand_{side}_wrist_orient_in_camera": orient,
            # Identity camera: the loader's re-pose reduces to trans_world = trans_camera.
            f"hand_{side}_wrist_trans_in_camera": np.asarray(d[f"mano_{side}_trans"], dtype=np.float64),
            f"hand_{side}_finger_pose": fingers.reshape(n, 45),
            f"hand_{side}_scale": np.ones(n, np.float32),
            f"hand_{side}_is_valid": np.ones(n, bool),
        }
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "result.npz", **arrays)
    obj = (SAMPLE / f"{SEQ}.obj").read_text().replace(f"mtllib {SEQ}.mtl", "mtllib material.mtl")
    (args.out / "mesh.obj").write_text(obj)
    shutil.copy(SAMPLE / f"{SEQ}.mtl", args.out / "material.mtl")
    for texture in SAMPLE.glob(f"{SEQ}_material_*.png"):
        shutil.copy(texture, args.out / texture.name)
    (args.out / "manifest.json").write_text(json.dumps(
        {"schema_version": 1, "n_frames": n, "result_npz": "result.npz", "source": f"rebuilt from {parquet}"},
        indent=1) + "\n")
    print(f"wrote {n}-frame bundle to {args.out} (source fps {d['fps']})")


if __name__ == "__main__":
    main()
