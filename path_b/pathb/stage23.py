"""Path B's stages 2-3 (James, 2026-10-29: no team code; replaces tools/track3/run_load_retarget.py).

Host side (Path B's venv): hand repair and the still scene on S4b's bundles, then drivers/nvidia_stage23.py
in $RETARGET_VENV on NVIDIA's loader, retarget, URDF and support scripts as imported (pathb/nvidia.py).
- `clean_hands`: P1d leaves every hand frame filled and rejects spikes over a window; a frame whose wrist
  still moves faster than P1d's own bar (p1d_hands.MAX_WRIST_SPEED) both from and to its neighbours is
  replaced by the pose interpolated between the nearest frames on either side that are not (position linear, rotation slerp, finger axis-angle linear; held
  at the clip's ends), and written back to the camera-frame MANO parameters the loader reads through the
  bundle's own root offset (J0: wrist_to_world = c2w (J0 + wrist_trans_in_camera), constant per hand).
- `still_scene`: the object at its frame-0 pose and both hands at their frame-0 pose for the whole clip.
- `run`: one scene (one or more objects sharing S4b's world, camera and hands) through the driver, then
  `check`: frames, robot frames (link origins) inside an object, support disks, the frame-0 seating gap.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from . import nvidia
from .paths import PATH_B, REPO, TRACK3_FPS

SIDES = ("right", "left")
GROUND_Z = 1.0  # m: table height of the task scene (the support surface sits under the objects there)
INSIDE_M = 0.02  # a robot frame this deep inside an object counts (S5's bar)


def mano_dir() -> Path:
    d = os.environ.get("MANO_DIR")
    return Path(d) if d else REPO / "baseline/reconstruction/data/weights/hand"


def wrist_world(data: dict, side: str) -> np.ndarray:
    return np.asarray(data[f"hand_{side}_wrist_to_world_transform"], np.float64)


def fast_frames(W: np.ndarray, fps: float, bar: float) -> np.ndarray:
    """(n,) spikes: frames whose wrist moves faster than bar (m/s) both from the previous frame and to the
    next (a single fast step is a fast motion, or P1d's to judge: it rejects spikes over a window)."""
    fast = np.linalg.norm(np.diff(W[:, :3, 3], axis=0), axis=1) * fps > bar
    bad = np.zeros(len(W), bool)
    bad[1:-1] = fast[:-1] & fast[1:]
    return bad


def fill(W: np.ndarray, fingers: np.ndarray, bad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bad frames from the nearest good ones (interpolated between, held at the ends)."""
    good = np.flatnonzero(~bad)
    if len(good) == 0 or not bad.any():
        return W, fingers
    t = np.arange(len(W))
    W2, F2 = W.copy(), fingers.copy()
    W2[:, :3, 3] = np.stack([np.interp(t, good, W[good, :3, 3][:, i]) for i in range(3)], axis=1)
    rots = Slerp(good, Rotation.from_matrix(W[good, :3, :3]))(np.clip(t, good[0], good[-1]))
    W2[bad, :3, :3] = rots.as_matrix()[bad]
    flat = fingers.reshape(len(fingers), -1)
    F2 = np.stack([np.interp(t, good, flat[good, i]) for i in range(flat.shape[1])], axis=1).reshape(fingers.shape)
    F2[~bad] = fingers[~bad]
    return W2, F2.astype(fingers.dtype)


def clean_hands(data: dict, fps: float = TRACK3_FPS) -> dict:
    """In place: each hand's fast frames filled (module doc); returns per side the frames replaced."""
    from .stages.p1d_hands import MAX_WRIST_SPEED
    c2w = np.asarray(data["camera_to_world_transform"], np.float64)
    report = {}
    for side in SIDES:
        W = wrist_world(data, side)
        cam_t = np.asarray(data[f"hand_{side}_wrist_trans_in_camera"], np.float64)
        J0 = np.median(np.einsum("nji,nj->ni", c2w[:, :3, :3], W[:, :3, 3] - c2w[:, :3, 3]) - cam_t, axis=0)
        bad = fast_frames(W, fps, MAX_WRIST_SPEED)
        report[side] = {"replaced": int(bad.sum()), "frames": np.flatnonzero(bad).tolist()[:40]}
        if not bad.any():
            continue
        W2, F2 = fill(W, np.asarray(data[f"hand_{side}_finger_pose"]), bad)
        R_cam = np.einsum("nji,njk->nik", c2w[:, :3, :3], W2[:, :3, :3])  # camera <- wrist
        dtype = data[f"hand_{side}_wrist_trans_in_camera"].dtype
        data[f"hand_{side}_wrist_to_world_transform"] = W2.astype(data[f"hand_{side}_wrist_to_world_transform"].dtype)
        data[f"hand_{side}_wrist_orient_in_camera"] = Rotation.from_matrix(R_cam).as_rotvec().astype(dtype)
        data[f"hand_{side}_wrist_trans_in_camera"] = (np.einsum("nji,nj->ni", c2w[:, :3, :3], W2[:, :3, 3] - c2w[:, :3, 3])
                                                      - J0).astype(dtype)
        if f"hand_{side}_wrist_to_camera_transform" in data:
            T = np.linalg.inv(c2w) @ W2
            data[f"hand_{side}_wrist_to_camera_transform"] = T.astype(data[f"hand_{side}_wrist_to_camera_transform"].dtype)
        data[f"hand_{side}_finger_pose"] = F2
    return report


def still_scene(data: dict) -> None:
    """In place: object and hands at frame 0 for every frame (the still candidate). The camera is held too:
    the loader poses the hands in the world through each frame's camera."""
    n = len(data["object_to_world_transform"])
    for key, value in list(data.items()):
        if key.startswith(("object_to_", "hand_", "camera_to_world")) and np.ndim(value) >= 1 and len(value) == n \
                and key not in ("object_is_valid",) and not key.endswith("_betas"):
            data[key] = np.repeat(np.asarray(value)[:1], n, axis=0)


def object_vertices(bundle: Path) -> np.ndarray:
    """The bundle's mesh vertices in the object frame, scaled by object_scale."""
    import trimesh
    with np.load(Path(bundle) / "result.npz") as z:
        scale = float(z["object_scale"])
    return np.asarray(trimesh.load(Path(bundle) / "mesh.obj", force="mesh", process=False).vertices) * scale


def retarget_venv() -> str:
    venv = os.environ.get("RETARGET_VENV")
    if not venv:
        raise RuntimeError("set RETARGET_VENV to the stages 2-3 environment (README §5.12)")
    return venv


def run(bundles: dict[str, Path], seq: str, out: Path, still: bool = False, device: str | None = None) -> dict:
    """One scene through drivers/nvidia_stage23.py: hands cleaned (and the still scene) in copies of the
    bundles under out/bundles/, then load, retarget, URDF, support, check. Returns its report."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    objects, cleaning = [], {}
    for name, src in bundles.items():
        dst = out / "bundles" / name
        shutil.copytree(src, dst, symlinks=False)
        data = dict(np.load(dst / "result.npz"))
        cleaning[name] = clean_hands(data)
        if still:
            still_scene(data)
        np.savez(dst / "result.npz", **data)
        objects.append({"name": name, "bundle": str(dst)})
    request = {"loader": str(nvidia.pristine(nvidia.LOADER)), "robotic": str(nvidia.pristine(nvidia.ROBOTIC)),
               "mano_dir": str(mano_dir()), "device": device or os.environ.get("RETARGET_DEVICE") or "cuda:0",
               "fps": TRACK3_FPS, "ground_z_offset": GROUND_Z, "sequence_id": seq, "objects": objects, "out": str(out)}
    (out / "request.json").write_text(json.dumps(request, indent=1) + "\n")
    log = out / "driver.log"
    with open(log, "w") as f:
        rc = subprocess.run([f"{retarget_venv()}/bin/python", str(PATH_B / "pathb/drivers/nvidia_stage23.py"),
                             str(out / "request.json")], cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"stage 2-3 failed (exit {rc}): " + "\n".join(log.read_text().splitlines()[-15:]))
    report = json.loads((out / "report.json").read_text())
    with np.load(Path(objects[0]["bundle"]) / "result.npz") as z:
        frames = len(z["object_to_world_transform"])
    report["output_check"] = check(Path(report["processed"]), [Path(m) for m in report["meshes"]],
                                   Path(report["support_usd"]), frames)
    report["bundle_check"] = {"warnings": []}
    report["hand_cleaning"] = cleaning
    report["still"] = still
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def supports(usda: Path) -> list[dict]:
    """The support cylinders of NVIDIA's support USD (text): centre, radius, top z."""
    import re
    out = []
    for name, body in re.findall(r'def Cylinder "([^"]+)"[^{]*\{(.*?)\n    \}', Path(usda).read_text(), re.S):
        h = float(re.search(r"double height = ([-\d.e]+)", body).group(1))
        r = float(re.search(r"double radius = ([-\d.e]+)", body).group(1))
        c = [float(x) for x in re.search(r"double3 xformOp:translate = \(([^)]*)\)", body).group(1).split(",")]
        out.append({"name": name, "centre": c, "radius": r, "top_z": c[2] + h / 2})
    return out


def check(part: Path, meshes: list[Path], usda: Path, frames: int) -> dict:
    """Frames; robot frames (link origins) more than INSIDE_M inside an object; support disks; per object the
    frame-0 gap between its lowest point and the top of the support disk under it (cm)."""
    import open3d as o3d
    import pyarrow.parquet as pq
    import trimesh
    table = pq.read_table(next(Path(part).glob("*.parquet")))
    row = {c: table.column(c)[0].as_py() for c in table.column_names}
    pos, wxyz = np.asarray(row["object_body_position"]), np.asarray(row["object_body_wxyz"])
    n, warnings = len(pos), []
    if n != frames:
        warnings.append(f"{n} frames, the clip has {frames}")
    robot = np.concatenate([np.asarray(row[f"robot_{s}_frames"])[:, :, :3] for s in SIDES if f"robot_{s}_frames" in row],
                           axis=1)
    deepest, disks, seating = np.zeros(n), supports(usda), {}
    for k, mesh_path in enumerate(meshes):
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                            o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
        R = Rotation.from_quat(wxyz[:, k][:, [1, 2, 3, 0]]).as_matrix()
        local = np.einsum("nji,npj->npi", R, robot - pos[:, k][:, None])  # robot frames in the object frame
        sd = scene.compute_signed_distance(o3d.core.Tensor(local.reshape(-1, 3).astype(np.float32))).numpy()
        deepest = np.maximum(deepest, -sd.reshape(n, -1).min(axis=1))  # open3d: negative inside
        world0 = np.asarray(mesh.vertices) @ R[0].T + pos[0, k]
        xy = world0[:, :2].mean(axis=0)
        under = [d for d in disks if np.hypot(*(xy - np.asarray(d["centre"][:2]))) <= d["radius"] + 0.02
                 and d["top_z"] <= world0[:, 2].min() + 0.05]
        top = max((d["top_z"] for d in under), default=None)
        seating[Path(mesh_path).stem] = None if top is None else round(float(world0[:, 2].min() - top) * 100, 2)
    return {"frames": n, "penetration_frames": {"over_2cm": int((deepest > INSIDE_M).sum()),
                                                "max_cm": round(float(deepest.max()) * 100, 2)},
            "support_disks": len(disks), "frame0_seating_cm": seating, "warnings": warnings}


def worst_seating(value) -> float | None:
    """The frame-0 seating gap farthest from zero over a scene's objects (a dict, a list or one number)."""
    values = list(value.values()) if isinstance(value, dict) else value if isinstance(value, list) else [value]
    values = [v for v in values if v is not None]
    return max(values, key=abs) if values else None
