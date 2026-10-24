"""S3b2 track points (V3_BUILD_SPEC.md §11 A2.4): a mesh-free object track from 3D point tracks.
Phase A2 variant: not in the default stage chain; the GPU half runs on the VM.

TAPIP3D (Apache-2.0, github.com/zbw001/TAPIP3D, in $TAPIP3D_PY / $TAPIP3D_DIR) tracks points on the
object in 3D with our depth and camera: its inference.py takes one .npz (video uint8 (T, H, W, 3), depths
(T, H, W) metres, intrinsics (T, 3, 3), extrinsics (T, 4, 4) world-to-camera, query_point (N, 4) as
[t, x, y, z] in the world) and writes coords (T, N, 3) and visibs (T, N). We give it ego_cam_a at half
resolution with P2's depth_a_half, P4's camera (v3 world), and up to 512 query points on the object: its
P1b mask (eroded) at the first rest-before frame A2.1 fused (else frame 0), unprojected with the depth.
Then, on the CPU: per frame a RANSAC-Kabsch rigid fit of the visible tracks to their query positions
(inliers within 1 cm) gives the object's world motion since the query frame; a fit that turns more than
60 degrees from the previous frame's takes its symmetric equivalent closest to it (the S3b continuity
rule; pathb/symmetry.py), else the frame is left out. The object's pose at the query frame comes from A2.3
(canonical mesh) when it has run, else from S3c's track; poses = camera_to_world^-1 x motion x that pose.
Out per object: poses/%06d.json (object_to_camera, the S3b layout), track_meta.json (per frame: inlier
share as "iou", visible tracks, fit RMSE, "source": tapip3d), tracks.npz (TAPIP3D's output).
Bars: frames with a fit >= 90 % (PASS) / 70 % (WARN); median fit RMSE <= 5 mm (PASS) / 1 cm (WARN).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .. import baseline as B
from .. import symmetry
from ..geometry import read_depth, read_transforms, write_transform
from ..report import Metric, StageResult
from ..runner import write_json

MAX_QUERIES = 512
INLIER_M = 0.01
RANSAC_ROUNDS = 200
MAX_TURN_DEG = 60.0


def kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """4x4 rigid T with Q ~ T P (points as rows)."""
    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    R = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cq - R @ cp
    return T


def ransac_rigid(P: np.ndarray, Q: np.ndarray, seed: int = 0) -> tuple[np.ndarray | None, np.ndarray]:
    """Rigid fit of P onto Q robust to bad tracks: (T, inlier mask)."""
    if len(P) < 3:
        return None, np.zeros(len(P), bool)
    rng = np.random.default_rng(seed)
    best = np.zeros(len(P), bool)
    for _ in range(RANSAC_ROUNDS):
        pick = rng.choice(len(P), 3, replace=False)
        T = kabsch(P[pick], Q[pick])
        inl = np.linalg.norm(P @ T[:3, :3].T + T[:3, 3] - Q, axis=1) <= INLIER_M
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < 3:
        return None, best
    return kabsch(P[best], Q[best]), best


def motions(coords: np.ndarray, visible: np.ndarray, q: int, G: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """World motion since frame q per frame ((T, 4, 4), NaN where no fit) and the per-frame record."""
    n = len(coords)
    M = np.full((n, 4, 4), np.nan)
    rows = []
    P = coords[q]
    previous = np.eye(4)
    order = list(range(q, n)) + list(range(q - 1, -1, -1))  # forward from q, then backward
    for t in order:
        if t == q - 1:
            previous = np.eye(4)
        vis = visible[t] & visible[q]
        T, inl = ransac_rigid(P[vis], coords[t][vis], seed=t)
        row = {"frame": int(t), "visible": int(vis.sum()), "iou": float(inl.mean()) if vis.any() else 0.0,
               "recovered": False, "source": "tapip3d"}
        if T is not None:
            if symmetry.angle(previous[:3, :3], T[:3, :3]) > np.radians(MAX_TURN_DEG):
                Tc, k = symmetry.closest(previous, T, G)  # a symmetric equivalent of the fit
                if symmetry.angle(previous[:3, :3], Tc[:3, :3]) > np.radians(MAX_TURN_DEG):
                    T = None
                    row["rejected"] = "turn > 60 deg, also for every symmetric equivalent"
                else:
                    T, row["symmetric_switch"] = Tc, int(k)
        if T is not None:
            res = np.linalg.norm(P[vis][inl] @ T[:3, :3].T + T[:3, 3] - coords[t][vis][inl], axis=1)
            row["rmse_m"] = float(np.sqrt(np.mean(res ** 2)))
            M[t], previous = T, T
        rows.append(row)
    return M, sorted(rows, key=lambda r: r["frame"])


def queries(ctx, k: int, q: int, K: np.ndarray, shape, c2w: np.ndarray) -> np.ndarray:
    """(N, 4) [t, x, y, z] world points on the object at frame q (eroded mask, stereo depth)."""
    m = B.read_mask(B.mask_tracks(ctx.root)[f"obj_{k}"], q)
    if m is None:
        return np.zeros((0, 4))
    mask = cv2.erode(cv2.resize(m.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST), np.ones((5, 5))) > 0
    depth = read_depth(ctx.root / f"p2_depth/depth_a_half/{q:06d}.png")
    v, u = np.nonzero(mask & (depth > 0))
    if len(u) == 0:
        return np.zeros((0, 4))
    pick = np.random.default_rng(0).choice(len(u), min(MAX_QUERIES, len(u)), replace=False)
    u, v = u[pick], v[pick]
    z = depth[v, u]
    cam = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=1)
    world = cam @ c2w[q, :3, :3].T + c2w[q, :3, 3]
    return np.c_[np.full(len(world), q), world]


def tapip3d(inputs: Path, out_dir: Path, log: Path) -> Path:
    py, repo = os.environ.get("TAPIP3D_PY", "/root/v2d/venvs/tapip3d/bin/python"), \
        os.environ.get("TAPIP3D_DIR", "/root/v2d/third_party/TAPIP3D")
    if not (Path(py).exists() and Path(repo).is_dir()):
        raise RuntimeError(f"TAPIP3D not installed ({py}, {repo}): A2.4 runs on the GPU machine")
    with open(log, "w") as f:
        rc = subprocess.run([py, "inference.py", "--input_path", str(inputs), "--checkpoint",
                             str(Path(repo) / "checkpoints/tapip3d_final.pth"), "--output_dir", str(out_dir)],
                            cwd=repo, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"TAPIP3D failed (exit {rc}): " + "\n".join(log.read_text().splitlines()[-15:]))
    return max(out_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime)


def run(ctx) -> StageResult:
    d = ctx.dir("s3b2_track_points")
    n = int(ctx.episode["num_frames"])
    intr = json.loads((ctx.root / "p2_depth/depth_a_half/intrinsics.json").read_text())
    K, shape = np.asarray(intr["K"], dtype=np.float64), (intr["size"][1], intr["size"][0])
    c2w, ok = read_transforms(ctx.root / "p4_export/camera_to_world", n)
    metrics, outputs = [], {}
    for k, name in enumerate(ctx.objects):
        out = d / name
        (out / "poses").mkdir(parents=True, exist_ok=True)
        fused = ctx.root / f"s3a2_fuse_scan/{name}/fuse_report.json"
        q = min(json.loads(fused.read_text())["fused"], default=0) if fused.is_file() else 0
        Q = queries(ctx, k, q, K, shape, c2w)
        if not (out / "tracks.npz").is_file():
            frames = sorted((ctx.root / "p1_calibrate/cam_a").glob("*.jpg"))[:n]
            video = np.stack([cv2.cvtColor(cv2.resize(cv2.imread(str(f)), shape[::-1], interpolation=cv2.INTER_AREA),
                                           cv2.COLOR_BGR2RGB) for f in frames])
            depths = np.stack([read_depth(ctx.root / f"p2_depth/depth_a_half/{t:06d}.png") for t in range(n)]).astype(np.float32)
            np.savez(out / "tapip3d_input.npz", video=video, depths=depths, intrinsics=np.repeat(K[None], n, 0).astype(np.float32),
                     extrinsics=np.linalg.inv(c2w).astype(np.float32), query_point=Q.astype(np.float32))
            result = tapip3d(out / "tapip3d_input.npz", out / "tapip3d_out", out / "tapip3d.log")
            r = np.load(result)
            np.savez(out / "tracks.npz", coords=r["coords"], visibs=r["visibs"], query_points=Q, query_frame=q)
            (out / "tapip3d_input.npz").unlink()
        tr = np.load(out / "tracks.npz")
        vertices = None
        reg = ctx.root / f"s3a2_register/{name}/registration.json"
        if reg.is_file():
            pose_q = np.asarray(json.loads(reg.read_text())["canonical_to_world0"])  # canonical mesh, rest-before
            source = "A2.3 canonical mesh"
            import trimesh
            vertices = np.asarray(trimesh.load(ctx.root.parent / f"canonical/{name}/mesh.obj", force="mesh",
                                               process=False).vertices)
        else:
            pose_q = np.load(ctx.root / f"s3c_fuse/{name}/object_to_world.npz")["object_to_world"][q]
            source = "S3c track"
        G = symmetry.group(name, vertices)
        M, rows = motions(np.asarray(tr["coords"], dtype=np.float64), np.asarray(tr["visibs"], bool), q, G)
        fitted = np.isfinite(M[:, 0, 0]) & ok
        for t in np.flatnonzero(fitted):
            write_transform(out / f"poses/{t:06d}.json", np.linalg.inv(c2w[t]) @ M[t] @ pose_q)
        write_json(out / "track_meta.json", {"source": "tapip3d", "query_frame": q, "object_frame_from": source,
                                             "frames": rows})
        rmse = [r["rmse_m"] for r in rows if "rmse_m" in r]
        outputs[name] = out / "poses"
        metrics += [
            Metric(f"{name}: frames with a rigid fit", float(fitted.mean()), "share", ok=(0.9, 1), warn=(0.7, 1),
                   note=f"{len(Q)} query points at frame {q}; object frame from {source}"),
            Metric(f"{name}: fit RMSE (median)", float(np.median(rmse)) * 1000 if rmse else None, "mm", ok=(0, 5), warn=(0, 10)),
            Metric(f"{name}: symmetric switches", sum("symmetric_switch" in r for r in rows), ""),
        ]
    return StageResult(metrics, outputs)
