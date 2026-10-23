"""S3a2 register (V3_BUILD_SPEC.md §11 A2.3): per object, the canonical mesh (run.py canonical, in
<run>/canonical/<object>/) placed on this clip's rest-before partial scan (s3a2_fuse_scan); scale fixed.
Phase A2 variant: not in the default stage chain.

Start: the move canonical.json recorded for this clip (canonical frame -> its world), else FPFH + RANSAC
(3 seeds); then point-to-point ICP (outlines) and point-to-plane ICP, as canonical.py registers clips.
Out per object: registration.json {canonical_to_world0 (the canonical mesh's frame-0 pose), fitness,
rmse, start, canonical_to_phase_a (D: canonical coordinates in phase A's S3a mesh frame, from the two
frame-0 poses)}, and object_to_world.npz: phase A's S3c track re-expressed for the canonical mesh,
T(t) D (spec A2 without a GPU: "the FoundationPose track from phase A re-registered to the new mesh").
For a revolution object the canonical frame's turn about the axis is whatever the fit gave; D carries
it, so the re-expressed track and the mesh stay consistent (the kit's rule: meshes in the body frame
of their rollout poses).
Bars: share of the partial the placed mesh explains (within 5 mm, normals agreeing) >= 0.8 (PASS) /
0.6 (WARN); RMSE <= 3 mm (PASS) / 6 mm (WARN); the mesh's own share on the partial is reported only (it
cannot exceed the clip's coverage of the object). Public only: CD-O of
the canonical mesh against the scan, next to phase A's S3a mesh (A2.2's acceptance: lower).
"""
from __future__ import annotations

import json

import numpy as np
import open3d as o3d
import trimesh

from .. import gt
from .. import scan as S
from ..report import Metric, StageResult
from ..runner import write_json


def canonical_move(info: dict, episode: int) -> np.ndarray | None:
    """canonical -> this clip's world from canonical.json (None if the clip was not used)."""
    for clip, move in zip(info["clips"], info["moves"]):
        if clip.startswith(f"episode_{episode:06d}/") and move["kept"]:
            shift = np.eye(4)
            shift[:3, 3] = -np.asarray(info["world_to_canonical"])  # canonical -> reference world
            return np.linalg.inv(np.asarray(move["T"])) @ shift  # reference world -> this clip's world
    return None


def register(mesh: o3d.geometry.TriangleMesh, partial: o3d.geometry.TriangleMesh, start: np.ndarray | None) -> dict:
    source, target = S.mesh_points(mesh, 20000), S.mesh_points(partial, 20000)
    starts = [("canonical.json", start)] if start is not None else []
    starts += [(f"global seed {s}", S.global_start(source, target, seed=s)) for s in range(3 if start is None else 0)]
    best = None
    for label, T0 in starts:
        T = S.icp(source, target, T0, max_dist=0.02, iterations=300, point_to_point=True)["T"]
        r = S.icp(source, target, T, max_dist=0.005)
        # fitness of the partial on the canonical mesh: the share of what this clip saw that the mesh explains
        back = S.icp(target, source, np.linalg.inv(r["T"]), max_dist=0.005, iterations=0)
        r.update(start=label, partial_explained=back["fitness"])
        if best is None or r["partial_explained"] > best["partial_explained"]:
            best = r
    return best


def run(ctx) -> StageResult:
    d = ctx.dir("s3a2_register")
    metrics, outputs = [], {}
    for name in ctx.objects:
        canon = ctx.root.parent / "canonical" / name
        if not (canon / "mesh.obj").is_file():
            raise RuntimeError(f"no canonical mesh for {name}: run.py canonical --object {name} --out {ctx.root.parent}")
        info = json.loads((canon / "canonical.json").read_text())
        tm = trimesh.load(canon / "mesh.obj", force="mesh", process=False)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tm.vertices), o3d.utility.Vector3iVector(tm.faces))
        mesh.compute_vertex_normals()
        partial = o3d.io.read_triangle_mesh(str(ctx.root / f"s3a2_fuse_scan/{name}/partial_scan.ply"))
        partial.compute_vertex_normals()
        r = register(mesh, partial, canonical_move(info, ctx.number))
        T0 = r["T"]  # canonical -> world at frame 0
        anchors_file = ctx.root / f"s3c_fuse/{name}/anchors.json"
        track = np.load(ctx.root / f"s3c_fuse/{name}/object_to_world.npz")["object_to_world"].astype(np.float64)
        anchors = json.loads(anchors_file.read_text()) if anchors_file.is_file() else {}
        old0 = np.asarray(anchors["rest_before"]["pose"]) if anchors.get("rest_before") else track[0]
        D = np.linalg.inv(old0) @ T0  # canonical coordinates -> phase A mesh coordinates
        out = d / name
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "object_to_world.npz", object_to_world=track @ D)
        report = {"object": name, "canonical": str(canon), "canonical_to_world0": T0, "fitness": r["fitness"],
                  "rmse_m": r["rmse_m"], "partial_explained": r["partial_explained"], "start": r["start"],
                  "canonical_to_phase_a": D, "phase_a_frame0": "S3c rest-before anchor" if anchors.get("rest_before")
                  else "S3c track frame 0"}
        metrics += [
            Metric(f"{name}: share of the partial the canonical mesh explains", r["partial_explained"], "",
                   ok=(0.8, 1), warn=(0.6, 1), note=f"start: {r['start']}; mesh on the partial {r['fitness']:.2f}"),
            Metric(f"{name}: registration RMSE", (r["rmse_m"] or np.nan) * 1000, "mm", ok=(0, 3), warn=(0, 6)),
        ]
        if ctx.public:
            new = gt.cdo_cm(ctx.episode, name, tm.vertices, tm.faces, ctx.files_read)
            old = trimesh.load(ctx.root / f"s3a_mesh/{name}/mesh.obj", force="mesh", process=False)
            phase_a = gt.cdo_cm(ctx.episode, name, old.vertices, old.faces, ctx.files_read)
            report.update(cdo_cm=new, phase_a_cdo_cm=phase_a)
            metrics.append(Metric(f"{name}: CD-O canonical minus phase A's S3a mesh", new - phase_a, "cm", kind="gt",
                                  ok=(-np.inf, 0), note=f"canonical {new:.3f} cm, phase A {phase_a:.3f} cm (A2.2: lower)"))
        write_json(out / "registration.json", report)
        outputs[name] = out / "registration.json"
    return StageResult(metrics, outputs)
