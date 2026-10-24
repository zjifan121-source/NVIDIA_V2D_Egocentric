"""S3a register (V4_BUILD_SPEC.md S3a "Registration" and its outputs; v3 A2.3): per object, the canonical
mesh (run.py canonical: <run>/canonical/<object>/) placed on this clip's rest-before partial scan
(s3a2_fuse_scan; on the rest-after one when rest-before has no frame), scale fixed; and the episode's
mesh files under the names the later stages read.

Start: the move canonical.json recorded for this clip's partial, else FPFH + RANSAC (3 seeds); then
point-to-point ICP (outlines) and point-to-plane ICP, as canonical.py registers clips.
Out: s3a2_register/<object>/registration.json {canonical_to_world (the canonical mesh's pose in that
window: the frame-0 pose for rest-before), window, frame, fitness, rmse_m, partial_explained, start};
s3a_mesh/<object>/: mesh.obj and mesh.glb (the canonical mesh; every pose of the object is expressed
in its frame, V4 §3), coacd/ (CoACD threshold 0.05, drivers/coacd_pieces.py in $COACD_PY, default
FlashCHORD's venv), mass.json (closed volume x pathb/density.py), sha256.txt (of mesh.glb).
Bars: share of the partial the placed mesh explains (within 5 mm, normals agreeing) >= 0.8 (PASS) /
0.6 (WARN) (WARN/FAIL: S3b_track seeds the orientation); RMSE <= 3 mm / 6 mm; the canonical mesh
watertight; CoACD pieces made; public only: CD-O against the scan < 0.3 cm (PASS) / 0.5 cm (WARN).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess

import numpy as np
import open3d as o3d
import trimesh

from .. import gt
from .. import scan as S
from ..density import density
from ..paths import PATH_B
from ..report import Metric, StageResult
from ..runner import write_json
from .s3a2_fuse_scan import WINDOWS

COACD_THRESHOLD = 0.05  # spec S3a


def coacd_python() -> str:
    return os.environ.get("COACD_PY") or f"{os.environ.get('FLASH_CHORD_VENV', '/root/v2d/venvs/flash_chord')}/bin/python"


def canonical_move(info: dict, clip: str) -> np.ndarray | None:
    """canonical -> this partial's world from canonical.json (None if canonical.py did not keep it)."""
    for c, move in zip(info["clips"], info["moves"]):
        if c == clip and move["kept"]:
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


def mesh_files(name: str, canon, out, log) -> dict:
    """s3a_mesh/<object>/: the canonical mesh, its convex pieces, mass and hash."""
    out.mkdir(parents=True, exist_ok=True)
    for f in ("mesh.obj", "mesh.glb"):
        shutil.copy2(canon / f, out / f)
    mesh = trimesh.load(out / "mesh.obj", force="mesh", process=False)
    watertight = bool(mesh.is_watertight)
    volume = float(abs(mesh.volume)) if watertight else float(mesh.convex_hull.volume)
    write_json(out / "mass.json", {"volume_m3": volume, "volume_from": "mesh" if watertight else "convex hull",
                                   "density_kg_m3": density(name), "mass_kg": volume * density(name)})
    (out / "sha256.txt").write_text(hashlib.sha256((out / "mesh.glb").read_bytes()).hexdigest() + "  mesh.glb\n")
    shutil.rmtree(out / "coacd", ignore_errors=True)
    py = coacd_python()
    with open(log, "w") as f:
        rc = subprocess.run([py, str(PATH_B / "pathb/drivers/coacd_pieces.py"), str(out / "mesh.obj"), str(out / "coacd"),
                             str(COACD_THRESHOLD)], stdout=f, stderr=subprocess.STDOUT).returncode if os.path.exists(py) else None
    pieces = json.loads((out / "coacd/pieces.json").read_text())["pieces"] if rc == 0 else []
    return {"watertight": watertight, "volume_m3": volume, "pieces": len(pieces),
            "coacd": "ok" if rc == 0 else f"failed (exit {rc}, {log})" if rc is not None else f"no CoACD python at {py}"}


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
        fused = json.loads((ctx.root / f"s3a2_fuse_scan/{name}/fuse_report.json").read_text())
        key = "rest_before" if fused["partials"]["rest_before"]["fused"] else "rest_after"
        file = WINDOWS[key]
        partial = o3d.io.read_triangle_mesh(str(ctx.root / f"s3a2_fuse_scan/{name}/{file}"))
        partial.compute_vertex_normals()
        r = register(mesh, partial, canonical_move(info, f"episode_{ctx.number:06d}/s3a2_fuse_scan/{name}/{file}"))
        out = d / name
        out.mkdir(parents=True, exist_ok=True)
        files = mesh_files(name, canon, ctx.root / f"s3a_mesh/{name}", out / "coacd.log")
        window = fused["windows"][key]
        report = {"object": name, "canonical": str(canon), "canonical_to_world": r["T"], "window": key,
                  "frame": window[0] if window else None, "fitness": r["fitness"], "rmse_m": r["rmse_m"],
                  "partial_explained": r["partial_explained"], "start": r["start"], "mesh_files": files}
        metrics += [
            Metric(f"{name}: share of the partial the canonical mesh explains", r["partial_explained"], "",
                   ok=(0.8, 1), warn=(0.6, 1), note=f"{key} partial; start: {r['start']}; mesh on the partial {r['fitness']:.2f}"),
            Metric(f"{name}: registration RMSE", (r["rmse_m"] or np.nan) * 1000, "mm", ok=(0, 3), warn=(0, 6)),
            Metric(f"{name}: canonical mesh watertight", int(files["watertight"]), "", ok=(1, 1)),
            Metric(f"{name}: convex pieces (CoACD {COACD_THRESHOLD})", files["pieces"], "", ok=(1, np.inf), note=files["coacd"]),
        ]
        if ctx.public:
            cdo = gt.cdo_cm(ctx.episode, name, tm.vertices, tm.faces, ctx.files_read)
            report["cdo_cm"] = cdo
            metrics.append(Metric(f"{name}: CD-O canonical mesh vs scan", cdo, "cm", kind="gt", ok=(0, 0.3), warn=(0, 0.5)))
        write_json(out / "registration.json", report)
        outputs[name] = out / "registration.json"
    return StageResult(metrics, outputs)
