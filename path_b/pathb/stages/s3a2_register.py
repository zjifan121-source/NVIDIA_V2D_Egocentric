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

Mesh candidates (V4 §13.1 A4), each registered onto the partial: `pooled` (run.py canonical) and `own`
(canonical.py on this episode's clips alone, in s3a2_register/<object>/own/); only meshes fused from our own
S1 evidence (the team's SAM 3D mesh was a third candidate until 2026-10-28: James, no team work in the
pipeline). The choice (`choose`): the more complete mesh first (ORDER: pooled, own) among those that pass
EXPLAIN_BAR or lie within TIE of the best share. Each candidate's B1 margin (V4 §13.2) is recorded but
does not rank pooled against own: like the share it is measured on the partial the own mesh is made
from. The choice and every candidate's numbers go into registration.json; the chosen mesh into s3a_mesh/.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

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
TIE = 0.05  # default (spec §13.1 A4 "ties"): explain-shares this close are tied
EXPLAIN_BAR = 0.8  # the stage's PASS bar on the share of the partial the mesh explains
ORDER = ("pooled", "own")  # the more complete mesh first
B1_FLAG_M = 0.0005  # §13.2 B1, calibrated 2026-10-26: a margin this small flags an ambiguous registration (4 of 6 flips)


def load_mesh(path) -> o3d.geometry.TriangleMesh:
    tm = trimesh.load(path, force="mesh", process=False)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tm.vertices), o3d.utility.Vector3iVector(tm.faces))
    mesh.compute_vertex_normals()
    return mesh


def write_mesh(mesh: o3d.geometry.TriangleMesh, d) -> None:
    d.mkdir(parents=True, exist_ok=True)
    tm = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False)
    tm.export(d / "mesh.obj")
    tm.export(d / "mesh.glb")


def candidates(ctx, name: str, partial, clip: str, out) -> dict:
    """Every A4 candidate's registration onto the partial (with "dir": where its mesh.obj/.glb are)."""
    from .. import canonical as C
    canon = ctx.root.parent / "canonical" / name
    info = json.loads((canon / "canonical.json").read_text())
    found = {"pooled": {"dir": canon, **register(load_mesh(canon / "mesh.obj"), partial, canonical_move(info, clip))}}
    try:
        mesh, rep = C.pool(name, ctx.root.parent, [ctx.number])
        C.write(mesh, rep, out / "own")
        found["own"] = {"dir": out / "own", "clips": len(rep["clips"]),
                        **register(load_mesh(out / "own/mesh.obj"), partial, canonical_move(rep, clip))}
    except SystemExit as e:
        found["own"] = {"skipped": str(e)}
    return found


def choose(found: dict) -> tuple[str, str]:
    """(candidate, why): the first of ORDER whose share passes EXPLAIN_BAR or lies within TIE of the best;
    so the pooled mesh stays unless it fails the bar and another explains clearly more. Neither the
    share nor the B1 margin can rank pooled against own: both are measured on the partial the own mesh
    is made from (ep45: own 0.91 / 10.7 mm vs pooled 0.86 / 8.2 mm, CD-O 0.78 vs 0.44 cm)."""
    shares = {c: r["partial_explained"] for c, r in found.items() if "partial_explained" in r}
    top = max(shares.values())
    good = [c for c in ORDER if c in shares and (shares[c] >= EXPLAIN_BAR or shares[c] >= top - TIE)]
    why = "passes the explain bar" if shares[good[0]] >= EXPLAIN_BAR else f"within {TIE} of the best share"
    return good[0], why + ("" if good[0] == "pooled" else f" (pooled {shares.get('pooled', 0):.2f})")


def anchors(names: list[str]) -> list[str]:
    """The objects the scorer aligns a scene by (s4b_reference.b3_objects: object 0, the first non-symmetric)."""
    from .s4b_reference import b3_objects
    return b3_objects(names)


def insure(found: dict, pick: str, why: str) -> tuple[str, str]:
    """§13.2 B2: an anchor whose chosen mesh registers ambiguously (B1 margin <= B1_FLAG_M) takes the candidate
    with the largest margin above the flag among those that pass EXPLAIN_BAR (ep21's cup: pooled 0.24 mm, own
    1.62 mm and explaining 0.95 of the partial vs 0.89). Otherwise the choice stands."""
    m = lambda c: (found[c].get("b1") or {}).get("margin_m")  # noqa: E731
    if m(pick) is None or m(pick) > B1_FLAG_M:
        return pick, why
    better = [c for c in ORDER if c != pick and m(c) is not None and m(c) > B1_FLAG_M
              and found[c].get("partial_explained", 0) >= EXPLAIN_BAR]
    if not better:
        return pick, why + f"; B2: margin {m(pick) * 1000:.2f} mm, no clearer candidate"
    c = max(better, key=m)
    return c, f"B2: {pick}'s margin {m(pick) * 1000:.2f} mm is ambiguous, {c}'s {m(c) * 1000:.2f} mm"


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
    """Open3D on one thread, as in pathb/invert.solve: its multithreaded ICP sums in a different order each run."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1, user_api="openmp"):
        return _run(ctx)


def _run(ctx) -> StageResult:
    from .. import canonical as C
    from ..ambiguity import margin
    d = ctx.dir("s3a2_register")
    metrics, outputs = [], {}
    for slot, name in enumerate(ctx.objects):
        canon = ctx.root.parent / "canonical" / name
        if not (canon / "mesh.obj").is_file():
            raise RuntimeError(f"no canonical mesh for {name}: run.py canonical --object {name} --out {ctx.root.parent}")
        fused = json.loads((ctx.root / f"s3a2_fuse_scan/{name}/fuse_report.json").read_text())
        key = "rest_before" if fused["partials"]["rest_before"]["fused"] else "rest_after"
        file = WINDOWS[key]
        partial = o3d.io.read_triangle_mesh(str(ctx.root / f"s3a2_fuse_scan/{name}/{file}"))
        partial.compute_vertex_normals()
        out = d / name
        out.mkdir(parents=True, exist_ok=True)
        found = candidates(ctx, name, partial, f"episode_{ctx.number:06d}/s3a2_fuse_scan/{name}/{file}", out)
        pv, pf = np.asarray(partial.vertices), np.asarray(partial.triangles)
        for c, x in found.items():  # B1 (V4 §13.2): each candidate onto the rest scan, the scorer's recipe
            if "dir" in x:
                m = trimesh.load(Path(x["dir"]) / "mesh.obj", force="mesh", process=False)
                x["b1"] = margin(m.vertices, m.faces, pv, pf, 1_000_003 * ctx.number + 101 * slot, name in C.REVOLUTION)
        pick, why = choose(found)
        if name in anchors(ctx.objects):
            pick, why = insure(found, pick, why)
        r = found[pick]
        files = mesh_files(name, r["dir"], ctx.root / f"s3a_mesh/{name}", out / "coacd.log")
        tm = trimesh.load(ctx.root / f"s3a_mesh/{name}/mesh.obj", force="mesh", process=False)
        window = fused["windows"][key]
        summary = {c: {k: v for k, v in x.items() if k not in ("T", "dir")} | ({"dir": str(x["dir"])} if "dir" in x else {})
                   for c, x in found.items()}
        report = {"object": name, "canonical": str(canon), "mesh": pick, "mesh_dir": str(r["dir"]), "choice": why,
                  "candidates": summary, "canonical_to_world": r["T"], "window": key,
                  "frame": window[0] if window else None, "fitness": r["fitness"], "rmse_m": r["rmse_m"],
                  "partial_explained": r["partial_explained"], "start": r["start"], "mesh_files": files}
        shares = ", ".join(f"{c} {x['partial_explained']:.2f}" for c, x in found.items() if "partial_explained" in x)
        metrics += [
            Metric(f"{name}: share of the partial the canonical mesh explains", r["partial_explained"], "",
                   ok=(0.8, 1), warn=(0.6, 1), note=f"{key} partial; mesh {pick} ({why}; {shares}); start: {r['start']}; "
                                                    f"mesh on the partial {r['fitness']:.2f}"),
            Metric(f"{name}: registration RMSE", (r["rmse_m"] or np.nan) * 1000, "mm", ok=(0, 3), warn=(0, 6)),
            Metric(f"{name}: B1 registration margin", None if r["b1"]["margin_m"] is None else r["b1"]["margin_m"] * 1000,
                   "mm", note=f"runner-up {r['b1']['runner_up_deg'] or 0:.0f} deg from the best of {r['b1']['starts']} "
                              "starts (the scorer's recipe onto the rest scan); bar: calibrated on public (V4 §13.2)"),
            Metric(f"{name}: canonical mesh watertight", int(files["watertight"]), "", ok=(1, 1)),
            Metric(f"{name}: convex pieces (CoACD {COACD_THRESHOLD})", files["pieces"], "", ok=(1, np.inf), note=files["coacd"]),
        ]
        if ctx.public:
            cdo = gt.cdo_cm(ctx.episode, name, tm.vertices, tm.faces, ctx.files_read)
            report["cdo_cm"] = cdo
            for c, x in found.items():  # what each candidate would have scored (the choice never reads it)
                if "dir" in x:
                    m = trimesh.load(Path(x["dir"]) / "mesh.obj", force="mesh", process=False)
                    report["candidates"][c]["cdo_cm"] = gt.cdo_cm(ctx.episode, name, m.vertices, m.faces, ctx.files_read)
            metrics.append(Metric(f"{name}: CD-O canonical mesh vs scan", cdo, "cm", kind="gt", ok=(0, 0.3), warn=(0, 0.5),
                                  note="candidates: " + ", ".join(f"{c} {x['cdo_cm']:.2f}" for c, x in report["candidates"].items()
                                                                  if "cdo_cm" in x)))
        write_json(out / "registration.json", report)
        outputs[name] = out / "registration.json"
    return StageResult(metrics, outputs)
