"""Trackability: how much of a reference a policy that follows the reference hand can track, measured
GT-free on the S5 task it trains on (task "trackability score", 2026-11-02).

F2's policy tax (inverted playback - rollout AUC) is small where the reference hand carries the object
and large where the object moves with no hand on it (slip-guard / failed-grasp bridges, no-grasp moves)
or where a rest does not stand (ep18). Features per episode x candidate, from S5's ManoSharpa parquet
(the policy's own reference: robot frames, object poses, contact slots), its meshes and support USD:
- UMS, unsupported-motion share: moving frames (posed-mesh centroid > MOVING_M per frame, or inside a
  grasp / bridge of S3c's invert report) on which no hand has >= HELD_TIPS fingertip sites within delta
  of the posed mesh (open3d signed distance; delta = max(DELTA_MIN_M, DELTA_EXTENT x the mesh's smallest
  oriented extent)). ums_run: the longest such run (frames); d_hand_cm: mean nearest fingertip-surface
  distance over moving frames; ums_opp: held needs two of those tips on surfaces facing each other
  (normals' dot < OPPOSED_DOT).
- RCC, reference contact coverage: share of moving frames with >= 1 / >= 2 of S5's 16 MANO-link contact
  slots filled for the object (NVIDIA contact_utils: object point within 1 cm of a MANO vertex; part id =
  the object's slot + 1).
- RSS, rest settle in S5's scene: CoACD pieces of each object's mesh (FlashCHORD decomposes the URDF's
  collision mesh, threshold 0.05), S5's support disks, the ground plane at z = 0 (collision.ground),
  hands removed; each object dropped RSS_SECONDS from frame 0 and from the first frame of every static
  segment, the other objects fixed where the reference has them (drivers/settle.py in $SETTLE_PY, S3d's
  contact settings). A rest is unsettled past UNSETTLED_M or UNSETTLED_DEG.
- support_ok: every object has a disk or another object under its frame-0 bottom.
The still candidate has no moving frames (ums 0); only RSS and support_ok apply to it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation

PATH_B = Path(__file__).resolve().parents[1]
MOVING_M = 0.01  # posed-mesh centroid travel per frame that makes a frame moving (as step 1)
DELTA_MIN_M = 0.015  # fingertip-surface distance that counts as touching, at least ...
DELTA_EXTENT = 0.5  # ... or half the object's smallest extent (a thin sheet keeps DELTA_MIN_M)
HELD_TIPS = 2  # fingertip sites of one hand within delta for a held frame
OPPOSED_DOT = -0.3  # ums_opp: two of those tips on surfaces facing each other
RSS_SECONDS, RSS_TIMESTEP = 0.5, 0.002  # S3d's drop
UNSETTLED_M, UNSETTLED_DEG = 0.01, 10.0  # a rest that moves more than this does not stand
SUPPORT_GAP_M = 0.02  # support_ok: a disk top or another object within this of the frame-0 bottom
COACD_THRESHOLD = 0.05  # FlashCHORD's convex_decompose default
GROUND_Z = 0.0  # FlashCHORD's ground plane (collision.ground)
SIDES = ("right", "left")
TIPS = ("thumb", "index", "middle", "ring", "pinky")


def settle_python() -> str:
    return os.environ.get("SETTLE_PY") or f"{os.environ.get('DAID_VENV', '/root/v2d/venvs/daid')}/bin/python"


def coacd_python() -> str:
    return os.environ.get("COACD_PY") or settle_python()


def cache_dir() -> Path:
    return Path(os.environ.get("PATHB_CACHE", PATH_B / "data/cache"))


def matrices(pos: np.ndarray, wxyz: np.ndarray) -> np.ndarray:
    T = np.tile(np.eye(4), (len(pos), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(np.asarray(wxyz)[:, [1, 2, 3, 0]]).as_matrix()
    T[:, :3, 3] = pos
    return T


def longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def task(cand_dir: Path) -> dict:
    """S5's candidate folder -> the sequence: parquet cells, the processed folder, the support USD. The paths
    S5 wrote are absolute on the machine that ran it, so files are found by name under cand_dir."""
    info = json.loads((cand_dir / "task.json").read_text())
    seq = info["sequence_id"]
    processed = cand_dir / "run/assets/human_motion_data/ego_recon/processed"
    partition = next(processed.glob(f"sequence_id={seq}/robot_name=*"))
    table = pq.read_table(next(partition.glob("*.parquet")))
    cells = {c: table.column(c)[0].as_py() for c in table.column_names}
    usd = processed.parent / "reconstructed_stage" / Path(info["support_usd"]).name
    return {"info": info, "seq": seq, "cells": cells, "processed": processed, "support_usd": usd}


def object_name(body: str, names: list[str]) -> str:
    """S5's body name (ep12_inverted_blue_cup; one object: the sequence id, ep40_inverted_mini_dust_pan) -> the
    roster name it ends with (the longest, so white_pot_lid is not white_pot)."""
    hits = [n for n in names if body == n or body.endswith("_" + n)]
    return max(hits, key=len) if hits else body


def objects(t: dict) -> list[dict]:
    """Per object (S5's body order): name, mesh (the watertight OBJ in the body frame), (n, 4, 4) poses."""
    cells = t["cells"]
    names = list(t["info"].get("bundles", {}))  # S5's objects by roster name
    P, Q = np.asarray(cells["object_body_position"], float), np.asarray(cells["object_body_wxyz"], float)
    out = []
    for k, body in enumerate(cells["object_body_names"]):
        mesh = t["processed"] / Path(cells["object_mesh_paths"][k]).name
        out.append({"name": object_name(body, names), "body": body, "mesh": mesh, "poses": matrices(P[:, k], Q[:, k])})
    return out


def fingertips(cells: dict) -> np.ndarray:
    """(n, 2, 5, 3) fingertip sites, right then left, from robot_{side}_frames (position, wxyz)."""
    out = []
    for side in SIDES:
        names = cells[f"{side}_robot_frame_names"]
        F = np.asarray(cells[f"robot_{side}_frames"], float)
        out.append(F[:, [names.index(f"{side}_{tip}_tip_site") for tip in TIPS], :3])
    return np.stack(out, axis=1)


def load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh", process=False)
    return m.dump(concatenate=True) if isinstance(m, trimesh.Scene) else m


def delta_m(mesh: trimesh.Trimesh) -> float:
    extents = trimesh.bounds.oriented_bounds(mesh)[1]
    return max(DELTA_MIN_M, DELTA_EXTENT * float(np.min(extents)))


def surface_distance(mesh: trimesh.Trimesh, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance (m, negative inside) of body-frame points to the mesh and the face normal at the
    closest point (open3d's raycasting scene)."""
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                        o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
    q = o3d.core.Tensor(np.asarray(points, np.float32).reshape(-1, 3))
    sd = scene.compute_signed_distance(q).numpy()
    normals = scene.compute_closest_points(q)["primitive_normals"].numpy()
    return sd.reshape(points.shape[:-1]), normals.reshape(points.shape)


def labels(report: dict | None, n: int) -> dict[str, np.ndarray]:
    """Per frame from S3c's invert report: in a grasp, a failed grasp, unstable, inside a bridge; static
    segments as (start, end)."""
    lab = {k: np.zeros(n, bool) for k in ("grasp", "failed", "unstable", "bridged")}
    static = []
    for s in (report or {}).get("segments", []):
        a, b = int(s["start"]), min(int(s["end"]) + 1, n)
        if s["kind"] == "grasp":
            lab["grasp"][a:b] = True
            lab["failed"][a:b] |= bool(s.get("failed"))
        elif s["kind"] == "unstable":
            lab["unstable"][a:b] = True
        elif s["kind"] == "static":
            static.append((a, b - 1))
        if s.get("bridge"):
            lab["bridged"][int(s["bridge"][0]):min(int(s["bridge"][1]) + 1, n)] = True
    lab["static"] = static
    return lab


def moving_frames(mesh: trimesh.Trimesh, poses: np.ndarray, lab: dict) -> np.ndarray:
    c = np.asarray(mesh.vertices).mean(axis=0)
    cw = np.einsum("tij,j->ti", poses[:, :3, :3], c) + poses[:, :3, 3]
    speed = np.r_[0.0, np.linalg.norm(np.diff(cw, axis=0), axis=1)]
    return (speed > MOVING_M) | lab["grasp"] | lab["bridged"]


def hand_support(mesh: trimesh.Trimesh, poses: np.ndarray, tips: np.ndarray, delta: float) -> dict:
    """Per frame: held (>= HELD_TIPS tips of one hand within delta), held with opposed normals, nearest tip
    distance (m, 0 inside)."""
    n = len(poses)
    inv = np.linalg.inv(poses)
    local = np.einsum("tij,thkj->thki", inv[:, :3, :3], tips) + inv[:, None, None, :3, 3]
    sd, normals = surface_distance(mesh, local)
    near = sd <= delta  # (n, 2, 5); inside counts as touching
    held = (near.sum(axis=2) >= HELD_TIPS).any(axis=1)
    opp = np.zeros(n, bool)
    for t in np.nonzero(held)[0]:
        for h in range(2):
            idx = np.nonzero(near[t, h])[0]
            if len(idx) >= 2:
                N = normals[t, h, idx]
                d = N @ N.T
                if (d[np.triu_indices(len(idx), 1)] < OPPOSED_DOT).any():
                    opp[t] = True
                    break
    return {"held": held, "held_opp": opp, "nearest_m": np.clip(sd, 0, None).min(axis=(1, 2))}


def contact_slots(cells: dict, slot: int) -> np.ndarray:
    """(n,) filled contact slots of object `slot`, both hands (NVIDIA zeroes an empty slot)."""
    count = 0
    for side in SIDES:
        pos = np.asarray(cells[f"mano_{side}_object_contact_positions"], float)
        part = np.asarray(cells[f"mano_{side}_object_contact_part_ids"])
        count = count + ((np.linalg.norm(pos, axis=-1) > 1e-3) & (part == slot + 1)).sum(axis=1)
    return count


def support_disks(usd: Path) -> list[dict]:
    """The support USD's cylinders: radius, height, centre (S5's support_surfaces.py writes them)."""
    text = usd.read_text() if usd.is_file() else ""
    disks = []
    for block in re.split(r'def Cylinder "', text)[1:]:
        r = re.search(r"double radius = ([-0-9.eE]+)", block)
        h = re.search(r"double height = ([-0-9.eE]+)", block)
        c = re.search(r"double3 xformOp:translate = \(([^)]*)\)", block)
        if r and h and c:
            disks.append({"name": block.split('"')[0], "radius": float(r.group(1)), "height": float(h.group(1)),
                          "centre": [float(x) for x in c.group(1).split(",")]})
    return disks


def support_ok(objs: list[dict], meshes: list[trimesh.Trimesh], disks: list[dict]) -> dict:
    """Frame 0: each object's bottom rests on a disk (xy inside it, top within SUPPORT_GAP_M) or on another
    object (a bottom point within SUPPORT_GAP_M of it, inside counting, its closest point not above)."""
    posed = [trimesh.Trimesh(m.vertices @ o["poses"][0][:3, :3].T + o["poses"][0][:3, 3], m.faces, process=False)
             for o, m in zip(objs, meshes)]
    out = {}
    for k, pm in enumerate(posed):
        v = np.asarray(pm.vertices)
        z0 = v[:, 2].min()
        foot = trimesh.sample.sample_surface(pm, 4000, seed=0)[0]
        foot = foot[foot[:, 2] <= z0 + 0.005]
        xy = foot[:, :2].mean(axis=0)
        on_disk = any(abs(z0 - (d["centre"][2] + d["height"] / 2)) <= SUPPORT_GAP_M
                      and np.linalg.norm(xy - np.asarray(d["centre"][:2])) <= d["radius"] for d in disks)
        on_object = False
        for j, other in enumerate(posed):
            if j != k:
                sd, _ = surface_distance(other, foot)
                closest = closest_points(other, foot)
                on_object |= bool(((sd <= SUPPORT_GAP_M) & (closest[:, 2] <= foot[:, 2] + 0.005)).any())
        out[objs[k]["name"]] = {"disk": bool(on_disk), "object": on_object, "bottom_z": float(z0)}
    return {"ok": all(v["disk"] or v["object"] for v in out.values()), "objects": out}


def closest_points(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                        o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
    return scene.compute_closest_points(o3d.core.Tensor(np.asarray(points, np.float32)))["points"].numpy()


def coacd_pieces(mesh_path: Path, log: Path) -> list[str]:
    """CoACD pieces of a mesh (drivers/coacd_pieces.py, threshold COACD_THRESHOLD), cached by the mesh's md5."""
    key = hashlib.md5(mesh_path.read_bytes()).hexdigest()
    d = cache_dir() / f"coacd/{key}_{COACD_THRESHOLD}"
    if not (d / "pieces.json").is_file():
        with open(log, "a") as f:
            subprocess.run([coacd_python(), str(PATH_B / "pathb/drivers/coacd_pieces.py"), str(mesh_path), str(d),
                            str(COACD_THRESHOLD)], stdout=f, stderr=subprocess.STDOUT, check=True)
    return [str(d / p["file"]) for p in json.loads((d / "pieces.json").read_text())["pieces"]]


def disk_pieces(disks: list[dict], out: Path) -> list[dict]:
    bodies = []
    for i, d in enumerate(disks):
        f = out / f"disk_{i}.obj"
        trimesh.creation.cylinder(radius=d["radius"], height=d["height"], sections=64).export(f)
        pose = np.eye(4)
        pose[:3, 3] = d["centre"]
        bodies.append({"name": f"disk_{i}", "pieces": [str(f)], "pose": pose.tolist()})
    return bodies


def settle(req: dict, work: Path, tag: str) -> dict:
    q, r = work / f"{tag}_request.json", work / f"{tag}_result.json"
    q.write_text(json.dumps(req))
    with open(work / "settle.log", "a") as f:
        subprocess.run([settle_python(), str(PATH_B / "pathb/drivers/settle.py"), str(q), str(r)],
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    return json.loads(r.read_text())


def urdf_mass(processed: Path, body: str) -> float:
    text = (processed / f"{body}_rigid.urdf").read_text()
    m = re.search(r'<mass value="([-0-9.eE]+)"', text)
    return float(m.group(1)) if m else 0.3


def s3a_pieces(root: Path, objs: list[dict]) -> dict[str, list[str]]:
    """S3a's CoACD pieces (s3a_mesh/<object>/coacd, threshold COACD_THRESHOLD) for every object whose S5 mesh
    is S3a's canonical mesh (same vertices), so RSS need not decompose again."""
    out = {}
    for o in objs:
        d = Path(root) / f"s3a_mesh/{o['name']}"
        if (d / "coacd/pieces.json").is_file() and (d / "mesh.obj").is_file():
            a, b = load_mesh(o["mesh"]), load_mesh(d / "mesh.obj")
            if a.vertices.shape == b.vertices.shape and np.allclose(a.vertices, b.vertices, atol=1e-6):
                out[o["name"]] = [str(d / "coacd" / p["file"])
                                  for p in json.loads((d / "coacd/pieces.json").read_text())["pieces"]]
    return out


def bottom_z(o: dict, f: int) -> float:
    v = np.asarray(o["vertices"])
    return float((v @ o["poses"][f][:3, :3].T + o["poses"][f][:3, 3])[:, 2].min())


def rest_settle(t: dict, objs: list[dict], drops: dict[str, list[int]], work: Path,
                given: dict[str, list[str]] | None = None) -> dict:
    """Drop each object from its rest frames in S5's scene; moved cm / turned deg per drop. Supports first (S3d):
    the objects whose bottom is lower stay fixed where the reference has them, the ones above are left out (a lid
    must not pin its pot). given: CoACD pieces per object (s3a_pieces); the others are decomposed here (cached)."""
    given = given or {}
    pieces = {o["name"]: given.get(o["name"]) or coacd_pieces(o["mesh"], work / "coacd.log") for o in objs}
    for o in objs:
        o.setdefault("vertices", load_mesh(o["mesh"]).vertices)
    disks = disk_pieces(support_disks(t["support_usd"]), work)
    rests = []
    for o in objs:
        mass = urdf_mass(t["processed"], o["body"])
        for f in drops[o["name"]]:
            z = bottom_z(o, f)
            fixed = [{"name": p["name"], "pieces": pieces[p["name"]], "pose": p["poses"][f].tolist()}
                     for p in objs if p is not o and bottom_z(p, f) < z - 0.001] + disks
            req = {"seconds": RSS_SECONDS, "timestep": RSS_TIMESTEP, "table": True, "table_z": GROUND_Z,
                   "fixed": fixed, "free": {"name": o["name"], "pieces": pieces[o["name"]],
                                            "pose": o["poses"][f].tolist(), "mass_kg": mass}}
            res = settle(req, work, f"{o['name']}_{f}")
            rests.append({"object": o["name"], "frame": int(f), "moved_cm": 100 * float(res["moved_m"]),
                          "turned_deg": float(res["turned_deg"])})
    unsettled = [r for r in rests if r["moved_cm"] > 100 * UNSETTLED_M or r["turned_deg"] > UNSETTLED_DEG]
    first = [r for r in rests if r["frame"] == 0]
    return {"rss0_cm": max((r["moved_cm"] for r in first), default=0.0),
            "rss0_deg": max((r["turned_deg"] for r in first), default=0.0),
            "rss_max_cm": max((r["moved_cm"] for r in rests), default=0.0),
            "rss_max_deg": max((r["turned_deg"] for r in rests), default=0.0),
            "unsettled_rests": len(unsettled), "rests": rests, "disks": len(disks)}


def features(cand_dir: Path, candidate: str, invert_reports: dict[str, dict | None], rss: bool = True,
             s3a_root: Path | None = None) -> dict:
    """Trackability of one S5 candidate. invert_reports: S3c's invert_report.json per object name (the
    trained reference's), None where there is none; s3a_root: the episode folder whose s3a_mesh pieces RSS
    reuses (default: the one cand_dir is in)."""
    t = task(Path(cand_dir))
    objs = objects(t)
    root = Path(s3a_root) if s3a_root else Path(cand_dir).parents[1]
    meshes = [load_mesh(o["mesh"]) for o in objs]
    tips = fingertips(t["cells"])
    n = len(tips)
    per, drops = {}, {}
    tot = {"moving": 0, "free": 0, "free_opp": 0, "rcc1": 0, "rcc2": 0}
    dist, runs, runs_opp = [], [], []
    for k, (o, mesh) in enumerate(zip(objs, meshes)):
        lab = labels(invert_reports.get(o["name"]), n)
        moving = np.zeros(n, bool) if candidate == "still" else moving_frames(mesh, o["poses"], lab)
        delta = delta_m(mesh)
        hs = hand_support(mesh, o["poses"], tips, delta)
        free, free_opp = moving & ~hs["held"], moving & ~hs["held_opp"]
        slots = contact_slots(t["cells"], k)
        per[o["name"]] = {"n_moving": int(moving.sum()), "ums": float(free.sum() / max(moving.sum(), 1)),
                          "ums_opp": float(free_opp.sum() / max(moving.sum(), 1)), "ums_run": longest_run(free),
                          "delta_cm": 100 * delta,
                          "d_hand_cm": 100 * float(hs["nearest_m"][moving].mean()) if moving.any() else None,
                          "rcc1": float((slots[moving] >= 1).mean()) if moving.any() else None,
                          "rcc2": float((slots[moving] >= 2).mean()) if moving.any() else None,
                          "held_frames": int(hs["held"].sum())}
        tot["moving"] += int(moving.sum())
        tot["free"] += int(free.sum())
        tot["free_opp"] += int(free_opp.sum())
        tot["rcc1"] += int((slots[moving] >= 1).sum())
        tot["rcc2"] += int((slots[moving] >= 2).sum())
        dist.append(hs["nearest_m"][moving])
        runs.append(longest_run(free))
        runs_opp.append(longest_run(free_opp))
        drops[o["name"]] = [0] if candidate == "still" else sorted({0} | {a for a, _ in lab["static"]})
    m = max(tot["moving"], 1)
    d = np.concatenate(dist)
    out = {"candidate": candidate, "frames": n, "n_moving": tot["moving"], "ums": tot["free"] / m,
           "ums_opp": tot["free_opp"] / m, "ums_run": max(runs), "ums_opp_run": max(runs_opp),
           "d_hand_cm": 100 * float(d.mean()) if len(d) else None,
           "rcc1": tot["rcc1"] / m if tot["moving"] else None, "rcc2": tot["rcc2"] / m if tot["moving"] else None,
           "support": support_ok(objs, meshes, support_disks(t["support_usd"])), "objects": per,
           "s5_warnings": t["info"].get("output_check", {}).get("warnings", [])}
    out["support_ok"] = out["support"]["ok"]
    if rss:
        with tempfile.TemporaryDirectory(prefix="rss_") as work:
            try:
                out["rss"] = rest_settle(t, objs, drops, Path(work), s3a_pieces(root, objs))
            except (OSError, subprocess.CalledProcessError, KeyError, ValueError) as e:  # no sim venv, a bad piece
                log = Path(work) / "settle.log"
                out["rss"] = {"error": f"{type(e).__name__}: {e}"[:300], "log": log.read_text()[-600:] if log.is_file() else ""}
    return out
