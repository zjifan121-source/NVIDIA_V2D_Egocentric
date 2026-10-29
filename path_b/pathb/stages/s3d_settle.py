"""S3d settle (V4_BUILD_SPEC.md S3d): every static segment's pose dropped onto its support in MuJoCo
(CPU; drivers/settle.py in $SETTLE_PY, default the daid venv), supports first.

Per static segment of each object (S3c's pose, S3b's `support`): the object's CoACD pieces
(s3a_mesh/<object>/coacd/) as a free body at that pose, its mass (mass.json); the table is the plane
z = 0; a supporting object is a fixed body at its own settled pose (objects on the table settle
first). 0.5 s of gravity; the resting pose replaces the static pose when it moved < 1.5 cm and turned
< 10 degrees, else S3c's pose stays and the segment is `unsettled`. Then S3c's grasps and bridges are
solved once more against the settled poses (their end terms; pathb/invert.solve with them fixed), with the other
bodies of revolution as S3c's hosts (invert.STACK). A rest S3c stacked on another body of revolution's axis rests on
that object here, whatever S3b's `support` says (S3b reads the nested cup's visible rim as standing on the table).
Out per object: object_to_world.npz (N, 4, 4: S4b's `inverted`), settle_report.json, bundle/.
Bars: penetration into the support after settling <= 2 mm; unsettled segments a WARN; convex pieces
present (no pieces: nothing settles, a FAIL).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np
import trimesh

from .. import invert as I
from ..bundle import p4_bundle
from ..observe import View
from ..paths import PATH_B, TRACK3_FPS
from ..report import Metric, StageResult
from ..runner import write_json
from .s3c_invert import clip as s3c_clip
from .s3c_invert import hosts as s3c_hosts
from .s4b_reference import write_bundle

SECONDS, TIMESTEP = 0.5, 0.002  # spec: 0.5 s
MAX_MOVE_M, MAX_TURN_DEG = 0.015, 10.0  # spec
MAX_PENETRATION_M = 0.002  # spec
TABLE_AT_REST = True  # default: a rest on the table whose lowest point S3c fitted more than SUNK_MIN_M below
# z = 0 (at most MAX_SUNK_M) drops onto a table plane TABLE_GAP_M under that point instead (`table_cm` per row). MuJoCo throws a body
# that starts inside the table (public v4_j10: rests starting over 1 cm inside z = 0 settled 3 of 22, those within
# 5 mm 14 of 17), while the task scene the policy trains in has no table at z = 0: NVIDIA's support script puts each
# rest's disk at that rest's own lowest point. So the drop tests what the policy meets (does it stay, tip or slide)
# and keeps the rest's height; raising the rest onto z = 0 instead (v4_k4) moved one object of a pair 1.2 cm against
# the other (ep12 0.718 -> 0.697)
TABLE_GAP_M, MAX_SUNK_M = 0.0005, 0.03
SUNK_MIN_M = 0.002  # default: MuJoCo lifts a body this shallow out of z = 0 without throwing it (v4_j10: 6 of 7 such
# rests settled); moving ep45's first rest (1.4 mm in) onto its own table cost 0.011 through the carry's re-solve


def settle_python() -> str:
    return os.environ.get("SETTLE_PY") or f"{os.environ.get('DAID_VENV', '/root/v2d/venvs/daid')}/bin/python"


def pieces(ctx, name: str) -> list[str]:
    d = ctx.root / f"s3a_mesh/{name}/coacd"
    info = d / "pieces.json"
    return [str(d / p["file"]) for p in json.loads(info.read_text())["pieces"]] if info.is_file() else []


def order(ctx, segs: dict) -> list[str]:
    """Objects resting only on the table first, then the ones that rest on them."""
    on_table = [o for o in ctx.objects if all(s.get("support") in (None, "table") for s in segs[o] if s["kind"] == "static")]
    return on_table + [o for o in ctx.objects if o not in on_table]


def run(ctx) -> StageResult:
    d = ctx.dir("s3d_settle")
    py = settle_python()
    segs = {o: json.loads((ctx.root / f"s3b_segments/{o}/segments.json").read_text()) for o in ctx.objects}
    for o in ctx.objects:  # S3c's stacked rests rest on their host
        f = ctx.root / f"s3c_invert/{o}/invert_report.json"
        rows = json.loads(f.read_text())["segments"] if f.is_file() else []
        on = {r["start"]: r["stacked"]["host"] for r in rows
              if r["kind"] == "static" and (r.get("stacked") or {}).get("kept") == "stacked"}
        for seg in segs[o]:
            if seg["kind"] == "static" and seg["start"] in on:
                seg["support"] = on[seg["start"]]
    tracks = {o: np.load(ctx.root / f"s3c_invert/{o}/object_to_world.npz")["object_to_world"] for o in ctx.objects}
    settled, metrics, outputs = {}, [], {}
    view = None  # built only when a grasp or bridge is solved again
    for name in order(ctx, segs):
        out = d / name
        out.mkdir(parents=True, exist_ok=True)
        mass = json.loads((ctx.root / f"s3a_mesh/{name}/mass.json").read_text())["mass_kg"]
        mine = pieces(ctx, name)
        vertices = np.asarray(trimesh.load(ctx.root / f"s3a_mesh/{name}/mesh.obj", force="mesh", process=False).vertices)
        rows, fixed = [], {}
        for s in (s for s in segs[name] if s["kind"] == "static"):
            a, b = s["start"], s["end"]
            pose = tracks[name][a]
            row = {"segment": [a, b], "support": s.get("support")}
            low = float((vertices @ pose[:3, :3].T + pose[:3, 3])[:, 2].min())
            table_z = 0.0
            if TABLE_AT_REST and s.get("support") in (None, "table") and -MAX_SUNK_M <= low < -SUNK_MIN_M:
                table_z = low - TABLE_GAP_M
                row["table_cm"] = table_z * 100
            if not mine or not os.path.exists(py):
                rows.append(row | {"settled": False, "why": "no convex pieces" if not mine else f"no MuJoCo python at {py}"})
                continue
            under = s.get("support")
            fixed_bodies = []
            if under not in (None, "table"):
                support = settled.get(under, tracks.get(under))
                if support is None or not pieces(ctx, under):
                    rows.append(row | {"settled": False, "why": f"support {under} not available"})
                    continue
                fixed_bodies = [{"name": under, "pieces": pieces(ctx, under), "pose": support[a].tolist()}]
            req = {"seconds": SECONDS, "timestep": TIMESTEP, "table": True, "table_z": table_z, "fixed": fixed_bodies,
                   "free": {"name": name, "pieces": mine, "pose": pose.tolist(), "mass_kg": mass}}
            rq, rs = out / f"settle_{a:06d}.json", out / f"settled_{a:06d}.json"
            write_json(rq, req)
            with open(out / "settle.log", "a") as f:
                rc = subprocess.run([py, str(PATH_B / "pathb/drivers/settle.py"), str(rq), str(rs)], stdout=f,
                                    stderr=subprocess.STDOUT).returncode
            if rc:
                rows.append(row | {"settled": False, "why": f"settle.py failed (exit {rc}, {out / 'settle.log'})"})
                continue
            r = json.loads(rs.read_text())
            ok = r["moved_m"] < MAX_MOVE_M and r["turned_deg"] < MAX_TURN_DEG
            if ok:
                fixed[a] = np.asarray(r["pose"])
            rows.append(row | {"settled": ok, "moved_cm": r["moved_m"] * 100, "turned_deg": r["turned_deg"],
                               "penetration_mm": r["penetration_m"] * 1000, "contacts": r["contacts"],
                               "why": "" if ok else "moved too far: S3c's pose kept (unsettled)"})
        P = tracks[name].copy()
        report = []
        for a, pose in fixed.items():
            b = next(s["end"] for s in segs[name] if s["start"] == a)
            P[a:b + 1] = pose
        if fixed and any(s["kind"] != "static" for s in segs[name]):  # S3c's end terms once more, against the settled poses
            reg = json.loads((ctx.root / f"s3a2_register/{name}/registration.json").read_text())
            mesh = trimesh.load(ctx.root / f"s3a_mesh/{name}/mesh.obj", force="mesh", process=False)
            view = view or View(ctx)
            others = {o: (settled.get(o, tracks[o]), [(g["start"], g["end"]) for g in segs[o] if g["kind"] == "static"])
                      for o in ctx.objects if o != name}
            k = ctx.objects.index(name)
            P, report = I.solve(s3c_clip(ctx, view, k, name, mesh, s3c_hosts(ctx, view, k, others)), segs[name],
                                np.asarray(reg["canonical_to_world"], dtype=np.float64), TRACK3_FPS, fixed=fixed)
        settled[name] = P
        np.savez(out / "object_to_world.npz", object_to_world=P)
        write_json(out / "settle_report.json", {"object": name, "statics": rows, "resolved": report})
        src = p4_bundle(ctx, name)
        base = dict(np.load(src / "result.npz"))
        c2w = base["camera_to_world_transform"].astype(np.float64)
        base["object_to_world_transform"] = P.astype(np.float32)
        base["object_to_camera_transform"] = (np.linalg.inv(c2w) @ P).astype(np.float32)
        base["object_scale"] = np.asarray(1.0, dtype=np.float32)
        bundle = write_bundle(src, out / "bundle", base)
        (bundle / "mesh.obj").unlink(missing_ok=True)
        shutil.copy2(ctx.root / f"s3a_mesh/{name}/mesh.obj", bundle / "mesh.obj")
        if not (bundle / "material.mtl").exists():  # run_load_retarget.merged_bundle reads each object's
            (bundle / "material.mtl").write_text("newmtl material_0\n")
        outputs[name] = out / "object_to_world.npz"
        done = [r for r in rows if "penetration_mm" in r]
        metrics += [
            Metric(f"{name}: convex pieces", len(mine), "", ok=(1, np.inf), note="s3a_mesh coacd/: nothing settles without them"),
            Metric(f"{name}: static segments settled / unsettled / not run",
                   f"{sum(r['settled'] for r in rows)} / {sum(1 for r in done if not r['settled'])} / {len(rows) - len(done)}",
                   "", note="; ".join(f"{r['segment'][0]}-{r['segment'][1]} on {r['support']}: "
                                      + (f"moved {r['moved_cm']:.2f} cm, turned {r['turned_deg']:.1f} deg" if "moved_cm" in r else r["why"])
                                      for r in rows)[:300]),
            Metric(f"{name}: penetration into the support after settling (worst)",
                   max(r["penetration_mm"] for r in done) if done else None, "mm", ok=(0, MAX_PENETRATION_M * 1000)),
            Metric(f"{name}: unsettled static segments", sum(1 for r in done if not r["settled"]), "", ok=(0, 0), warn=(0, np.inf)),
        ]
        if report:
            grasps = [r for r in report if r["kind"] == "grasp"]
            metrics.append(Metric(f"{name}: grasp segments failed after the settle", sum(r["failed"] for r in grasps), "",
                                  ok=(0, 0), warn=(0, np.inf)))
    return StageResult(metrics, outputs)
