"""S3c segment solving (V4_BUILD_SPEC.md S3c; the rules in pathb/invert.py): per object, the track that
solves each of S3b's segments by its own rule, in the canonical mesh's frame.

In: S3b's segments.json (fingertip sets per grasp), s3a2_register's canonical_to_world (the first
static segment's start), the canonical mesh (s3a_mesh/<object>/mesh.obj), the object's observed
depth inside its P1b mask (pathb/observe.py), the silhouette of its and the hands' masks, a TSDF of
each static segment's masked depth (pathb/scan.py, at most TSDF_FRAMES frames), P1d's world hands
(p4_export/hands/), its symmetry group (pathb/symmetry.py).
Out per object: object_to_world.npz (N, 4, 4), invert_report.json (per segment: kind, frames, start
it came from, residuals, drift, refinement, failed), bundle/ (P4's bundle with this track and the
canonical mesh: the BUNDLES.md layout S4b and the waterfall read).
Bars: static residual median <= 5 mm; grasp segments `failed` (residual (a) within 1 cm on fewer
than half the frames: S4b takes keyframe there) a WARN; no frame-to-frame jump > 3 cm inside a
static or grasp segment. Public, one object: the track's playback AUC (official metric; the waterfall
scores scenes as `ref_inverted` after S3d and S4b).
"""
from __future__ import annotations

import json
import shutil
import time

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .. import gt, symmetry
from .. import invert as I
from .. import scan as S
from ..bundle import p4_bundle
from ..observe import View
from ..paths import TRACK3_FPS
from ..report import Metric, StageResult
from ..runner import write_json
from .s4b_reference import write_bundle

PIXELS = 1500  # observed mask pixels kept per frame (each residual evaluation draws invert.RAYS of them)
TSDF_FRAMES = 30  # default: frames fused per static segment


def clip(ctx, view: View, k: int, name: str, mesh: trimesh.Trimesh) -> I.Clip:
    n = int(ctx.episode["num_frames"])
    rng = np.random.default_rng(k)
    obs = [view.pixels(k, t, PIXELS, rng) for t in range(n)]
    sil = []
    for t in range(n):
        allowed = view.allowed(k, t)
        sil.append(I.silhouette_map(allowed) if allowed is not None else None)
    hd = ctx.root / "p4_export/hands"
    tips = np.load(hd / "fingertips_world.npz")
    wrist = np.load(hd / "wrist_world.npz")["wrist"]

    def target(a, b):
        frames = [t for t in range(a, b + 1) if obs[t] is not None]
        frames = frames[:: max(1, len(frames) // TSDF_FRAMES)]
        fused = []
        for t in frames:
            depth = view.masked_depth(k, t)
            if depth is not None:
                fused.append(S.Frame(t, depth, view.K, view.c2w[t]))
        if not fused:
            return None
        surface = S.largest_part(S.integrate(fused))
        return S.mesh_points(surface, 5000) if len(surface.triangles) else None

    return I.Clip(K=view.K, c2w=view.c2w, obs=obs, tips=tips["tips"], wrist=wrist, valid=tips["valid"],
                  vertices=np.asarray(mesh.vertices, np.float64), faces=np.asarray(mesh.faces),
                  group=symmetry.group(name, np.asarray(mesh.vertices)), target=target, silhouette=sil, seed=k)


def run(ctx) -> StageResult:
    d = ctx.dir("s3c_invert")
    view = View(ctx)
    metrics, outputs = [], {}
    for k, name in enumerate(ctx.objects):
        t0 = time.time()
        mesh = trimesh.load(ctx.root / f"s3a_mesh/{name}/mesh.obj", force="mesh", process=False)
        segs = json.loads((ctx.root / f"s3b_segments/{name}/segments.json").read_text())
        reg = json.loads((ctx.root / f"s3a2_register/{name}/registration.json").read_text())
        c = clip(ctx, view, k, name, mesh)
        P, report = I.solve(c, segs, np.asarray(reg["canonical_to_world"], dtype=np.float64), TRACK3_FPS)
        out = d / name
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "object_to_world.npz", object_to_world=P)
        seconds = time.time() - t0
        write_json(out / "invert_report.json", {"object": name, "seconds": seconds, "segments": report,
                                                "start_pose": reg["canonical_to_world"], "start_window": reg["window"]})
        src = p4_bundle(ctx, name)
        base = dict(np.load(src / "result.npz"))
        c2w = base["camera_to_world_transform"].astype(np.float64)
        base["object_to_world_transform"] = P.astype(np.float32)
        base["object_to_camera_transform"] = (np.linalg.inv(c2w) @ P).astype(np.float32)
        base["object_scale"] = np.asarray(1.0, dtype=np.float32)
        bundle = write_bundle(src, out / "bundle", base)
        for f in ("mesh.obj",):  # the canonical mesh (P4's bundle has none)
            (bundle / f).unlink(missing_ok=True)
            shutil.copy2(ctx.root / f"s3a_mesh/{name}/{f}", bundle / f)
        outputs[name] = out / "object_to_world.npz"
        statics = [r for r in report if r["kind"] == "static" and r.get("residual_m") is not None]
        grasps = [r for r in report if r["kind"] == "grasp"]
        held = [r for r in report if r["kind"] in ("static", "grasp")]
        jump = max((r["max_jump_m"] for r in held), default=0.0)
        metrics += [
            Metric(f"{name}: segments static / grasp / unstable",
                   " / ".join(str(sum(1 for r in report if r["kind"] == kd)) for kd in ("static", "grasp", "unstable")), "",
                   note="; ".join(f"{r['kind']} {r['start']}-{r['end']}" + (f" ({r['why']})" if r.get("why") else "")
                                  for r in report)[:300]),
            Metric(f"{name}: static residual (worst segment median)", max(r["residual_m"] for r in statics) * 1000
                   if statics else None, "mm", ok=(0, I.STATIC_BAR_M * 1000)),
            Metric(f"{name}: grasp segments failed", sum(r["failed"] for r in grasps), "", ok=(0, 0), warn=(0, np.inf),
                   note=", ".join(f"{r['start']}-{r['end']}: {r['share_within_1cm']:.2f} within 1 cm, median "
                                  f"{(r['residual_a_median_m'] or 0) * 100:.2f} cm, drift {r['drift_max_m'] * 100:.1f} cm / "
                                  f"{r['drift_max_deg']:.1f} deg" for r in grasps) or "no grasp"),
            Metric(f"{name}: largest frame-to-frame jump inside static and grasp segments", jump * 100, "cm", ok=(0, 3)),
            Metric(f"{name}: solving time", seconds / 60, "min"),
        ]
        if ctx.public and len(ctx.objects) == 1:
            poses = np.concatenate([P[:, :3, 3], Rotation.from_matrix(P[:, :3, :3]).as_quat()], axis=1)[:, None]
            scores = gt.score(ctx.episode, poses, {name: ctx.root / f"s3a_mesh/{name}/mesh.obj"}, ctx.files_read)
            metrics.append(Metric(f"{name}: playback AUC of the solved track", scores["AUC"], "", kind="gt",
                                  note=f"MPPE {scores['MPPE cm']:.2f} cm; the waterfall's ref_inverted adds S3d's settle"))
    return StageResult(metrics, outputs)
