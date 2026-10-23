"""S3a mesh: a metric mesh per object from SAM 3D Objects on chosen views (V3_BUILD_SPEC.md S3a).

1. Views: frames whose P1b object mask covers >= 2 % of the image with <= 5 % of it under a hand
   mask; up to 5, one from rest-before and one from rest-after (the opening / closing frames before
   / after a hand touches the object, pathb.phases.rest_window_2d), the rest chosen to spread the
   camera-to-object directions (P4's camera, stereo depth at the mask centre) as far as possible.
2. SAM 3D Objects per view (v2d_sam3d run_image_to_mesh, seed 0) on the RGB crop, the object mask
   and the stereo depth (the module's own _depth_to_pointmap makes the PyTorch3D pointmap); its
   transform.json is the raw mesh's pose in that view's OpenCV camera, with SAM3D's per-axis scale.
3. MV-SAM3D fusion is not wired in (not installed yet); the spec's fallback picks the best single
   view by stereo-depth fit: the lowest median |rendered - stereo depth| inside the mask
   (tools/track3/fit_shape.py's renderer and frame cost).
4. Metric size: that view's mesh x SAM3D's scale, then one uniform correction (0.5-2x) and per-axis
   factors (0.8-1.25x) fitted to mask and stereo depth (fit_shape's cost, trimmed mean) on every
   view where the mesh's pose is known: its own, and the views of the same rest window (the object
   holds still there, so P4's camera carries the pose).
Out per object: mesh.obj (SAM3D's texture kept), mesh.glb (metres, SAM3D canonical frame),
mesh_views.json, coacd/ (CoACD, threshold 0.05, in $FLASH_CHORD_VENV), mass.json (volume x
pathb/density.py), sha256.txt (of mesh.glb). Bars: watertight; largest extent 0.05-0.6 m; stereo
residual < 1 cm (PASS) / 2 cm (WARN); public: CD-O vs the scan < 0.3 / 0.5 cm, size 0.8-1.25x.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess

import cv2
import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .. import baseline as B
from .. import gt
from ..density import density
from ..geometry import read_depth, read_transform, read_transforms
from ..paths import PATH_B, TOOLS_TRACK3
from ..phases import rest_window_2d
from ..report import Metric, StageResult
from ..runner import write_json
from .p3_scale_bridge import stereo_along

MIN_AREA = 0.02  # default: share of the image
MAX_HAND_OVERLAP = 0.05  # default: share of the object mask under a hand mask
VIEWS = 5  # default
REST_CAP = 30  # frames, as P1c/P3
UNIFORM = (0.5, 2.0)  # bounds of the uniform correction (r78's R8 bounds)
AXES = (0.8, 1.25)  # spec: per-axis factors only within this
COACD_THRESHOLD = 0.05


def fit_shape():
    import sys
    if str(TOOLS_TRACK3) not in sys.path:
        sys.path.insert(0, str(TOOLS_TRACK3))
    import fit_shape as F
    return F


def view_candidates(ctx, obj_dir, hand_dirs, n: int, size: tuple[int, int]) -> dict[int, float]:
    """frame -> mask area share, for the frames that pass the area and hand-overlap bars."""
    W, H = size
    out = {}
    for k in range(n):
        m = B.read_mask(obj_dir, k)
        if m is None or m.sum() < MIN_AREA * W * H:
            continue
        hands = np.zeros_like(m)
        for h in hand_dirs:
            hm = B.read_mask(h, k)
            if hm is not None:
                hands |= hm
        if (m & hands).sum() <= MAX_HAND_OVERLAP * m.sum():
            out[k] = float(m.mean())
    return out


def direction(ctx, k: int, mask: np.ndarray, c2w: np.ndarray, K: np.ndarray, K_half: np.ndarray) -> np.ndarray | None:
    """Unit camera-to-object direction in the world (stereo depth along the mask-centre ray)."""
    ys, xs = np.nonzero(mask)
    ray = np.linalg.inv(K) @ np.array([xs.mean(), ys.mean(), 1.0])
    p = stereo_along(read_depth(ctx.root / f"p2_depth/depth_a_half/{k:06d}.png"), K_half, ray / np.linalg.norm(ray))
    if p is None:
        return None
    v = c2w[k, :3, :3] @ p
    return v / np.linalg.norm(v)


def select(cands: dict[int, float], dirs: dict[int, np.ndarray], before: list[int], after: list[int]) -> list[int]:
    """Up to VIEWS frames: the largest mask in rest-before and in rest-after, then greedily the frame
    whose direction is farthest (largest minimum angle) from those already chosen."""
    usable = [k for k in cands if k in dirs]
    chosen = []
    for window in (before, after):
        inside = [k for k in usable if window and window[0] <= k <= window[1] and k not in chosen]
        if inside:
            chosen.append(max(inside, key=lambda k: cands[k]))
    if not chosen and usable:
        chosen.append(max(usable, key=lambda k: cands[k]))
    while len(chosen) < VIEWS:
        rest = [k for k in usable if k not in chosen]
        if not rest:
            break
        chosen.append(max(rest, key=lambda k: min(np.arccos(np.clip(dirs[k] @ dirs[c], -1, 1)) for c in chosen)))
    return sorted(chosen)


SAMPLES = 20000  # surface points per fit, as fit_shape.py


def surface_points(path) -> np.ndarray:
    """SAMPLES points on a mesh's surface (fixed seed), as fit_shape.py fits on."""
    mesh = trimesh.load(path, force="mesh", process=False)
    return np.asarray(mesh.sample(SAMPLES, seed=0), dtype=np.float64)


def scaled_obj(src, dest, transform) -> None:
    """Copy an OBJ with every vertex mapped by `transform` (a function of (N, 3) vertices), keeping
    its material and texture lines and files."""
    lines = []
    for line in open(src):
        if line.startswith("v "):
            p = line.split()
            v = transform(np.array([[float(x) for x in p[1:4]]]))[0]
            line = "v " + " ".join(f"{x:.9g}" for x in v) + ("".join(" " + x for x in p[4:]) + "\n" if len(p) > 4 else "\n")
        lines.append(line)
    dest.write_text("".join(lines))
    for f in src.parent.iterdir():
        if f.suffix.lower() in (".mtl", ".png", ".jpg", ".jpeg") and not (dest.parent / f.name).exists():
            shutil.copy(f, dest.parent / f.name)


def observation(mask: np.ndarray, depth: np.ndarray, S: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape[0] // S, mask.shape[1] // S
    return (cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0,
            cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST))


def residual_cm(F, points_cam: np.ndarray, K4, mask, depth) -> tuple[float, float]:
    """(median |rendered - stereo depth| inside the mask in cm, silhouette IoU)."""
    sil, zb = F.render(points_cam, K4, mask.shape)
    sel = mask & np.isfinite(zb) & (depth > 0)
    res = float(np.median(np.abs(depth[sel] - zb[sel])) * 100) if sel.sum() >= 40 else np.inf
    union = (mask | sil).sum()
    return res, float((mask & sil).sum() / union) if union else 0.0


def size_fit(F, verts: np.ndarray, views: list[tuple[np.ndarray, np.ndarray, np.ndarray]], K4) -> dict:
    """Uniform and per-axis factors (+ a rigid correction in the object frame) minimising fit_shape's
    trimmed frame cost over the views (pose, mask, depth)."""
    c = verts.mean(axis=0)
    _, _, vt = np.linalg.svd(verts - c, full_matrices=False)
    P = vt.T
    lu, la = np.log(UNIFORM), np.log(AXES)

    def shape(x):
        A = P @ np.diag(np.exp(np.clip(x[1:4], *la))) @ P.T * np.exp(np.clip(x[0], *lu))
        return (verts - c) @ A.T + c

    def cost(x):
        v = shape(x) @ Rotation.from_rotvec(x[4:7]).as_matrix().T + x[7:10]
        costs = np.sort([F.frame_cost(m, d, *F.render(v @ T[:3, :3].T + T[:3, 3], K4, m.shape)) for T, m, d in views])
        return float(costs[:max(1, int(F.KEEP * len(costs)) if len(costs) >= 3 else len(costs))].mean())

    before = cost(np.zeros(10))
    best = None
    for u0 in (0.0, -0.2, 0.2):
        x0 = np.r_[u0, np.zeros(9)]
        simplex = [x0] + [x0 + np.eye(10)[k] * (0.15 if k < 4 else 0.05 if k < 7 else 0.02) for k in range(10)]
        r = minimize(cost, x0, method="Nelder-Mead",
                     options={"initial_simplex": np.array(simplex), "maxfev": 600, "xatol": 1e-3, "fatol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    return {"vertices": shape(x), "uniform": float(np.exp(np.clip(x[0], *lu))),
            "axes": np.exp(np.clip(x[1:4], *la)).round(4).tolist(), "cost_before": before, "cost_after": float(best.fun),
            "rigid_deg": float(np.degrees(np.linalg.norm(x[4:7]))), "rigid_cm": float(np.linalg.norm(x[7:10]) * 100),
            "transform": lambda v: ((np.asarray(v) * 1.0 - c) @ (P @ np.diag(np.exp(np.clip(x[1:4], *la))) @ P.T).T
                                    * np.exp(np.clip(x[0], *lu)) + c)}


def mesh_for(ctx, name: str, track: str, d, calib, c2w, have, n) -> tuple[dict, list[Metric], list]:
    F = fit_shape()
    S = F.S
    K = np.array(calib["cam_a"]["K"], dtype=np.float64)
    K4 = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    K_half = np.array(json.loads((ctx.root / "p2_depth/depth_a_half/intrinsics.json").read_text())["K"])
    size = tuple(calib["cam_a"]["size"])
    tracks = B.mask_tracks(ctx.root)
    obj_dir, hand_dirs = tracks[track], [p for t, p in tracks.items() if t.startswith("hand_")]
    out = d / name
    out.mkdir(parents=True, exist_ok=True)
    opening = range(min(n, REST_CAP))
    closing = list(range(n - 1, max(-1, n - 1 - REST_CAP), -1))
    before = rest_window_2d([[B.read_mask(obj_dir, k) for k in opening]],
                            [[B.read_mask(h, k) for k in opening] for h in hand_dirs], cap=REST_CAP)
    after_rev = rest_window_2d([[B.read_mask(obj_dir, k) for k in closing]],
                               [[B.read_mask(h, k) for k in closing] for h in hand_dirs], cap=REST_CAP)
    after = [n - 1 - after_rev[1], n - 1]
    cands = view_candidates(ctx, obj_dir, hand_dirs, n, size)
    dirs = {k: v for k in cands if have[k] and (v := direction(ctx, k, B.read_mask(obj_dir, k), c2w, K, K_half)) is not None}
    views = select(cands, dirs, before, after)
    if not views:
        raise RuntimeError(f"{name}: no frame passes the view bars (mask >= {MIN_AREA:.0%}, hand overlap <= "
                           f"{MAX_HAND_OVERLAP:.0%})")
    log = out / "sam3d.log"
    log.write_text("")
    rows = []
    for k in views:
        v = out / f"views/{k:06d}"
        v.mkdir(parents=True, exist_ok=True)
        inputs = view_inputs(ctx, obj_dir, k)  # a cached view counts only for the same image, mask and depth
        sha = v / "inputs.sha256"
        cached = (v / "transform.json").is_file() and sha.is_file() and sha.read_text().strip() == inputs
        if not cached:
            (v / "transform.json").unlink(missing_ok=True)
            try:
                B.call("v2d.sam3d.docker.run_image_to_mesh:run_image_to_mesh",
                       {"image_path": ctx.root / f"p1_calibrate/cam_a/{k:06d}.jpg", "mask_path": obj_dir / f"{k:06d}.png",
                        "mesh_path": v / "mesh.obj", "transform_path": v / "transform.json",
                        "intrinsics_path": v / "intrinsics.json", "weights_dir": B.WEIGHTS / "sam3d", "seed": 0,
                        "with_mesh_postprocess": True, "with_texture_baking": True,
                        "depth_path": ctx.root / f"p1d_hands/depth/{k:06d}.png",
                        "depth_intrinsics_path": ctx.root / "p1c_camera/intrinsics.json",
                        "depth_mask_path": obj_dir / f"{k:06d}.png"}, log)
            except RuntimeError as e:
                rows.append({"frame": k, "error": str(e).splitlines()[0]})
                continue
            (v / "inputs.sha256").write_text(inputs + "\n")
        T, s = read_transform(v / "transform.json")  # raw mesh -> this view's OpenCV camera, scale s
        raw = surface_points(v / "mesh.obj")
        mask, depth = observation(B.read_mask(obj_dir, k), read_depth(ctx.root / f"p1d_hands/depth/{k:06d}.png"), S)
        res, iou = residual_cm(F, (raw * s) @ T[:3, :3].T + T[:3, 3], K4, mask, depth)
        rows.append({"frame": k, "scale": s.tolist(), "residual_cm": res, "iou": iou,
                     "extents_cm": (np.ptp(raw * s, axis=0) * 100).round(1).tolist()})
    good = [r for r in rows if "error" not in r and np.isfinite(r["residual_cm"])]
    if not good:
        raise RuntimeError(f"{name}: SAM 3D produced no usable view ({[r.get('error') for r in rows]})")
    best = min(good, key=lambda r: r["residual_cm"])
    k = best["frame"]
    v = out / f"views/{k:06d}"
    T_k, s_k = read_transform(v / "transform.json")
    raw = surface_points(v / "mesh.obj") * s_k  # metres, SAM3D canonical frame
    Tk = np.eye(4)
    Tk[:3, :3], Tk[:3, 3] = T_k[:3, :3], T_k[:3, 3]
    window = before if before[0] <= k <= before[1] else after if after[0] <= k <= after[1] else None
    fit_views = []
    for w in ([f for f in range(window[0], window[1] + 1) if have[f]] if window else [k]):
        mask_w = B.read_mask(obj_dir, w)
        if mask_w is None or not mask_w.any() or not (ctx.root / f"p1d_hands/depth/{w:06d}.png").is_file():
            continue
        Tw = np.linalg.inv(c2w[w]) @ c2w[k] @ Tk  # the object holds still in the window
        fit_views.append((Tw, *observation(mask_w, read_depth(ctx.root / f"p1d_hands/depth/{w:06d}.png"), S)))
    fit_views = fit_views[:: max(1, len(fit_views) // 10)]  # at most ~10 frames of one window
    sized = size_fit(F, raw, fit_views, K4)
    after_res = [residual_cm(F, sized["vertices"] @ T[:3, :3].T + T[:3, 3], K4, m, dd)[0] for T, m, dd in fit_views]
    scaled_obj(v / "mesh.obj", out / "mesh.obj", lambda x: sized["transform"](x * s_k))
    mesh = trimesh.load(out / "mesh.obj", force="mesh", process=False)
    trimesh.load(out / "mesh.obj").export(out / "mesh.glb")
    watertight = bool(mesh.is_watertight)
    volume = float(abs(mesh.volume)) if watertight else float(mesh.convex_hull.volume)
    write_json(out / "mass.json", {"volume_m3": volume, "volume_from": "mesh" if watertight else "convex hull",
                                   "density_kg_m3": density(name), "mass_kg": volume * density(name)})
    (out / "sha256.txt").write_text(hashlib.sha256((out / "mesh.glb").read_bytes()).hexdigest() + "  mesh.glb\n")
    flash = os.environ.get("FLASH_CHORD_VENV", "/root/v2d/venvs/flash_chord")
    with open(out / "coacd.log", "w") as f:
        rc = subprocess.run([f"{flash}/bin/python", str(PATH_B / "pathb/drivers/coacd_pieces.py"), str(out / "mesh.obj"),
                             str(out / "coacd"), str(COACD_THRESHOLD)], stdout=f, stderr=subprocess.STDOUT).returncode
    pieces = json.loads((out / "coacd/pieces.json").read_text())["pieces"] if rc == 0 else []
    extents = np.sort(mesh.extents)[::-1]
    report = {"views": views, "rest_before": before, "rest_after": after, "per_view": rows, "chosen_view": k,
              "fusion": "best single view by stereo-depth fit (MV-SAM3D not installed)", "fit_window": window,
              "fit_frames": len(fit_views), **{key: sized[key] for key in ("uniform", "axes", "cost_before", "cost_after",
                                                                            "rigid_deg", "rigid_cm")},
              "residual_cm_after": after_res, "extents_cm": (extents * 100).round(1).tolist(),
              "watertight": watertight, "coacd_pieces": len(pieces)}
    write_json(out / "mesh_views.json", report)
    metrics = [
        Metric(f"{name}: views", len(views), "", note=f"frames {views}; chosen {k} (best stereo fit)"),
        Metric(f"{name}: watertight", int(watertight), "", ok=(1, 1)),
        Metric(f"{name}: largest extent", float(extents[0]), "m", ok=(0.05, 0.6),
               note=" x ".join(f"{e * 100:.1f}" for e in extents) + " cm"),
        Metric(f"{name}: stereo-fit residual (median)", float(np.median(after_res)) if after_res else np.inf, "cm",
               ok=(0, 1), warn=(0, 2), note=f"{len(fit_views)} frames; uniform {sized['uniform']:.3f}, axes {sized['axes']}"),
        Metric(f"{name}: convex pieces", len(pieces), "", note=f"CoACD threshold {COACD_THRESHOLD}"),
    ]
    read = []
    if ctx.public:
        if gt.scan(name):
            metrics.append(Metric(f"{name}: CD-O vs the scan", gt.cdo_cm(ctx.episode, name, mesh.vertices, mesh.faces, read),
                                  "cm", kind="gt", ok=(0, 0.3), warn=(0, 0.5)))
            metrics.append(Metric(f"{name}: size vs the scan", gt.scan_size_ratio(ctx.episode, mesh.vertices, name, read),
                                  "x", kind="gt", ok=(0.8, 1.25)))
    return report, metrics, read


def view_inputs(ctx, obj_dir: Path, k: int) -> str:
    """SHA-256 over one view's SAM 3D inputs: the frame, the object mask and P1d's depth."""
    h = hashlib.sha256()
    for f in (ctx.root / f"p1_calibrate/cam_a/{k:06d}.jpg", obj_dir / f"{k:06d}.png", ctx.root / f"p1d_hands/depth/{k:06d}.png"):
        h.update(f.read_bytes() if f.is_file() else b"-")
    return h.hexdigest()


def run(ctx) -> StageResult:
    d = ctx.dir("s3a_mesh")
    calib = json.loads((ctx.root / "p1_calibrate/calib.json").read_text())
    n = int(ctx.episode["num_frames"])
    c2w, have = read_transforms(ctx.root / "p4_export/camera_to_world", n)
    metrics, outputs = [], {}
    for i, name in enumerate(ctx.objects):
        report, m, read = mesh_for(ctx, name, f"obj_{i}", d, calib, c2w, have, n)
        metrics += m
        ctx.files_read.extend(read)
        outputs[name] = d / name / "mesh.glb"
    return StageResult(metrics, outputs)
