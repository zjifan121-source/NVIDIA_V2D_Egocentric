#!/usr/bin/env python3
"""render_review.py -- render and measure one V2D Track-3 reconstruction scene (GT-free).

Writes into --out (every JPG <= --max-width px wide, default 2000):
  <label>_overlay.jpg   key frames (4 x 3): RGB frame + every object mesh as a semi-transparent
                        silhouette + outline (one colour per object), both reference hands as MANO
                        skeletons (left green, right red; is_valid False = not drawn + marked;
                        valid but not observed by HaMeR = dotted + marked), rollout achieved poses
                        as dashed outlines (second colour) with --rollout; frame index + heights
  <label>_side.jpg      the same key frames as orthographic side views (gravity up): view A keeps
                        the objects apart, view B has the camera ray in the picture plane; object
                        meshes, hand joints, support prims from the support USD, table plane
  <label>_frame0.jpg    per object: frame-0 RGB crop with outline + two side close-ups against the
                        support (view across the camera ray and view along the camera ray)
  <label>_plots.jpg     time series over all frames (heights, rotation step + twist about the mesh
                        symmetry axis, cumulative twist, centroid step, per-hand nearest-joint
                        distance + is_valid + observed/filled + contact labels, silhouette IoU vs
                        our SAM3 masks, rollout ADD / ADD-S)
  <label>_numbers.json  the numbers behind the plots + frame source / alignment / transform notes

Usage (Babel, inside a CPU step; the anaconda python imports in <1 s):
  srun --jobid=11035728 --overlap -n1 -c2 /home/haisenk/anaconda3/bin/python \\
      ~/v2d_ops/kenc/review/tool/render_review.py --scene <scene dir> --episode N \\
      --label <short name> --out <dir> [--frames <dir|video>] [--bundle <dir|result.npz>] \\
      [--parquet <processed parquet>] [--rollout <results/episode_0000NN.parquet> \\
      --rollout-label <name>] [--nkey 12] [--keyframes 0,20,...] [--ncol 4] \\
      [--frame-offset 0] [--masks-root <root>|none] [--side-azimuth DEG]
  Scenes without bundle_cleaned/ (e.g. Wentao's wv2_* scenes): pass --bundle <dir with the
  result.npz the loader read>; the tool refuses a bundle whose frame count differs and reports
  the residual of the loader transform (a wrong bundle shows up as a large residual).

Coordinates: everything is drawn in the loader (sim) world of the processed parquet (z up; the
world the retarget, the policy, the evaluator and the support USD use).  The camera comes from the
bundle result.npz and is mapped into that world with the loader transform
A = T_parquet(obj, t) @ inv(T_bundle(obj, t)), estimated over all objects and frames (residual is
reported; the ego_recon loader applies one rigid A to camera, hands and objects).  Hands are the
parquet MANO joints (what retarget/policy consume).  Rollout achieved poses are in the same loader
world; this is checked against the rollout's own reference columns.

Frames (auto, when --frames is not given): manifest sources.frames_dir if it still exists, else
its sibling anycalib/undistorted.mp4 (the undistorted ego_cam_a video the ken_c pipeline ran on),
else any <kenc-root>/episode_0000NN/*/anycalib/undistorted.mp4 whose undistorted_intrinsics.json
equals the bundle intrinsics, else the raw ego_cam_a video (distorted; warned).  Videos are
decoded sequentially (ffmpeg, passthrough timing) and bundle frame i = decoded frame i.  If our
SAM3 object masks exist (<kenc-root>/episode_0000NN/<object>/masks/1/%06d.png) the tool checks the
index alignment automatically (silhouette-vs-mask IoU at offsets -3..+3 on moving frames).
"""
import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402
from scipy.spatial import ConvexHull, cKDTree  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

KENC_ROOT = ("/data/user_data/haisenk/v2d/v2d_recon/baseline/reconstruction/data/outputs/"
             "ken_c_scored")
RAW_VIDEO_ROOT = ("/data/user_data/haisenk/v2d/v2d_train/assets/data/track_3/evaluation/videos/"
                  "chunk-000")

OBJ_COLORS = ["#2a78d6", "#eb6834", "#6250d6", "#eda100", "#e87ba4", "#8c564b"]
ROLL_COLORS = ["#00e5ff", "#ffe14d", "#ff5cf4", "#f5f5f5", "#9dff00", "#d2b48c"]
HAND_IMG = {"left": "#2ee66b", "right": "#ff3b3b"}      # on photos
HAND_PLOT = {"left": "#008300", "right": "#e34948"}     # on white
MANO_PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
SIDES = ("left", "right")
TXT_BOX = dict(facecolor="black", alpha=0.6, pad=1.5, edgecolor="none")
_T0 = time.time()


def log(msg):
    print(f"[render_review +{time.time() - _T0:6.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- small geometry utils
def to_T(R, t):
    R = np.asarray(R, float)
    T = np.zeros(R.shape[:-2] + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def safe_xyzw(q):
    q = np.array(q, float).reshape(-1, 4)
    n = np.linalg.norm(q, axis=1)
    bad = ~np.isfinite(n) | (n < 1e-8)
    q[bad] = [0.0, 0.0, 0.0, 1.0]
    return q


def wxyz_to_R(q):
    q = np.asarray(q, float).reshape(-1, 4)
    return Rot.from_quat(safe_xyzw(q[:, [1, 2, 3, 0]])).as_matrix()


def xyzw_to_R(q):
    return Rot.from_quat(safe_xyzw(q)).as_matrix()


def rot_angle_deg(Ra, Rb):
    """Geodesic angle (deg) between rotation stacks Ra, Rb (..., 3, 3)."""
    M = np.einsum("...ji,...jk->...ik", Ra, Rb)
    c = (np.trace(M, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def xf(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def project(Pw, w2c, K):
    """Pw (V,3) world -> pixel (V,2) (OpenCV, pixel centres at integers), valid, depth."""
    Pc = Pw @ w2c[:3, :3].T + w2c[:3, 3]
    z = Pc[:, 2]
    ok = z > 0.02
    zz = np.where(ok, z, 1.0)
    uv = np.stack([K[0] * Pc[:, 0] / zz + K[2], K[1] * Pc[:, 1] / zz + K[3]], 1)
    return uv, ok, z


def raster_mask(xy, faces, H, W, vok=None, chunk=20000):
    """Union of projected triangles (pixel-centre test; pixel centres at integer coords)."""
    mask = np.zeros((H, W), bool)
    faces = np.asarray(faces)
    if vok is not None:
        faces = faces[vok[faces].all(1)]
    if len(faces) == 0 or H <= 0 or W <= 0:
        return mask
    S = 8
    oy, ox = np.divmod(np.arange(S * S), S)
    big_tris = []
    for c0 in range(0, len(faces), chunk):
        tri = xy[faces[c0:c0 + chunk]]
        if not np.isfinite(tri).all():
            tri = tri[np.isfinite(tri).all((1, 2))]
        x0 = np.ceil(tri[:, :, 0].min(1)).astype(np.int64)
        x1 = np.floor(tri[:, :, 0].max(1)).astype(np.int64)
        y0 = np.ceil(tri[:, :, 1].min(1)).astype(np.int64)
        y1 = np.floor(tri[:, :, 1].max(1)).astype(np.int64)
        ok = (x1 >= x0) & (y1 >= y0) & (x1 >= 0) & (y1 >= 0) & (x0 < W) & (y0 < H)
        small = ok & (x1 - x0 < S) & (y1 - y0 < S)
        big = ok & ~small
        if big.any():
            big_tris.append(tri[big])
        if small.any():
            t = tri[small]
            px = x0[small][:, None] + ox[None, :]
            py = y0[small][:, None] + oy[None, :]
            ax_, ay_ = t[:, 0, 0, None], t[:, 0, 1, None]
            bx_, by_ = t[:, 1, 0, None], t[:, 1, 1, None]
            cx_, cy_ = t[:, 2, 0, None], t[:, 2, 1, None]
            e0 = (bx_ - ax_) * (py - ay_) - (by_ - ay_) * (px - ax_)
            e1 = (cx_ - bx_) * (py - by_) - (cy_ - by_) * (px - bx_)
            e2 = (ax_ - cx_) * (py - cy_) - (ay_ - cy_) * (px - cx_)
            ins = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))
            ins &= (px >= 0) & (px < W) & (py >= 0) & (py < H)
            mask[py[ins], px[ins]] = True
    if big_tris:
        img = Image.new("L", (W, H), 0)
        dr = ImageDraw.Draw(img)
        for t in np.concatenate(big_tris):
            dr.polygon([(float(p[0]), float(p[1])) for p in t], fill=1)
        mask |= np.asarray(img, bool)
    return mask


def blend(img, m, color, alpha):
    if not m.any():
        return img
    c = np.array(matplotlib.colors.to_rgb(color))
    img[m] = img[m] * (1 - alpha) + c * alpha
    return img


def over_rgba(rgba, m, color, alpha):
    if not m.any():
        return rgba
    c = np.array(matplotlib.colors.to_rgb(color))
    a0 = rgba[m, 3:4]
    a_out = alpha + a0 * (1 - alpha)
    rgba[m, :3] = (c * alpha + rgba[m, :3] * a0 * (1 - alpha)) / np.maximum(a_out, 1e-6)
    rgba[m, 3:4] = a_out
    return rgba


def draw_skel(ax, P2, color, ls="-", lw=1.6, ms=6, z=6, ok=None):
    segs = []
    for j, p in enumerate(MANO_PARENTS):
        if p < 0:
            continue
        if ok is not None and not (ok[j] and ok[p]):
            continue
        segs.append([P2[p], P2[j]])
    if segs:
        ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, linestyles=ls,
                                         zorder=z, capstyle="round"))
    sel = np.ones(len(P2), bool) if ok is None else ok
    ax.scatter(P2[sel, 0], P2[sel, 1], s=ms, c=color, zorder=z + 1, linewidths=0)


def contour(ax, m, color, lw=1.4, ls="-", x=None, y=None, z=5):
    """Outline of a boolean mask; x/y = pixel-centre coordinates (default: pixel indices)."""
    if m is None or not m.any() or m.all():
        return
    if x is None:
        x = np.arange(m.shape[1], dtype=float)
        y = np.arange(m.shape[0], dtype=float)
    if len(y) > 1 and y[0] > y[-1]:
        y = y[::-1]
        m = m[::-1]
    mm = np.pad(m.astype(float), 1)
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    dy = y[1] - y[0] if len(y) > 1 else 1.0
    xs = np.concatenate([[x[0] - dx], x, [x[-1] + dx]])
    ys = np.concatenate([[y[0] - dy], y, [y[-1] + dy]])
    ax.contour(xs, ys, mm, levels=[0.5], colors=[color], linewidths=lw, linestyles=ls, zorder=z)


# ----------------------------------------------------------------------------- file readers
def load_obj(path):
    vs, fs = [], []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                vs.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    fs.append((idx[0], idx[k], idx[k + 1]))
    V = np.array(vs, float)
    F = np.array(fs, np.int64).reshape(-1, 3)
    F = np.where(F > 0, F - 1, len(V) + F)
    return V, F


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _nums(s):
    return [float(x) for x in re.findall(_NUM, s or "")]


def _local_matrix(attrs):
    M = np.eye(4)
    order = attrs.get("xformOpOrder")
    if not order:
        return M
    for op in re.findall(r'"([^"]+)"', order):
        inv = op.startswith("!invert!")
        key = op[len("!invert!"):] if inv else op
        val = attrs.get(key)
        if val is None:
            continue
        v = _nums(val)
        kind = key.split(":")[1] if ":" in key else key
        T = np.eye(4)
        if kind == "translate" and len(v) >= 3:
            T[:3, 3] = v[:3]
        elif kind == "scale" and len(v) >= 1:
            T = np.diag((v[:3] if len(v) >= 3 else [v[0]] * 3) + [1.0])
        elif kind.startswith("rotate") and len(kind) > 6 and v:
            axes = kind[6:]
            angs = v[:len(axes)]
            R = np.eye(3)
            for a_, g in zip(axes, angs):
                R = Rot.from_euler(a_.lower(), g, degrees=True).as_matrix() @ R
            T[:3, :3] = R
        elif kind == "orient" and len(v) >= 4:
            T[:3, :3] = Rot.from_quat(safe_xyzw([v[1], v[2], v[3], v[0]]))[0].as_matrix()
        elif kind == "transform" and len(v) >= 16:
            T = np.array(v[:16]).reshape(4, 4).T
        else:
            continue
        if inv:
            T = np.linalg.inv(T)
        M = M @ T
    return M


def _geom_points(prim):
    t, a = prim["type"], prim["attrs"]

    def num(key, default):
        v = _nums(a.get(key, ""))
        return v[0] if v else default

    ax_ = re.findall(r'"(\w)"', a.get("axis", '"Z"'))
    ax_ = ax_[0].upper() if ax_ else "Z"
    if t in ("Cylinder", "Capsule", "Cone"):
        r, h = num("radius", 1.0), num("height", 2.0)
        if t == "Capsule":
            h += 2 * r
        ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        ring = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
        P = np.concatenate([np.c_[ring, np.full(64, -h / 2)], np.c_[ring, np.full(64, h / 2)]])
        if ax_ == "X":
            P = P[:, [2, 0, 1]]
        elif ax_ == "Y":
            P = P[:, [1, 2, 0]]
        return P
    if t == "Cube":
        s = num("size", 2.0) / 2
        return np.array([[x, y, z] for x in (-s, s) for y in (-s, s) for z in (-s, s)])
    if t == "Sphere":
        r = num("radius", 1.0)
        i = np.arange(200) + 0.5
        phi = np.arccos(1 - 2 * i / 200)
        th = np.pi * (1 + 5 ** 0.5) * i
        return r * np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    if t == "Mesh":
        pts = _nums(a.get("points", ""))
        if len(pts) >= 9:
            return np.array(pts[:len(pts) // 3 * 3]).reshape(-1, 3)
    return None


def parse_usda(path):
    """Minimal ASCII-USD reader for support files (Xform/Cylinder/Cube/Sphere/Capsule/Mesh)."""
    root = dict(type="__root__", name="", attrs={}, children=[])
    stack, pending, acc = [root], None, None
    with open(path, "r", errors="ignore") as fh:
        lines = fh.read().splitlines()
    for raw in lines:
        s = raw.strip()
        if acc is not None:
            acc[1] += " " + s
            if acc[1].count("[") <= acc[1].count("]") and acc[1].count("(") <= acc[1].count(")"):
                stack[-1]["attrs"][acc[0]] = acc[1]
                acc = None
            continue
        if not s or s.startswith("#"):
            continue
        m = re.match(r'^(def|over|class)\s+(?:(\w+)\s+)?"([^"]*)"', s)
        if m:
            pending = dict(type=m.group(2) or "", name=m.group(3), attrs={}, children=[])
            if s.endswith("{"):
                stack[-1]["children"].append(pending)
                stack.append(pending)
                pending = None
            continue
        if s.endswith("{") and "=" in s:
            stack.append(dict(type="__dict__", name="", attrs={}, children=[]))
            continue
        if s == "{" or s.endswith("{"):
            if pending is not None:
                stack[-1]["children"].append(pending)
                stack.append(pending)
                pending = None
            else:
                stack.append(dict(type="__block__", name="", attrs={}, children=[]))
            continue
        if s.startswith("}"):
            if len(stack) > 1:
                stack.pop()
            continue
        if stack[-1]["type"] in ("__root__", "__dict__", "__block__") or pending is not None:
            continue
        m = re.match(r"^(?:(?:uniform|custom|varying|prepend|append)\s+)*([\w\[\]]+)\s+([\w:!]+)"
                     r"\s*=\s*(.*)$", s)
        if m:
            name, val = m.group(2), m.group(3)
            if val.count("[") > val.count("]") or val.count("(") > val.count(")"):
                acc = [name, val]
            else:
                stack[-1]["attrs"][name] = val
    prims = []

    def walk(node, Mp):
        for ch in node["children"]:
            M = Mp @ _local_matrix(ch["attrs"])
            g = _geom_points(ch)
            if g is not None and len(g):
                P = g @ M[:3, :3].T + M[:3, 3]
                prims.append(dict(name=ch["name"], type=ch["type"], pts=P))
            walk(ch, M)

    walk(root, np.eye(4))
    for p in prims:
        P = p["pts"]
        p["top_z"] = float(P[:, 2].max())
        p["bottom_z"] = float(P[:, 2].min())
        xy = P[:, :2]
        try:
            hull = ConvexHull(xy)
            p["hull_xy"] = xy[hull.vertices]
            p["area"] = float(hull.volume)
        except Exception:
            p["hull_xy"] = None
            p["area"] = 0.0
        p["path"] = MplPath(p["hull_xy"]) if p["hull_xy"] is not None else None
    return prims


def find_parquet(scene):
    c = glob.glob(os.path.join(scene, "assets/human_motion_data/ego_recon/processed/sequence_id=*",
                               "robot_name=*", "*.parquet"))
    if not c:
        sys.exit(f"ERROR: no processed parquet under {scene}/assets/human_motion_data/ego_recon/"
                 "processed (pass --parquet)")
    c.sort(key=os.path.getmtime)
    if len(c) > 1:
        log(f"WARNING: {len(c)} processed parquets; using newest {c[-1]}")
    return c[-1]


def load_scene_parquet(path):
    want = ["fps", "raw_motion_file", "object_name", "object_body_names", "object_mesh_paths",
            "object_body_position", "object_body_wxyz", "mano_left_joints", "mano_right_joints",
            "mano_left_object_contact_part_ids", "mano_right_object_contact_part_ids"]
    names = pq.read_schema(path).names
    t = pq.read_table(path, columns=[c for c in want if c in names])
    d = {c: t.column(c)[0].as_py() for c in t.column_names}
    out = dict(path=path, fps=float(d.get("fps") or 20.25411910142744),
               raw_motion_file=d.get("raw_motion_file"), object_name=d.get("object_name") or "",
               body_names=list(d["object_body_names"]), mesh_paths=list(d["object_mesh_paths"]))
    out["pos"] = np.array(d["object_body_position"], float)
    out["wxyz"] = np.array(d["object_body_wxyz"], float)
    if out["pos"].ndim == 2:
        out["pos"] = out["pos"][:, None]
        out["wxyz"] = out["wxyz"][:, None]
    out["N"] = out["pos"].shape[0]
    out["joints"] = {s: np.array(d[f"mano_{s}_joints"], float) for s in SIDES
                     if d.get(f"mano_{s}_joints") is not None}
    out["contact"] = {s: np.array(d[f"mano_{s}_object_contact_part_ids"], np.int64) for s in SIDES
                      if d.get(f"mano_{s}_object_contact_part_ids") is not None}
    return out


def load_bundle(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    c2w = z["camera_to_world_transform"].astype(float)
    N = c2w.shape[0]
    K = np.asarray(z["camera_intrinsics"], float)
    if K.shape == (3, 3):
        K = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
    if K.ndim == 1:
        K = np.tile(K[None, :4], (N, 1))
    elif K.ndim == 3:
        K = np.stack([K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]], 1)
    names = [str(n) for n in z["object_names"]] if "object_names" in z.files else None
    nobj = len(names) if names else 1
    objT = []
    for i in range(nobj):
        key = "object_to_world_transform" if i == 0 else f"object_{i}_to_world_transform"
        objT.append(z[key].astype(float) if key in z.files else None)
    hv = {s: (z[f"hand_{s}_is_valid"].astype(bool) if f"hand_{s}_is_valid" in z.files
              else np.ones(N, bool)) for s in SIDES}
    ov = z["object_is_valid"].astype(bool) if "object_is_valid" in z.files else np.ones(N, bool)
    cv = z["camera_is_valid"].astype(bool) if "camera_is_valid" in z.files else np.ones(N, bool)
    return dict(path=npz_path, N=N, c2w=c2w, K=K, names=names, objT=objT, hand_valid=hv,
                obj_valid=ov, cam_valid=cv)


def _rigid_ok(T):
    ok = np.isfinite(T).all((1, 2))
    det = np.full(len(T), np.nan)
    det[ok] = np.linalg.det(T[ok, :3, :3])
    return ok & (np.abs(det - 1) < 1e-2)


def estimate_A(Tsim_list, Tb_list, corners_list):
    """A with Tsim ~= A @ Tb for every object/frame; residual = max bbox-corner error (m).

    Frames whose bundle or parquet transform is not a finite rotation are ignored (NaN residual)."""
    As = []
    oks = []
    for Ts, Tb in zip(Tsim_list, Tb_list):
        ok = _rigid_ok(Tb) & _rigid_ok(Ts)
        oks.append(ok)
        As.append(Ts[ok] @ np.linalg.inv(Tb[ok]))
    Aall = np.concatenate(As)
    if len(Aall) == 0:
        raise ValueError("no valid object transforms to estimate the loader transform")
    t = np.median(Aall[:, :3, 3], 0)
    U, _, Vt = np.linalg.svd(Aall[:, :3, :3].mean(0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    A = to_T(R, t)
    res = []
    for Ts, Tb, C, ok in zip(Tsim_list, Tb_list, corners_list, oks):
        r = np.full(len(Ts), np.nan)
        ATb = A[None] @ Tb[ok]
        pa = np.einsum("nij,cj->nci", ATb[:, :3, :3], C) + ATb[:, None, :3, 3]
        ps = np.einsum("nij,cj->nci", Ts[ok, :3, :3], C) + Ts[ok][:, None, :3, 3]
        r[ok] = np.linalg.norm(pa - ps, axis=2).max(1)
        res.append(r)
    return A, np.array(res)


# ----------------------------------------------------------------------------- frames
def list_image_dir(d):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    fs = sorted(f for f in os.listdir(d) if f.lower().endswith(exts))
    nums = []
    for f in fs:
        m = re.findall(r"\d+", os.path.splitext(f)[0])
        nums.append(int(m[-1]) if m else None)
    if fs and all(n is not None for n in nums) and len(set(nums)) == len(nums):
        return {n: os.path.join(d, f) for n, f in zip(nums, fs)}, "frame number parsed from file name"
    return {i: os.path.join(d, f) for i, f in enumerate(fs)}, "sorted file order"


def read_intr_json(p):
    try:
        j = json.load(open(p))
        return [float(j["fx"]), float(j["fy"]), float(j["cx"]), float(j["cy"])]
    except Exception:
        return None


def resolve_frames(args, bundle, manifest):
    K0 = bundle["K"][0]
    tried = []

    def kdiff(p):
        kk = read_intr_json(p)
        return None if kk is None else max(abs(kk[i] - K0[i]) for i in range(4))

    if args.frames:
        f = args.frames
        if os.path.isdir(f):
            return dict(kind="dir", path=f, how="--frames (image dir)", tried=tried)
        if os.path.isfile(f):
            return dict(kind="video", path=f, how="--frames (video)", tried=tried)
        sys.exit(f"ERROR: --frames {f} does not exist")
    fd = ((manifest or {}).get("sources") or {}).get("frames_dir")
    if fd:
        if os.path.isdir(fd) and os.listdir(fd):
            return dict(kind="dir", path=fd, how="manifest sources.frames_dir", tried=tried)
        tried.append(f"manifest frames_dir {fd} (missing)")
        v = os.path.join(os.path.dirname(fd), "anycalib", "undistorted.mp4")
        if os.path.isfile(v):
            d = kdiff(os.path.join(os.path.dirname(v), "undistorted_intrinsics.json"))
            if d is not None and d < 0.5:
                return dict(kind="video", path=v, tried=tried, intr_maxdiff_px=d,
                            how="sibling of the manifest frames_dir: anycalib/undistorted.mp4 "
                                f"(undistorted_intrinsics.json == bundle K, max diff {d:.3g} px)")
            tried.append(f"{v} (intrinsics diff {d})")
    best = None
    for v in sorted(glob.glob(os.path.join(args.kenc_root, f"episode_{args.episode:06d}", "*",
                                           "anycalib", "undistorted.mp4"))):
        d = kdiff(os.path.join(os.path.dirname(v), "undistorted_intrinsics.json"))
        tried.append(f"{v} (intrinsics diff {d})")
        if d is not None and (best is None or d < best[0]):
            best = (d, v)
    if best and best[0] < 0.5:
        return dict(kind="video", path=best[1], tried=tried, intr_maxdiff_px=best[0],
                    how="ken_c_scored anycalib/undistorted.mp4 whose undistorted_intrinsics.json "
                        f"== bundle K (max diff {best[0]:.3g} px)")
    if best and best[0] < 5:
        return dict(kind="video", path=best[1], tried=tried, intr_maxdiff_px=best[0],
                    how=f"WARNING closest ken_c undistorted video, intrinsics differ by {best[0]:.2f}"
                        " px")
    raw = os.path.join(args.raw_video_root, "observation.images.ego_cam_a",
                       f"episode_{args.episode:06d}.mp4")
    if os.path.isfile(raw):
        return dict(kind="video", path=raw, tried=tried,
                    how="WARNING raw ego_cam_a video (DISTORTED; bundle intrinsics are for the "
                        "undistorted frames -> outlines off towards the image border)")
    sys.exit("ERROR: no frames found (tried: " + "; ".join(tried) + "); pass --frames")


def load_frames(src, needed, off=0):
    """Frames keyed by BUNDLE index; source index = bundle index + off."""
    vid = {int(i) + off: int(i) for i in needed}
    if src["kind"] == "dir":
        m, how = list_image_dir(src["path"])
        frames = {}
        for vi, bi in vid.items():
            if vi in m:
                frames[bi] = np.asarray(Image.open(m[vi]).convert("RGB"))
        n = len(m)
        size = (frames[min(frames)].shape[1], frames[min(frames)].shape[0]) if frames else None
        return frames, n, size, f"image dir, index = {how}"
    v = src["path"]
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=width,height,r_frame_rate,avg_frame_rate", "-of", "json", v],
                        capture_output=True, text=True)
    st = json.loads(pr.stdout)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    frames, n = {}, 0
    for mode in (["-fps_mode", "passthrough"], ["-vsync", "0"]):
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "2", "-i", v, *mode, "-f",
               "rawvideo", "-pix_fmt", "rgb24", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        fsz = w * h * 3
        frames, n = {}, 0
        while True:
            buf = proc.stdout.read(fsz)
            if len(buf) < fsz:
                break
            if n in vid:
                frames[vid[n]] = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
            n += 1
        proc.stdout.close()
        proc.wait()
        if n > 0:
            break
    note = (f"video decoded sequentially with ffmpeg {' '.join(mode)} ({n} frames, "
            f"container r_frame_rate {st.get('r_frame_rate')}); bundle frame i = decoded frame "
            f"i{'' if off == 0 else ' %+d (--frame-offset)' % off}")
    return frames, n, (w, h), note


# ----------------------------------------------------------------------------- measurements
def sym_axis(V):
    C = V - V.mean(0)
    w, E = np.linalg.eigh(C.T @ C / max(len(V), 1))
    w, E = w[::-1], E[:, ::-1]
    lam = np.sqrt(np.maximum(w, 0))
    dist = [min(abs(lam[i] - lam[j]) for j in range(3) if j != i) / max(lam[0], 1e-9)
            for i in range(3)]
    i = int(np.argmax(dist))
    return E[:, i], ("long" if i == 0 else ("short" if i == 2 else "middle")), lam


def twist_deg(dR, axis):
    q = Rot.from_matrix(dR).as_quat()
    tw = 2 * np.degrees(np.arctan2(q[:, :3] @ axis, q[:, 3]))
    return (tw + 180.0) % 360.0 - 180.0


def first_run(mask, run=2):
    c = 0
    for i, v in enumerate(mask):
        c = c + 1 if v else 0
        if c >= run:
            return i - run + 1
    return None


def pick_keyframes(N, nkey, objs, contact_cnt, short):
    cands = [(0, 0, "f0")]
    for k, o in enumerate(objs):
        nm = short[k]
        dz = o["bottom"] - o["bottom"][0]
        dzs = median_filter(dz, size=5, mode="nearest")
        if np.nanmax(dzs) >= 0.015:
            pk = int(np.nanargmax(dzs))
            on = int(np.argmax(dzs > 0.01))
            while on > 0 and dzs[on - 1] > 0.003:
                on -= 1
            cands += [(on, 1, f"{nm} lift"), (pk, 1, f"{nm} peak")]
            rest = dzs[-1]
            if rest < 0.015:
                after = np.where(dzs[pk:] <= rest + 0.01)[0]
                if len(after):
                    cands.append((pk + int(after[0]), 1, f"{nm} set-down"))
        else:
            disp = np.linalg.norm(o["centroid"] - o["centroid"][0], axis=1)
            if disp.max() > 0.02:
                cands += [(int(np.argmax(disp > 0.01)), 1, f"{nm} moves"),
                          (int(np.argmax(disp)), 1, f"{nm} max disp")]
    for s in SIDES:
        for k in range(len(objs)):
            cnt = contact_cnt.get(s, {}).get(k)
            if cnt is None or not (cnt > 0).any():
                continue
            on = first_run(cnt > 0, 2)
            if on is not None:
                cands.append((on, 2, f"{s[0].upper()} touch {short[k]}"))
            cands.append((int(np.where(cnt > 0)[0][-1]), 4, f"{s[0].upper()} release {short[k]}"))
    cands.append((N - 1, 3, "last"))
    cands.sort(key=lambda c: (c[1], c[0]))
    chosen = []
    for f, p, r in cands:
        f = int(np.clip(f, 0, N - 1))
        hit = next((c for c in chosen if abs(c[0] - f) <= 2), None)
        if hit is not None:
            hit[2].append(r)
            continue
        if len(chosen) < nkey:
            chosen.append([f, p, [r]])
    while len(chosen) < min(nkey, N):
        fs = np.array([c[0] for c in chosen])
        d = np.abs(np.arange(N)[:, None] - fs[None, :]).min(1)
        if d.max() < 1:
            break
        chosen.append([int(np.argmax(d)), 5, ["fill"]])
    chosen.sort(key=lambda c: c[0])
    return [c[0] for c in chosen], {c[0]: c[2] for c in chosen}


# ----------------------------------------------------------------------------- main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default=None, help="image dir or video; default auto")
    ap.add_argument("--bundle", default=None, help="dir with result.npz (or the npz); default auto")
    ap.add_argument("--parquet", default=None, help="processed parquet; default auto")
    ap.add_argument("--rollout", default=None, help="policy-eval results/episode_0000NN.parquet")
    ap.add_argument("--rollout-label", default="rollout")
    ap.add_argument("--nkey", type=int, default=12)
    ap.add_argument("--keyframes", default=None, help="comma list overriding auto key frames")
    ap.add_argument("--kenc-root", default=KENC_ROOT)
    ap.add_argument("--raw-video-root", default=RAW_VIDEO_ROOT)
    ap.add_argument("--masks-root", default=None,
                    help="root with <root>/episode_%%06d/<object>/masks/1/%%06d.png (default "
                         "--kenc-root); 'none' disables the mask IoU / alignment check")
    ap.add_argument("--max-width", type=int, default=2000)
    ap.add_argument("--frame-offset", type=int, default=0,
                    help="source frame index = bundle index + offset (default 0; the mask check "
                         "in numbers.json reports the best offset relative to this one)")
    ap.add_argument("--ncol", type=int, default=4,
                    help="panels per row in overlay/side (e.g. --keyframes 48 --ncol 1 = one "
                         "full-width full-resolution panel for a close look)")
    ap.add_argument("--side-azimuth", type=float, default=None,
                    help="force side-view azimuth (deg, view direction in loader-world xy)")
    return ap.parse_args()


def main():
    args = parse_args()
    scene = os.path.abspath(args.scene)
    os.makedirs(args.out, exist_ok=True)
    timing = {}
    notes = []
    L = args.label

    # ---------------- scene parquet + meshes
    pq_path = args.parquet or find_parquet(scene)
    P = load_scene_parquet(pq_path)
    N, Kb = P["N"], P["pos"].shape[1]
    proc_dir = os.path.dirname(os.path.dirname(os.path.dirname(pq_path)))
    log(f"parquet {pq_path}: {N} frames, {Kb} bodies {P['body_names']}")

    # ---------------- bundle (camera)
    cands = []
    if args.bundle:
        cands.append(args.bundle if args.bundle.endswith(".npz")
                     else os.path.join(args.bundle, "result.npz"))
    cands += [os.path.join(scene, "bundle_cleaned", "result.npz")]
    if P["raw_motion_file"]:
        cands.append(P["raw_motion_file"])
    cands += [os.path.join(scene, "bundle", "result.npz"),
              os.path.join(os.path.dirname(scene), "bundle", "result.npz")]
    bundle, tried_b = None, []
    for c in cands:
        if not os.path.isfile(c):
            tried_b.append(f"{c} (missing)")
            continue
        try:
            b = load_bundle(c)
        except Exception as e:  # noqa: BLE001
            tried_b.append(f"{c} ({type(e).__name__}: {e})")
            continue
        if b["N"] != N:
            tried_b.append(f"{c} (N={b['N']} != parquet {N})")
            continue
        bundle = b
        break
    if bundle is None:
        sys.exit("ERROR: no usable result.npz (tried: " + "; ".join(tried_b) + "); pass --bundle")
    bdir = os.path.dirname(bundle["path"])
    manifest = None
    try:
        manifest = json.load(open(os.path.join(bdir, "manifest.json")))
    except Exception:
        pass
    names = bundle["names"] if bundle["names"] and len(bundle["names"]) == Kb else None
    if names is None:
        pre = (P["object_name"] + "_") if P["object_name"] else ""
        names = [b[len(pre):] if pre and b.startswith(pre) else b for b in P["body_names"]]
    short = names
    log(f"bundle {bundle['path']} objects {names}")

    objs = []
    for k in range(Kb):
        mp = P["mesh_paths"][k]
        cand_m = [mp, os.path.join(proc_dir, os.path.basename(mp))]
        cand_m.append(os.path.join(bdir, "mesh.obj" if k == 0 else f"mesh_{k}.obj"))
        mpath = next((c for c in cand_m if c and os.path.isfile(c)), None)
        if mpath is None:
            sys.exit(f"ERROR: mesh for body {k} not found: {cand_m}")
        V, F = load_obj(mpath)
        if mpath.startswith(bdir):
            sc = 1.0
            try:
                z = np.load(bundle["path"])
                sc = float(z["object_scale"] if k == 0 else z[f"object_{k}_scale"])
            except Exception:
                pass
            V = V * sc
            notes.append(f"body {k}: mesh taken from the bundle dir (scale {sc})")
        R = wxyz_to_R(P["wxyz"][:, k]).reshape(N, 3, 3)
        T = to_T(R, P["pos"][:, k])
        lo, hi = V.min(0), V.max(0)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])])
        ax_, axname, lam = sym_axis(V)
        surf = np.concatenate([V, V[F].mean(1)]) if len(F) else V
        objs.append(dict(name=names[k], mesh=mpath, V=V, F=F, T=T, R=R, corners=corners,
                         axis=ax_, axis_kind=axname, lam=lam, tree=cKDTree(surf),
                         color=OBJ_COLORS[k % len(OBJ_COLORS)],
                         rcolor=ROLL_COLORS[k % len(ROLL_COLORS)]))
    timing["load"] = time.time() - _T0

    # ---------------- loader transform A (bundle world -> loader world)
    Tb_list = [bundle["objT"][k] if k < len(bundle["objT"]) else None for k in range(Kb)]
    use = [k for k in range(Kb) if Tb_list[k] is not None]
    A, res = estimate_A([objs[k]["T"] for k in use], [Tb_list[k] for k in use],
                        [objs[k]["corners"] for k in use])
    res_info = {objs[k]["name"]: dict(max_mm=float(np.nanmax(res[i]) * 1000),
                                      median_mm=float(np.nanmedian(res[i]) * 1000),
                                      frames_over_1mm=int((res[i] > 1e-3).sum()),
                                      frames_not_rigid=int((~np.isfinite(res[i])).sum()))
                for i, k in enumerate(use)}
    A_info = dict(R=A[:3, :3].round(6).tolist(), t=A[:3, 3].round(6).tolist(),
                  rot_deg=float(rot_angle_deg(np.eye(3), A[:3, :3])),
                  residual_per_object=res_info)
    worst = max(v["max_mm"] for v in res_info.values())
    log(f"loader transform A: t={np.round(A[:3, 3], 4)} rot={A_info['rot_deg']:.3f} deg; residual "
        f"max {worst:.3f} mm")
    if worst > 2.0:
        notes.append(f"WARNING parquet object poses differ from bundle poses after the loader "
                     f"transform (max {worst:.1f} mm); drawn = parquet (what the sim uses)")
    c2w = A[None] @ bundle["c2w"]
    w2c = np.linalg.inv(c2w)
    Kc = bundle["K"]

    # ---------------- support USD
    usd = sorted(glob.glob(os.path.join(scene, "assets/human_motion_data/ego_recon/"
                                        "reconstructed_stage", "*.usda")))
    supports = []
    if usd:
        try:
            supports = parse_usda(usd[0])
        except Exception as e:  # noqa: BLE001
            notes.append(f"WARNING support USD parse failed: {e}")
    log(f"support USD {usd[0] if usd else None}: {len(supports)} prims "
        f"{[(s['name'], round(s['top_z'], 4)) for s in supports]}")

    # ---------------- per-object per-frame measurements (loader world, z up)
    for o in objs:
        Pw = np.einsum("nij,vj->nvi", o["R"], o["V"]) + o["T"][:, None, :3, 3]
        z = Pw[..., 2]
        o["bottom"] = z.min(1)
        o["top"] = z.max(1)
        o["cz"] = z.mean(1)
        low = z < (o["bottom"][:, None] + 0.003)
        o["xy_low"] = (Pw[..., :2] * low[..., None]).sum(1) / low.sum(1)[:, None]
        o["centroid"] = Pw.mean(1)
        del Pw
    f0_bottoms = np.array([o["bottom"][0] for o in objs])
    near = [s for s in supports if abs(s["top_z"] - f0_bottoms.min()) < 0.10 and s["area"] > 0]
    if near:
        tbl = max(near, key=lambda s: s["area"])
        z_table, table_note = tbl["top_z"], f"top of largest support near the f0 bottoms ({tbl['name']})"
    elif supports:
        tbl = max(supports, key=lambda s: s["area"])
        z_table, table_note = tbl["top_z"], f"top of largest support ({tbl['name']})"
    else:
        z_table = float(f0_bottoms.min())
        table_note = "no support prims: table plane = lowest object bottom at frame 0 (loader ground)"
    log(f"table plane z={z_table:.4f} ({table_note})")
    for o in objs:
        sup_top = np.full(N, np.nan)
        sup_name = np.array([""] * N, dtype=object)
        for s in supports:
            if s["path"] is None:
                continue
            inside = s["path"].contains_points(o["xy_low"])
            okk = inside & (s["top_z"] <= o["cz"]) & ~(sup_top >= s["top_z"])
            sup_top[okk] = s["top_z"]
            sup_name[okk] = s["name"]
        o["sup_top"], o["sup_name"] = sup_top, sup_name
        o["h_support"] = o["bottom"] - sup_top
        o["h_table"] = o["bottom"] - z_table
        dR = np.einsum("nji,njk->nik", o["R"][:-1], o["R"][1:])
        o["rot_step"] = np.r_[0.0, rot_angle_deg(np.eye(3)[None], dR)]
        o["twist"] = np.r_[0.0, twist_deg(dR, o["axis"])]
        o["twist_cum"] = np.cumsum(o["twist"])
        o["pos_step"] = np.r_[0.0, np.linalg.norm(np.diff(o["centroid"], axis=0), axis=1)]

    # ---------------- hands
    hv = bundle["hand_valid"]
    refilled, raw_note, detected = hand_observation(args, scene, bundle, manifest, N, hv)
    for s in SIDES:
        log(f"{s} hand: is_valid {int(hv[s].sum())}/{N}, filled (not observed) "
            f"{int(refilled[s].sum())} frames")
    hand_dist = {s: {} for s in SIDES}
    hand_dist_joint = {s: {} for s in SIDES}
    for s in SIDES:
        J = P["joints"].get(s)
        if J is None:
            continue
        for k, o in enumerate(objs):
            Jb = np.einsum("nji,nkj->nki", o["R"], J - o["T"][:, None, :3, 3])
            d, _ = o["tree"].query(Jb.reshape(-1, 3))
            d = d.reshape(N, -1)
            hand_dist[s][k] = d.min(1)
            hand_dist_joint[s][k] = d.argmin(1)
    contact_cnt = {s: {k: (P["contact"][s] == k + 1).sum(1) for k in range(Kb)}
                   for s in SIDES if s in P["contact"]}

    # ---------------- rollout
    roll = None
    if args.rollout:
        roll = load_rollout(args.rollout, P, objs, names, N, notes)

    # ---------------- key frames
    if args.keyframes:
        keys = sorted(set(int(x) for x in args.keyframes.split(",") if x.strip() != ""))
        keys = [k for k in keys if 0 <= k < N]
        reasons = {k: ["given"] for k in keys}
    else:
        keys, reasons = pick_keyframes(N, args.nkey, objs, contact_cnt, short)
    log(f"key frames {keys}")

    # ---------------- frames
    fsrc = resolve_frames(args, bundle, manifest)
    frames, nfr, (Wimg, Himg), fnote = load_frames(fsrc, set(keys) | {0}, args.frame_offset)
    timing["frames"] = time.time() - _T0
    log(f"frames: {fsrc['how']}: {fsrc['path']} -> {nfr} frames {Wimg}x{Himg}")
    if nfr != N:
        notes.append(f"WARNING frame source has {nfr} frames, bundle {N}")
    if abs(Kc[0, 2] - Wimg / 2) > 0.2 * Wimg or abs(Kc[0, 3] - Himg / 2) > 0.2 * Himg:
        notes.append("WARNING principal point far from the image centre: frames/intrinsics mismatch?")

    # ---------------- mask IoU + index alignment check (our SAM3 masks, GT-free)
    iou, align = mask_check(args, objs, w2c, Kc, N, Wimg, Himg)
    timing["masks"] = time.time() - _T0

    ctx = dict(args=args, P=P, objs=objs, N=N, keys=keys, reasons=reasons, frames=frames,
               W=Wimg, H=Himg, w2c=w2c, c2w=c2w, K=Kc, hv=hv, refilled=refilled,
               supports=supports, z_table=z_table, roll=roll, short=short, fps=P["fps"],
               cam_valid=bundle["cam_valid"])
    if not bundle["cam_valid"].all():
        notes.append(f"camera is_valid False on {int((~bundle['cam_valid']).sum())} frames "
                     "(marked on the panels)")
    side, sideB = choose_side(ctx)
    ctx["side"], ctx["sideB"] = side, sideB
    log(f"side view A azimuth {side['phi_deg']:.0f} deg (camera ray {side['cam_angle_deg']:.0f} deg"
        f" out of the view direction); view B "
        f"{'omitted' if sideB is None else '%.0f deg' % sideB['phi_deg']}")

    out = {}
    out["overlay"] = os.path.join(args.out, f"{L}_overlay.jpg")
    render_overlay(out["overlay"], ctx)
    log("overlay written")
    out["side"] = os.path.join(args.out, f"{L}_side.jpg")
    render_side(out["side"], ctx)
    log("side written")
    out["frame0"] = os.path.join(args.out, f"{L}_frame0.jpg")
    render_frame0(out["frame0"], ctx)
    log("frame0 written")
    out["plots"] = os.path.join(args.out, f"{L}_plots.jpg")
    render_plots(out["plots"], ctx, contact_cnt, hand_dist, iou)
    log("plots written")
    timing["total_s"] = time.time() - _T0

    # ---------------- numbers
    def r4(a):
        a = np.asarray(a, float)
        return [None if not np.isfinite(x) else round(float(x), 4) for x in a.ravel()]

    align_note = (f"frames: {fsrc['how']} -> {fsrc['path']}; {fnote}. "
                  + ("Index alignment (silhouette vs our SAM3 masks): "
                     + "; ".join(f"{a['object']}: {a['verdict']}" for a in align)
                     if align else "No SAM3 masks found: alignment only checked visually."))
    J = dict(
        label=L, episode=args.episode, scene=scene, parquet=pq_path, bundle=bundle["path"],
        bundle_candidates_tried=tried_b, manifest_frames_dir=((manifest or {}).get("sources") or {})
        .get("frames_dir"), frames=dict(source=fsrc["path"], how=fsrc["how"], decode=fnote,
                                        n_frames_source=nfr, size=[Wimg, Himg],
                                        tried=fsrc.get("tried")),
        frame_alignment_notes=align_note, alignment_check=align, n_frames=N, fps=P["fps"],
        intrinsics_frame0=r4(Kc[0]), loader_transform_A=A_info,
        world_note="all heights/poses in the loader (sim) world, z up; camera = A @ bundle c2w",
        table_plane=dict(z=round(float(z_table), 5), how=table_note),
        supports=[dict(name=s["name"], type=s["type"], top_z=round(s["top_z"], 5),
                       area_m2=round(s["area"], 5)) for s in supports],
        support_usd=usd[0] if usd else None,
        objects=[dict(name=o["name"], mesh=o["mesh"], n_vertices=len(o["V"]),
                      n_faces=len(o["F"]), extents_cm=r4((o["V"].max(0) - o["V"].min(0)) * 100),
                      twist_axis_body=r4(o["axis"]), twist_axis_kind=o["axis_kind"],
                      pca_sd_cm=r4(o["lam"] * 100), color=o["color"]) for o in objs],
        keyframes=[dict(frame=k, reasons=reasons.get(k, [])) for k in keys],
        side_view={nm: (None if v is None else dict(
            what=v["name"], azimuth_deg=v["phi_deg"], camera_ray_out_of_view_dir_deg=v["cam_angle_deg"],
            bounds_cm=r4(v["bounds"]), note="u = horizontal (cm) about the frame-0 object median, "
            "v = height above the table plane (cm)")) for nm, v in (("A", side), ("B", sideB))},
        notes=notes + ([raw_note] if raw_note else []),
        runtime_s=round(time.time() - _T0, 1), timing=timing, outputs=out,
        per_frame=dict(
            camera_is_valid=[bool(x) for x in bundle["cam_valid"]],
            objects={o["name"]: dict(
                h_table_cm=r4(o["h_table"] * 100), h_support_cm=r4(o["h_support"] * 100),
                support_beneath=[str(x) for x in o["sup_name"]],
                bottom_z=r4(o["bottom"]), rot_step_deg=r4(o["rot_step"]),
                twist_deg=r4(o["twist"]), twist_cum_deg=r4(o["twist_cum"]),
                centroid_step_cm=r4(o["pos_step"] * 100), centroid=r4(o["centroid"]),
                mask_iou=(r4(iou[k]) if iou is not None and k in iou else None))
                for k, o in enumerate(objs)},
            hands={s: dict(is_valid=[bool(x) for x in hv[s]],
                           filled_not_observed=[bool(x) for x in refilled[s]],
                           observed_raw=(None if detected[s] is None
                                         else [bool(x) for x in detected[s]]),
                           nearest_joint_dist_cm={objs[k]["name"]: r4(v * 100)
                                                  for k, v in hand_dist[s].items()},
                           nearest_joint_index={objs[k]["name"]: [int(x) for x in v]
                                                for k, v in hand_dist_joint[s].items()},
                           contact_links={objs[k]["name"]: [int(x) for x in v]
                                          for k, v in contact_cnt.get(s, {}).items()})
                   for s in SIDES}),
    )
    if roll is not None:
        J["rollout"] = dict(path=args.rollout, label=args.rollout_label, mapping=roll["mapping"],
                            frame0_check=roll["frame0"], ref_vs_scene=roll["ref_vs_scene"],
                            summary=roll["summary"],
                            per_frame={objs[k]["name"]: dict(
                                add_cm=r4(roll["add"][k] * 100), adds_cm=r4(roll["adds"][k] * 100),
                                h_table_cm=r4((roll["bottom"][k] - z_table) * 100))
                                for k in range(Kb)})
    with open(os.path.join(args.out, f"{L}_numbers.json"), "w") as fh:
        json.dump(J, fh, indent=1)
    log(f"done in {time.time() - _T0:.1f} s -> {args.out}")


# ----------------------------------------------------------------------------- hand observation
def hand_observation(args, scene, bundle, manifest, N, hv):
    """Frames where a hand is valid in the bundle but was not observed (filled/interpolated).

    Sources (GT-free): (a) the ken_c run the bundle names in manifest sources.{left,right}_hand_dir:
    a frame counts as observed when hamer_aligned/<track>/<frame>.json exists (the *_filled dir has
    every frame); (b) otherwise the first ken_c run of the episode, track chosen by
    hand_tracks.json is_right (labelled 'assumed'); (c) frames invalid in scene/bundle_merged but
    valid in the bundle (refilled by the cleaner)."""
    filled = {s: np.zeros(N, bool) for s in SIDES}
    detected = {s: None for s in SIDES}
    notes = []
    src = (manifest or {}).get("sources") or {}
    for s in SIDES:
        d = src.get(f"{s}_hand_dir")
        cand, how = None, None
        if d:
            cand = d.replace("hamer_aligned_filled", "hamer_aligned")
            how = f"manifest {s}_hand_dir -> {cand}"
        else:
            runs = sorted(glob.glob(os.path.join(args.kenc_root, f"episode_{args.episode:06d}", "*",
                                                 "hand_tracks.json")))
            if runs:
                try:
                    tr = json.load(open(runs[0]))["tracks"]
                    tid = [t["object_id"] for t in tr if bool(t.get("is_right")) == (s == "right")]
                    if tid:
                        cand = os.path.join(os.path.dirname(runs[0]), "hamer_aligned", str(tid[0]))
                        how = (f"ASSUMED (bundle names no hand source): ken_c run "
                               f"{os.path.dirname(runs[0])} track {tid[0]}")
                except Exception:
                    pass
        if cand and os.path.isdir(cand) and cand != d:
            det = np.array([os.path.isfile(os.path.join(cand, f"{f:06d}.json")) for f in range(N)])
            detected[s] = det
            filled[s] |= hv[s] & ~det
            notes.append(f"{s} hand observed frames from {how}: {int(det.sum())}/{N}")
    mpath = os.path.join(scene, "bundle_merged", "result.npz")
    if os.path.isfile(mpath) and os.path.abspath(mpath) != os.path.abspath(bundle["path"]):
        try:
            zm = np.load(mpath)
            for s in SIDES:
                kk = f"hand_{s}_is_valid"
                if kk in zm.files and len(zm[kk]) == N:
                    ref = hv[s] & ~zm[kk].astype(bool)
                    filled[s] |= ref
                    notes.append(f"{s} hand: {int(ref.sum())} frames valid in the bundle but invalid "
                                 f"in {mpath} (refilled by the cleaner)")
        except Exception:
            pass
    return filled, "; ".join(notes) if notes else None, detected


# ----------------------------------------------------------------------------- rollout
def load_rollout(path, P, objs, names, N, notes):
    t = pq.read_table(path)
    md = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
    d = t.to_pydict()
    env = np.array(d["environment_id"])
    sel = env == env.min()
    frame = np.array(d.get("reference_frame", d["step"]))[sel].astype(int)
    bid = np.array(d["object_body_id"])[sel].astype(int)
    bname = np.array([str(x) for x in d["object_body_name"]])[sel]
    valid = np.array(d.get("environment_valid", [True] * len(env)))[sel].astype(bool)
    order = md.get("flash_chord.quaternion_order", "xyzw")

    def stack(pref):
        p = np.stack([np.array(d[f"{pref}_position_{a}"], float)[sel] for a in "xyz"], 1)
        q = np.stack([np.array(d[f"{pref}_quaternion_{a}"], float)[sel] for a in "xyzw"], 1)
        return p, q

    ap_, aq_ = stack("achieved")
    rp_, rq_ = stack("reference")
    if order.lower() == "wxyz":   # columns named x,y,z,w but stored w-first (not seen so far)
        aq_ = aq_[:, [1, 2, 3, 0]]
        rq_ = rq_[:, [1, 2, 3, 0]]
    Kb = len(objs)
    ubodies = sorted(set(zip(bid.tolist(), bname.tolist())))
    mapping = {}
    for k in range(Kb):
        nm = P["body_names"][k]
        m = [b for b in ubodies if b[1] == nm]
        how = "exact body name"
        if not m:
            m = [b for b in ubodies if b[1].endswith("_" + names[k]) or b[1] == names[k]]
            how = "object-name suffix"
        if not m:
            m = [b for b in ubodies if b[0] == k]
            how = "body index"
        mapping[k] = (m[0], how) if m else (None, "unmatched")
    ach = [np.full((N, 4, 4), np.nan) for _ in range(Kb)]
    ref = [np.full((N, 4, 4), np.nan) for _ in range(Kb)]
    for k in range(Kb):
        b = mapping[k][0]
        if b is None:
            continue
        rows = np.where((bid == b[0]) & valid & (frame >= 0) & (frame < N))[0]
        fr = frame[rows]
        ach[k][fr] = to_T(xyzw_to_R(aq_[rows]), ap_[rows])
        ref[k][fr] = to_T(xyzw_to_R(rq_[rows]), rp_[rows])
    # rollout reference vs this scene's reference
    rvs = {}
    allA = []
    for k, o in enumerate(objs):
        okf = np.isfinite(ref[k][:, 0, 0])
        if not okf.any():
            continue
        dp = np.linalg.norm(ref[k][okf, :3, 3] - o["T"][okf, :3, 3], axis=1)
        da = rot_angle_deg(ref[k][okf, :3, :3], o["T"][okf, :3, :3])
        rvs[o["name"]] = dict(max_pos_mm=float(dp.max() * 1000), max_rot_deg=float(da.max()),
                              frames=int(okf.sum()))
        allA.append(o["T"][okf] @ np.linalg.inv(ref[k][okf]))
    worst_p = max([v["max_pos_mm"] for v in rvs.values()] or [0])
    worst_a = max([v["max_rot_deg"] for v in rvs.values()] or [0])
    B = np.eye(4)
    if allA and (worst_p > 2.0 or worst_a > 1.0):
        Aall = np.concatenate(allA)
        U, _, Vt = np.linalg.svd(Aall[:, :3, :3].mean(0))
        B = to_T(U @ Vt, np.median(Aall[:, :3, 3], 0))
        notes.append(f"WARNING rollout reference != scene reference (max {worst_p:.1f} mm / "
                     f"{worst_a:.2f} deg): achieved poses mapped with a rigid median transform B")
        mp = "rigid median transform B (rollout reference does not equal this scene's reference)"
    else:
        mp = ("identity: rollout reference == scene parquet reference (loader world), "
              f"max {worst_p:.2f} mm / {worst_a:.3f} deg")
    ach = [B[None] @ a for a in ach]
    ref = [B[None] @ r for r in ref]
    # frame 0 + ADD / ADD-S (vs the rollout's own reference)
    rng = np.random.default_rng(0)
    f0 = {}
    add, adds, bottom = [], [], []
    for k, o in enumerate(objs):
        V = o["V"]
        Vs = V[rng.choice(len(V), min(len(V), 2000), replace=False)]
        a_k = np.full(N, np.nan)
        s_k = np.full(N, np.nan)
        b_k = np.full(N, np.nan)
        for f in range(N):
            if not np.isfinite(ach[k][f, 0, 0]) or not np.isfinite(ref[k][f, 0, 0]):
                continue
            pa = xf(ach[k][f], Vs)
            pr = xf(ref[k][f], Vs)
            a_k[f] = np.linalg.norm(pa - pr, axis=1).mean()
            s_k[f] = cKDTree(pr).query(pa)[0].mean()
            b_k[f] = xf(ach[k][f], V)[:, 2].min()
        add.append(a_k)
        adds.append(s_k)
        bottom.append(b_k)
        if np.isfinite(ach[k][0, 0, 0]):
            f0[o["name"]] = dict(
                pos_mm=float(np.linalg.norm(ach[k][0, :3, 3] - ref[k][0, :3, 3]) * 1000),
                rot_deg=float(rot_angle_deg(ach[k][0, :3, :3], ref[k][0, :3, :3])),
                vs_scene_ref_pos_mm=float(np.linalg.norm(ach[k][0, :3, 3] - o["T"][0, :3, 3])
                                          * 1000))
    summ = {objs[k]["name"]: dict(mean_add_cm=float(np.nanmean(add[k]) * 100),
                                  mean_adds_cm=float(np.nanmean(adds[k]) * 100),
                                  frames=int(np.isfinite(add[k]).sum()))
            for k in range(Kb) if np.isfinite(add[k]).any()}
    log(f"rollout {path}: mapping {mp}; frame0 {f0}; summary {summ}")
    return dict(ach=ach, ref=ref, add=add, adds=adds, bottom=bottom, frame0=f0,
                ref_vs_scene=rvs, summary=summ, B=B,
                mapping=dict(how=mp, bodies={objs[k]["name"]: [mapping[k][0][1]
                                                               if mapping[k][0] else None,
                                                               mapping[k][1]]
                                             for k in range(Kb)},
                             quaternion_order=order, metadata_reference_parquet=md.get(
                                 "flash_chord.reference_parquet")))


# ----------------------------------------------------------------------------- masks / alignment
def mask_check(args, objs, w2c, Kc, N, W, H):
    root = args.masks_root or args.kenc_root
    if str(root).lower() == "none":
        return None, []
    s = 4
    h4, w4 = H // s, W // s
    iou, align = {}, []

    def load_small(p):
        try:
            im = Image.open(p).convert("L")
        except Exception:
            return None
        if im.size != (W, H):
            return None
        if im.getextrema()[1] == 1:          # 0/1-valued masks
            im = im.point(lambda v: 255 if v else 0)
        return np.asarray(im.crop((0, 0, w4 * s, h4 * s)).reduce(s)) > 127

    for k, o in enumerate(objs):
        d = os.path.join(root, f"episode_{args.episode:06d}", o["name"], "masks", "1")
        if not os.path.isdir(d):
            continue
        t0 = time.time()
        off = args.frame_offset
        with ThreadPoolExecutor(8) as ex:
            loaded = list(ex.map(load_small, [os.path.join(d, f"{f + off:06d}.png")
                                              for f in range(N)]))
        t1 = time.time()
        have = np.array([m is not None for m in loaded])
        if not have.any():
            continue
        msk = np.stack([m if m is not None else np.zeros((h4, w4), bool) for m in loaded])
        sil = np.zeros((N, h4, w4), bool)
        cen = np.full((N, 2), np.nan)
        for f in range(N):
            uv, ok, _ = project(xf(o["T"][f], o["V"]), w2c[f], Kc[f])
            if ok.any():
                cen[f] = uv[ok].mean(0)
            sil[f] = raster_mask((uv + 0.5) / s - 0.5, o["F"], h4, w4, ok)
        log(f"mask check {o['name']}: masks loaded in {t1 - t0:.1f} s, silhouettes "
            f"{time.time() - t1:.1f} s")

        def IoU(a, b):
            u = (a | b).sum()
            return (a & b).sum() / u if u else np.nan

        iou[k] = np.array([IoU(sil[f], msk[f]) if have[f] else np.nan for f in range(N)])
        spd = np.full(N, 0.0)
        spd[1:-1] = np.linalg.norm(cen[2:] - cen[:-2], axis=1) / 2
        spd = np.nan_to_num(spd)
        order = np.argsort(-spd)
        picked = []
        for f in order:
            if spd[f] < 3.0 or len(picked) >= 8:
                break
            if 3 <= f < N - 3 and all(abs(f - g) >= 4 for g in picked):
                picked.append(int(f))
        res = dict(object=o["name"], mask_dir=d, moving_frames=sorted(picked),
                   speed_px_per_frame=[round(float(spd[f]), 1) for f in sorted(picked)])
        if len(picked) >= 2:
            tab = {}
            for off in range(-3, 4):
                v = [IoU(sil[f], msk[f + off]) for f in picked if have[f + off]]
                tab[off] = float(np.nanmean(v)) if v else float("nan")
            best = max(tab, key=lambda x: -1 if not np.isfinite(tab[x]) else tab[x])
            res["mean_iou_by_offset"] = {str(k2): round(v2, 4) for k2, v2 in tab.items()}
            res["best_offset"] = int(best)
            res["verdict"] = (f"best offset {best:+d} (IoU {tab[best]:.3f}; offset 0 "
                              f"{tab[0]:.3f}, -1 {tab[-1]:.3f}, +1 {tab[1]:.3f}) on "
                              f"{len(picked)} moving frames")
        else:
            res["verdict"] = "object barely moves in the image: index alignment not testable"
        res["mean_iou_all_frames"] = round(float(np.nanmean(iou[k])), 4)
        align.append(res)
        log(f"mask check {o['name']}: {res['verdict']}; mean IoU {res['mean_iou_all_frames']}")
    return (iou if iou else None), align


# ----------------------------------------------------------------------------- side view setup
def choose_side(ctx):
    """View A: horizontal view direction that keeps the objects apart (camera ray >= 30 deg out of
    the view direction); view B: camera ray fully in the picture plane (omitted if A is within
    15 deg of it)."""
    objs, keys, c2w = ctx["objs"], ctx["keys"], ctx["c2w"]
    fwd = c2w[0][:3, 2]
    fh = fwd[:2] / max(np.linalg.norm(fwd[:2]), 1e-9)
    ez = np.array([0.0, 0.0, 1.0])
    frames = sorted(set(keys) | {0})
    pts = {f: [xf(o["T"][f], o["V"][:: max(1, len(o["V"]) // 1500)]) for o in objs]
           for f in frames}
    best = None
    phis = [ctx["args"].side_azimuth] if ctx["args"].side_azimuth is not None else range(0, 360, 5)
    for phi in phis:
        d = np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi)), 0.0])
        eu = np.cross(d, ez)
        sc = abs(d[0] * fh[1] - d[1] * fh[0])
        if ctx["args"].side_azimuth is None:
            if sc < 0.5 or np.dot(fh, eu[:2]) < 0:
                continue
        gaps = []
        for f in frames:
            iv = [(p @ eu).min() for p in pts[f]], [(p @ eu).max() for p in pts[f]]
            g = 0.05
            for a in range(len(objs)):
                for b in range(a + 1, len(objs)):
                    g = min(g, max(iv[0][b] - iv[1][a], iv[0][a] - iv[1][b]))
            gaps.append(max(g, -0.2))
        score = float(np.mean(gaps)) + 0.01 * sc
        if best is None or score > best[0]:
            best = (score, phi, d, eu, sc)
    _, phi, d, eu, sc = best
    A = _view(d, pts, ctx["z_table"], fwd, fh, "A: azimuth chosen so the objects do not overlap")
    dB = np.array([-fh[1], fh[0], 0.0])           # camera ray fully in the picture plane
    B = _view(dB, pts, ctx["z_table"], fwd, fh, "B: camera ray in the picture plane")
    if A["cam_angle_deg"] >= 75:
        B = None
    return A, B


def _view(d, pts, z_table, fwd, fh, name):
    ez = np.array([0.0, 0.0, 1.0])
    eu = np.cross(d, ez)
    sc = abs(d[0] * fh[1] - d[1] * fh[0])
    allp = np.concatenate([np.concatenate(v) for v in pts.values()])
    u = allp @ eu
    v = (allp[:, 2] - z_table) * 100
    uc = float(np.median(np.concatenate(pts[min(pts)]) @ eu))
    uu = (u - uc) * 100
    u0, u1 = uu.min() - 4, uu.max() + 4
    v0, v1 = min(-4.0, v.min() - 2), v.max() + 4
    if u1 - u0 < 24:
        m = (u0 + u1) / 2
        u0, u1 = m - 12, m + 12
    aspect = 0.62
    if (v1 - v0) < aspect * (u1 - u0):
        v1 = v0 + aspect * (u1 - u0)
    elif (v1 - v0) > 1.1 * (u1 - u0):
        m = (u0 + u1) / 2
        half = (v1 - v0) / 1.1 / 2
        u0, u1 = m - half, m + half
    return dict(name=name, phi_deg=float(np.degrees(np.arctan2(d[1], d[0])) % 360), d=d, eu=eu,
                uc=uc, cam_angle_deg=float(np.degrees(np.arcsin(np.clip(sc, 0, 1)))),
                bounds=(u0, u1, v0, v1), fwd=fwd)


def side_coords(P, side, z_table):
    return np.stack([(P @ side["eu"] - side["uc"]) * 100, (P[:, 2] - z_table) * 100], 1)


def draw_side(ax, ctx, f, side, bounds, res, focus=None, fade_others=False, show_hands=True,
              show_roll=True, text=True):
    objs, z_table = ctx["objs"], ctx["z_table"]
    u0, u1, v0, v1 = bounds
    Wp, Hp = int(round((u1 - u0) * res)), int(round((v1 - v0) * res))
    xs = u0 + (np.arange(Wp) + 0.5) / res
    ys = v1 - (np.arange(Hp) + 0.5) / res
    near = None
    if focus is not None:   # close-up: only things whose xy footprint is near the focus object
        Pf = xf(objs[focus]["T"][f], objs[focus]["V"])
        flo, fhi = Pf[:, :2].min(0), Pf[:, :2].max(0)

        def near(P, m):
            lo, hi = P[:, :2].min(0), P[:, :2].max(0)
            return bool((lo <= fhi + m).all() and (hi >= flo - m).all())
    for s in ctx["supports"]:
        if near is not None and not near(s["pts"], 0.03):
            continue
        q = side_coords(s["pts"], side, z_table)
        try:
            h = ConvexHull(q)
            ax.add_patch(MplPolygon(q[h.vertices], closed=True, facecolor="#c9c7c0",
                                    edgecolor="#6e6c66", lw=0.8, zorder=1))
        except Exception:
            pass
    ax.axhline(0, color="#52514e", lw=0.8, ls="--", zorder=1.5)
    order = sorted(range(len(objs)), key=lambda k: -float(objs[k]["centroid"][f] @ side["d"]))
    rgba = np.zeros((Hp, Wp, 4))
    masks = {}
    for k in order:
        o = objs[k]
        Pk = xf(o["T"][f], o["V"])
        if near is not None and k != focus and not near(Pk, 0.03):
            continue
        q = side_coords(Pk, side, z_table)
        xy = np.stack([(q[:, 0] - u0) * res - 0.5, (v1 - q[:, 1]) * res - 0.5], 1)
        m = raster_mask(xy, o["F"], Hp, Wp)
        col = o["color"] if (focus is None or k == focus or not fade_others) else "#b5b3ad"
        rgba = over_rgba(rgba, m, col, 0.55)
        masks[k] = (m, col)
    ax.imshow((np.clip(rgba, 0, 1) * 255).astype(np.uint8), extent=(u0, u1, v0, v1),
              origin="upper", zorder=2, interpolation="antialiased")
    for k, (m, col) in masks.items():
        contour(ax, m, col, lw=1.2, x=xs, y=ys, z=3)
    roll = ctx["roll"]
    if show_roll and roll is not None:
        for k, o in enumerate(objs):
            if focus is not None and k != focus:
                continue
            T = roll["ach"][k][f]
            if not np.isfinite(T[0, 0]):
                continue
            q = side_coords(xf(T, o["V"]), side, z_table)
            xy = np.stack([(q[:, 0] - u0) * res - 0.5, (v1 - q[:, 1]) * res - 0.5], 1)
            contour(ax, raster_mask(xy, o["F"], Hp, Wp), darker(o["rcolor"]), lw=1.4, ls="--",
                    x=xs, y=ys, z=4)
    if show_hands:
        for s in SIDES:
            Jh = ctx["P"]["joints"].get(s)
            if Jh is None or not ctx["hv"][s][f]:
                continue
            if near is not None and not near(Jh[f], 0.08):
                continue
            q = side_coords(Jh[f], side, z_table)
            draw_skel(ax, q, HAND_PLOT[s], ls=":" if ctx["refilled"][s][f] else "-", lw=1.4, ms=5)
    ax.set_xlim(u0, u1)
    ax.set_ylim(v0, v1)
    ax.set_aspect("equal")
    ax.grid(True, color="#e4e2dc", lw=0.5, zorder=0)
    ax.tick_params(labelsize=7, length=2, pad=1)


def darker(c):
    rgb = np.array(matplotlib.colors.to_rgb(c))
    return tuple(rgb * 0.7)


def height_lines(ctx, f, compact=False):
    """Bottom height per object: vs the support prim beneath it, and vs the table plane."""
    lines = []
    for k, o in enumerate(ctx["objs"]):
        hs = o["h_support"][f]
        ht = o["h_table"][f]
        if np.isfinite(hs) and abs(hs - ht) > 0.005:
            s = f"{o['name']}: {hs * 100:+.1f} cm vs disk below ({ht * 100:+.1f} vs table)"
        elif np.isfinite(hs):
            s = f"{o['name']}: {hs * 100:+.1f} cm"
        else:
            s = f"{o['name']}: {ht * 100:+.1f} cm vs table (no disk below)"
        roll = ctx["roll"]
        if roll is not None and np.isfinite(roll["bottom"][k][f]):
            hr = roll["bottom"][k][f] - (o["sup_top"][f] if np.isfinite(hs) else ctx["z_table"])
            s += f" | {ctx['args'].rollout_label} {hr * 100:+.1f}"
        lines.append(s)
    return lines


def hand_marks(ctx, f):
    out = [] if ctx["cam_valid"][f] else ["camera is_valid False"]
    for s in SIDES:
        if not ctx["hv"][s][f]:
            out.append(f"{s[0].upper()} invalid (not drawn)")
        elif ctx["refilled"][s][f]:
            out.append(f"{s[0].upper()} filled (dotted)")
    return out


def legend_handles(ctx, img=True):
    hs = []
    for o in ctx["objs"]:
        hs.append(Patch(facecolor=o["color"], edgecolor=o["color"], alpha=0.6, label=o["name"]))
    hc = HAND_IMG if img else HAND_PLOT
    hs.append(Line2D([], [], color=hc["left"], lw=2, label="left hand (MANO joints)"))
    hs.append(Line2D([], [], color=hc["right"], lw=2, label="right hand"))
    if any(ctx["refilled"][s].any() for s in SIDES):
        hs.append(Line2D([], [], color="#aaaaaa", lw=2, ls=":",
                         label="hand filled (not observed)"))
    if ctx["roll"] is not None:
        for o in ctx["objs"]:
            hs.append(Line2D([], [], color=o["rcolor"] if img else darker(o["rcolor"]), lw=2,
                             ls="--", label=f"{ctx['args'].rollout_label}: {o['name']} achieved"))
    return hs


def grid_rects(n, ncol, cell_w, cell_h, title_h, top, fig_w, fig_h, pad=4, lpad=None):
    lpad = pad if lpad is None else lpad
    rects = []
    for i in range(n):
        r, c = divmod(i, ncol)
        x0 = c * cell_w + lpad
        ytop = top + r * (cell_h + title_h) + title_h
        rects.append([x0 / fig_w, 1 - (ytop + cell_h) / fig_h, (cell_w - lpad - pad) / fig_w,
                      cell_h / fig_h])
    return rects


def header(fig, ctx, what, fig_h):
    a = ctx["args"]
    t = (f"{a.label} | episode {a.episode} | {what} | {ctx['N']} frames @ {ctx['fps']:.2f} fps | "
         f"scene {ctx['args'].scene}")
    fig.text(0.005, 1 - 6 / fig_h, t, fontsize=10, va="top", ha="left", family="monospace")


def save(fig, path):
    fig.savefig(path, dpi=100, pil_kwargs={"quality": 90})
    plt.close(fig)


# ----------------------------------------------------------------------------- overlay
def render_overlay(path, ctx):
    keys, objs = ctx["keys"], ctx["objs"]
    W, H = ctx["W"], ctx["H"]
    ncol = max(1, min(ctx["args"].ncol, len(keys)))
    nrow = int(math.ceil(len(keys) / ncol))
    fig_w = ctx["args"].max_width
    cell_w = fig_w // ncol
    cell_h = int(round((cell_w - 8) * H / W))
    title_h, top = 20, 64
    fig_h = top + nrow * (cell_h + title_h) + 6
    fig = plt.figure(figsize=(fig_w / 100, fig_h / 100), dpi=100)
    header(fig, ctx, "overlay: mesh silhouettes + MANO skeletons on the RGB frames", fig_h)
    lg = fig.legend(handles=legend_handles(ctx, img=True), loc="upper left",
                    bbox_to_anchor=(0.0, 1 - 22 / fig_h), ncol=8, fontsize=8.5, frameon=True,
                    facecolor="#3a3a38", edgecolor="#3a3a38", labelcolor="white",
                    handlelength=2.2, columnspacing=1.2)
    lg.get_frame().set_alpha(1.0)
    pw = min(W, 2 * cell_w)
    s = pw / W
    ph = int(round(H * s))
    for ax_rect, f in zip(grid_rects(len(keys), ncol, cell_w, cell_h, title_h, top, fig_w, fig_h),
                          keys):
        ax = fig.add_axes(ax_rect)
        img = (np.asarray(Image.fromarray(ctx["frames"][f]).resize((pw, ph), Image.BILINEAR),
                          np.float32) / 255.0 if f in ctx["frames"]
               else np.zeros((ph, pw, 3), np.float32))
        ms = []
        for o in objs:
            uv, ok, _ = project(xf(o["T"][f], o["V"]), ctx["w2c"][f], ctx["K"][f])
            m = raster_mask((uv + 0.5) * s - 0.5, o["F"], ph, pw, ok)
            img = blend(img, m, o["color"], 0.35)
            ms.append(m)
        ax.imshow((np.clip(img, 0, 1) * 255).astype(np.uint8), interpolation="antialiased")
        for o, m in zip(objs, ms):
            contour(ax, m, o["color"], lw=1.5)
        roll = ctx["roll"]
        if roll is not None:
            for k, o in enumerate(objs):
                T = roll["ach"][k][f]
                if not np.isfinite(T[0, 0]):
                    continue
                uv, ok, _ = project(xf(T, o["V"]), ctx["w2c"][f], ctx["K"][f])
                contour(ax, raster_mask((uv + 0.5) * s - 0.5, o["F"], ph, pw, ok), o["rcolor"],
                        lw=1.6, ls="--")
        for sd in SIDES:
            Jh = ctx["P"]["joints"].get(sd)
            if Jh is None or not ctx["hv"][sd][f]:
                continue
            uv, ok, _ = project(Jh[f], ctx["w2c"][f], ctx["K"][f])
            draw_skel(ax, (uv + 0.5) * s - 0.5, HAND_IMG[sd],
                      ls=":" if ctx["refilled"][sd][f] else "-", lw=1.6, ms=7, ok=ok)
        ax.set_xlim(-0.5, pw - 0.5)
        ax.set_ylim(ph - 0.5, -0.5)
        ax.axis("off")
        rs = ", ".join(ctx["reasons"].get(f, []))
        ax.set_title(f"f{f}  t={f / ctx['fps']:.2f}s  [{rs}]"[:max(30, cell_w // 7)], fontsize=9,
                     pad=3, loc="left")
        ax.text(0.01, 0.99, "\n".join(height_lines(ctx, f)), transform=ax.transAxes, fontsize=8,
                color="white", va="top", ha="left", bbox=TXT_BOX, family="monospace")
        hm = hand_marks(ctx, f)
        if hm:
            ax.text(0.99, 0.01, "\n".join(hm), transform=ax.transAxes, fontsize=8.5,
                    color="#ffdd55", va="bottom", ha="right", bbox=TXT_BOX)
    save(fig, path)


# ----------------------------------------------------------------------------- side
def render_side(path, ctx):
    keys = ctx["keys"]
    views = [v for v in (ctx["side"], ctx.get("sideB")) if v is not None]
    ncol = max(1, min(ctx["args"].ncol, len(keys)))
    nrow = int(math.ceil(len(keys) / ncol))
    fig_w = ctx["args"].max_width
    cell_w = fig_w // ncol
    lpad, rpad, title_h, tick_h = 30, 6, 20, 16
    aw = cell_w - lpad - rpad
    ahs = [int(round(aw * (v["bounds"][3] - v["bounds"][2]) / (v["bounds"][1] - v["bounds"][0])))
           for v in views]
    top = 44 + 15 * len(views) + 26
    fig_h = top + nrow * sum(title_h + ah + tick_h for ah in ahs) + 8
    fig = plt.figure(figsize=(fig_w / 100, fig_h / 100), dpi=100)
    header(fig, ctx, "side views (orthographic, gravity up; cm; 0 = table plane)", fig_h)
    fwd = ctx["side"]["fwd"]
    below = np.degrees(np.arctan2(-fwd[2], np.linalg.norm(fwd[:2])))
    fig.text(0.005, 1 - 24 / fig_h,
             f"Camera looks {below:.0f} deg below horizontal.  Grey = support prims from the USD; "
             f"dashed line = table plane z={ctx['z_table']:.3f} (loader world).  Arrow = camera ray "
             "direction in the picture plane.  Rows alternate view A / view B for each row of key "
             "frames.", fontsize=8.5, va="top")
    for i, v in enumerate(views):
        fig.text(0.005, 1 - (40 + 15 * i) / fig_h,
                 f"view {v['name']}: azimuth {v['phi_deg']:.0f} deg; the camera ray is "
                 f"{v['cam_angle_deg']:.0f} deg out of the view direction (90 = fully in the "
                 "picture plane: depth errors along the ray show at full length)",
                 fontsize=8.5, va="top")
    fig.legend(handles=legend_handles(ctx, img=False), loc="upper left",
               bbox_to_anchor=(0.0, 1 - (44 + 15 * len(views)) / fig_h), ncol=8, fontsize=8.5,
               frameon=False)
    y = top
    for r in range(nrow):
        for vi, v in enumerate(views):
            u0, u1, _, _ = v["bounds"]
            res = max(2.0, aw / (u1 - u0) * 2)
            for c in range(ncol):
                idx = r * ncol + c
                if idx >= len(keys):
                    break
                f = keys[idx]
                ax = fig.add_axes([(c * cell_w + lpad) / fig_w, 1 - (y + title_h + ahs[vi]) / fig_h,
                                   aw / fig_w, ahs[vi] / fig_h])
                draw_side(ax, ctx, f, v, v["bounds"], res)
                rs = ", ".join(ctx["reasons"].get(f, []))
                ax.set_title(f"f{f} t={f / ctx['fps']:.2f}s view {v['name'][0]} [{rs}]"
                             [:max(30, cell_w // 7)], fontsize=9, pad=3, loc="left")
                ax.text(0.01, 0.99, "\n".join(height_lines(ctx, f)), transform=ax.transAxes,
                        fontsize=7.2, va="top", ha="left", family="monospace",
                        bbox=dict(facecolor="white", alpha=0.75, pad=1.2, edgecolor="none"))
                hm = hand_marks(ctx, f)
                if hm:
                    ax.text(0.99, 0.01, "  ".join(hm), transform=ax.transAxes, fontsize=8,
                            color="#b00000", va="bottom", ha="right")
                a = np.array([fwd @ v["eu"], fwd[2]])
                a = a / max(np.linalg.norm(a), 1e-9) * 0.07
                ax.annotate("", xy=(0.92 + a[0], 0.80 + a[1] * (u1 - u0) / max(
                    v["bounds"][3] - v["bounds"][2], 1e-6)), xytext=(0.92, 0.80),
                    xycoords="axes fraction", arrowprops=dict(arrowstyle="->", color="#333333",
                                                              lw=1.3))
                ax.text(0.92, 0.84, "cam ray", transform=ax.transAxes, fontsize=7, ha="center")
            y += title_h + ahs[vi] + tick_h
    save(fig, path)


# ----------------------------------------------------------------------------- frame 0 close-ups
def render_frame0(path, ctx):
    objs = ctx["objs"]
    f = 0
    fwd = ctx["side"]["fwd"]
    fh = fwd[:2] / max(np.linalg.norm(fwd[:2]), 1e-9)
    ez = np.array([0.0, 0.0, 1.0])
    views = []
    for nm, d in (("view across the camera ray (ray in picture plane)",
                   np.array([-fh[1], fh[0], 0.0])),
                  ("view along the camera ray (image left-right)", np.array([fh[0], fh[1], 0.0]))):
        eu = np.cross(d, ez)
        if nm.startswith("view across") and np.dot(fh, eu[:2]) < 0:
            d, eu = -d, -eu
        views.append((nm, d, eu))
    fig_w = ctx["args"].max_width
    cell = fig_w // 3
    top, title_h = 40, 22
    nrow = len(objs)
    fig_h = top + nrow * (cell + title_h) + 6
    fig = plt.figure(figsize=(fig_w / 100, fig_h / 100), dpi=100)
    header(fig, ctx, "frame 0 close-ups (crop: outline + light fill; side: 0 = table plane, cm)",
           fig_h)
    rects = grid_rects(nrow * 3, 3, cell, cell - 16, title_h + 16, top - 16, fig_w, fig_h, pad=8,
                       lpad=34)
    img0 = ctx["frames"].get(0)
    W, H = ctx["W"], ctx["H"]
    for k, o in enumerate(objs):
        ax = fig.add_axes(rects[3 * k])
        uv, ok, _ = project(xf(o["T"][f], o["V"]), ctx["w2c"][f], ctx["K"][f])
        if img0 is not None and ok.any():
            lo, hi = uv[ok].min(0), uv[ok].max(0)
            c = (lo + hi) / 2
            half = max(hi - lo) * 0.8 + 30
            x0, x1 = int(max(0, c[0] - half)), int(min(W, c[0] + half))
            y0, y1 = int(max(0, c[1] - half)), int(min(H, c[1] + half))
            if x1 - x0 > 4 and y1 - y0 > 4:
                crop = img0[y0:y1, x0:x1].astype(np.float32) / 255.0
                cms = []
                for kk, oo in enumerate(objs):
                    uv2, ok2, _ = project(xf(oo["T"][f], oo["V"]), ctx["w2c"][f], ctx["K"][f])
                    cms.append(raster_mask(uv2 - [x0, y0], oo["F"], y1 - y0, x1 - x0, ok2))
                crop = blend(crop, cms[k], o["color"], 0.18)
                ax.imshow((np.clip(crop, 0, 1) * 255).astype(np.uint8))
                for kk, oo in enumerate(objs):
                    contour(ax, cms[kk], oo["color"], lw=1.6 if kk == k else 1.0)
                for sd in SIDES:
                    Jh = ctx["P"]["joints"].get(sd)
                    if Jh is None or not ctx["hv"][sd][f]:
                        continue
                    uvj, okj, _ = project(Jh[f], ctx["w2c"][f], ctx["K"][f])
                    draw_skel(ax, uvj - [x0, y0], HAND_IMG[sd], ok=okj, lw=1.4, ms=6)
                ax.set_xlim(-0.5, x1 - x0 - 0.5)
                ax.set_ylim(y1 - y0 - 0.5, -0.5)
        ax.axis("off")
        ax.set_title(f"{o['name']}: frame 0 RGB crop (full-res)", fontsize=9.5, loc="left", pad=3)
        V0 = xf(o["T"][f], o["V"])
        for j, (nm, d, eu) in enumerate(views):
            ax = fig.add_axes(rects[3 * k + 1 + j])
            sd_ = dict(d=d, eu=eu, uc=float(np.median(V0 @ eu)))
            q = side_coords(V0, sd_, ctx["z_table"])
            umin, umax = q[:, 0].min() - 3, q[:, 0].max() + 3
            vmin, vmax = min(-3.0, q[:, 1].min() - 2), q[:, 1].max() + 2
            span = max(umax - umin, vmax - vmin)
            uc_, vc_ = (umin + umax) / 2, (vmin + vmax) / 2
            b = (uc_ - span / 2, uc_ + span / 2, vc_ - span / 2, vc_ + span / 2)
            res = max(4.0, (cell - 20) / span * 2)
            draw_side(ax, ctx, f, sd_, b, res, focus=k, fade_others=True, show_roll=False)
            bot = o["bottom"][f]
            ib = int(np.argmin(V0[:, 2]))
            qb = q[ib]
            ax.plot([qb[0]], [qb[1]], marker="v", color="black", ms=6, zorder=7)
            hs, ht = o["h_support"][f], o["h_table"][f]
            txt = (f"bottom vs support beneath ({o['sup_name'][f]}): {hs * 1000:+.1f} mm"
                   if np.isfinite(hs) else "NO support prim beneath the object footprint")
            txt += f"\nbottom vs table plane: {ht * 1000:+.1f} mm   (bottom z={bot:.4f})"
            ax.text(0.01, 0.99, txt, transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(facecolor="white", alpha=0.85, pad=1.5, edgecolor="none"))
            ax.set_title(f"{o['name']}: {nm}", fontsize=9.5, loc="left", pad=3)
    save(fig, path)


# ----------------------------------------------------------------------------- plots
def render_plots(path, ctx, contact_cnt, hand_dist, iou):
    objs, N, keys = ctx["objs"], ctx["N"], ctx["keys"]
    roll = ctx["roll"]
    x = np.arange(N)
    rows = [("height", 2.3), ("rot", 2.1), ("twc", 1.6), ("pos", 1.5)]
    for s in SIDES:
        rows += [(f"dist_{s}", 2.0), (f"con_{s}", 0.35 + 0.22 * len(objs))]
    if iou is not None:
        rows.append(("iou", 1.6))
    if roll is not None:
        rows.append(("add", 2.0))
    fig_w = ctx["args"].max_width / 100
    head = 0.75
    fig_h = head + sum(r[1] for r in rows) + 0.35 * len(rows) + 0.6
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    header(fig, ctx, "time series (loader world, z up)", fig_h * 100)
    gs = fig.add_gridspec(len(rows), 1, height_ratios=[r[1] for r in rows], hspace=0.55,
                          left=0.045, right=0.86, top=1 - head / fig_h, bottom=0.55 / fig_h)
    axes = {}
    first = None
    for i, (nm, _) in enumerate(rows):
        ax = fig.add_subplot(gs[i], sharex=first)
        first = first or ax
        axes[nm] = ax
        for kf in keys:
            ax.axvline(kf, color="#bdbbb5", lw=0.6, ls=":", zorder=0)
        ax.grid(True, axis="y", color="#ecebe7", lw=0.6)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, N - 1)
    for kf in keys:
        axes["height"].text(kf, 1.01, str(kf), transform=axes["height"].get_xaxis_transform(),
                            fontsize=7, ha="center", va="bottom", color="#52514e")

    def leg(ax):
        ax.legend(loc="upper left", bbox_to_anchor=(1.003, 1.0), fontsize=8, frameon=False)

    ax = axes["height"]
    for k, o in enumerate(objs):
        ax.plot(x, o["h_support"] * 100, color=o["color"], lw=1.8, label=f"{o['name']} vs support beneath")
        ax.plot(x, o["h_table"] * 100, color=o["color"], lw=1.0, ls=":",
                label=f"{o['name']} vs table plane")
        if roll is not None:
            ax.plot(x, (roll["bottom"][k] - ctx["z_table"]) * 100, color=darker(o["rcolor"]),
                    lw=1.2, ls="--", label=f"{ctx['args'].rollout_label} {o['name']} vs table")
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_ylabel("bottom height (cm)", fontsize=8.5)
    ax.set_title("object bottom height above the support beneath it (solid) / above the table "
                 "plane (dotted); negative = sunk.  Numbers above = key frames", fontsize=9,
                 loc="left", pad=13)
    leg(ax)

    ax = axes["rot"]
    mx = 0.0
    for o in objs:
        ax.plot(x, o["rot_step"], color=o["color"], lw=1.5, label=f"{o['name']} |step|")
        ax.plot(x, o["twist"], color=o["color"], lw=1.0, ls="--",
                label=f"{o['name']} twist ({o['axis_kind']} axis)")
        mx = max(mx, float(np.nanmax(np.abs(o["rot_step"]))), float(np.nanmax(np.abs(o["twist"]))))
    ax.axhline(0, color="#52514e", lw=0.6)
    if mx > 20:
        ax.set_yscale("symlog", linthresh=5, linscale=1.5)
        tk = [t for t in (-180, -90, -45, -20, -10, -5, 0, 5, 10, 20, 45, 90, 180)
              if abs(t) <= max(mx * 1.05, 5)]
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(tk))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        scale_note = "symlog: linear within +-5 deg"
    else:
        scale_note = "linear"
    ax.set_ylabel("deg / frame", fontsize=8.5)
    ax.set_title("rotation step per frame (solid) and signed twist about the mesh symmetry axis "
                 f"(dashed); {scale_note}; max {mx:.1f} deg/frame", fontsize=9, loc="left")
    leg(ax)

    ax = axes["twc"]
    for o in objs:
        ax.plot(x, o["twist_cum"], color=o["color"], lw=1.5, label=o["name"])
    ax.axhline(0, color="#52514e", lw=0.6)
    ax.set_ylabel("deg", fontsize=8.5)
    ax.set_title("cumulative twist about the symmetry axis (spin of the track)", fontsize=9,
                 loc="left")
    leg(ax)

    ax = axes["pos"]
    for o in objs:
        ax.plot(x, o["pos_step"] * 100, color=o["color"], lw=1.4, label=o["name"])
    ax.set_ylabel("cm / frame", fontsize=8.5)
    ax.set_title("mesh-centroid step per frame", fontsize=9, loc="left")
    leg(ax)

    for s in SIDES:
        ax = axes[f"dist_{s}"]
        inv = ~ctx["hv"][s]
        rf = ctx["refilled"][s]
        for msk, kw in ((inv, dict(color="#9b9a95", alpha=0.35, label="is_valid False")),
                        (rf, dict(facecolor="none", edgecolor="#9b9a95", hatch="///", lw=0,
                                  label="filled: not observed"))):
            if msk.any():
                runs = np.flatnonzero(np.diff(np.r_[0, msk.astype(int), 0]))
                for a_, b_ in zip(runs[::2], runs[1::2]):
                    ax.axvspan(a_ - 0.5, b_ - 0.5, zorder=0, **kw)
                    kw.pop("label", None)
        for k, o in enumerate(objs):
            if k in hand_dist[s]:
                ax.plot(x, hand_dist[s][k] * 100, color=o["color"], lw=1.5, label=o["name"])
        ax.axhline(0, color="#52514e", lw=0.6)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("cm", fontsize=8.5)
        nobs = int(ctx["refilled"][s].sum())
        ax.set_title(f"{s} hand: nearest MANO joint to each object surface (grey = is_valid False "
                     f"in the bundle: {int((~ctx['hv'][s]).sum())} fr; hatched = valid but not "
                     f"observed / filled: {nobs} fr)", fontsize=9, loc="left", color=HAND_PLOT[s])
        leg(ax)
        ax = axes[f"con_{s}"]
        for k, o in enumerate(objs):
            cnt = contact_cnt.get(s, {}).get(k)
            if cnt is None:
                continue
            on = cnt > 0
            if on.any():
                runs = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
                for a_, b_ in zip(runs[::2], runs[1::2]):
                    ax.add_patch(matplotlib.patches.Rectangle((a_ - 0.5, k - 0.35), b_ - a_, 0.7,
                                                              color=o["color"], alpha=0.85))
            ax.text(N - 1, k, f" max {int(cnt.max())} links", fontsize=7, va="center",
                    ha="left", clip_on=False)
        ax.set_ylim(-0.6, len(objs) - 0.4)
        ax.set_yticks(range(len(objs)))
        ax.set_yticklabels([o["name"] for o in objs], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_title(f"{s} hand: contact labels from the parquet (>=1 of 16 MANO links labelled "
                     f"on the object)", fontsize=9, loc="left", color=HAND_PLOT[s])
        ax.grid(False)

    if iou is not None:
        ax = axes["iou"]
        for k, o in enumerate(objs):
            if k in iou:
                ax.plot(x, iou[k], color=o["color"], lw=1.4, label=o["name"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("IoU", fontsize=8.5)
        ax.set_title("projected silhouette vs our SAM3 object mask (visible-part mask: hand "
                     "occlusion lowers it; GT-free)", fontsize=9, loc="left")
        leg(ax)
    if roll is not None:
        ax = axes["add"]
        for k, o in enumerate(objs):
            ax.plot(x, roll["add"][k] * 100, color=o["color"], lw=1.6, label=f"{o['name']} ADD")
            ax.plot(x, roll["adds"][k] * 100, color=o["color"], lw=1.0, ls="--",
                    label=f"{o['name']} ADD-S")
        ax.set_ylim(bottom=0)
        ax.set_ylabel("cm", fontsize=8.5)
        ax.set_title(f"{ctx['args'].rollout_label}: achieved vs reference (ADD solid, ADD-S "
                     "dashed, mean over mesh points)", fontsize=9, loc="left")
        leg(ax)
    for nm, ax in axes.items():
        if nm != list(axes)[-1]:
            ax.tick_params(labelbottom=True)
    list(axes.values())[-1].set_xlabel(f"frame (fps {ctx['fps']:.2f}); dotted verticals = key "
                                       "frames", fontsize=8.5)
    save(fig, path)


if __name__ == "__main__":
    main()
