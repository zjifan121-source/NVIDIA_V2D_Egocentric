"""Segment solving (V4_BUILD_SPEC.md S3c): one object's track from S3b's segments, each solved by the rule
that holds there. Poses map the canonical mesh frame to the world (V4 §3).

  static    one pose: ICP of the canonical mesh onto the segment's fused surface (scale fixed), from
            the previous segment's end pose (the registration's for the first), and from that pose
            times each element of the object's symmetry group; the lowest median point-to-mesh
            residual wins (bar <= 5 mm)
  grasp     pass 1: one hand->object offset T for the grasping hand, object(t) = wrist(t) T, fitted
            over the segment on (a) the canonical mesh's rendered depth vs the stereo depth inside the
            mask, with its silhouette against the mask the other way (Clip.silhouette_residual), (b)
            the contact fingertips' signed distance to the mesh, (c) the end poses equal to
            the neighbouring static poses (weight W_END); pass 2: per frame, object(t) = wrist(t) T
            D(t), D refined on (a) + (b), its change per frame bounded by 5 mm / 3 degrees per component
            (and pulled towards none by a soft prior): the object may shift in the hand gradually
            (ROHIT's relaxation), not jump (a soft prior alone let a 6 cm jump through in one frame,
            the depth outweighs it, and the slip guard never saw it). Slip guard: (a) above
            SLIP_M for SLIP_FRAMES frames in a row splits the grasp there; that run becomes an unstable
            bridge and the parts are solved again. `failed`: (a) within 1 cm on fewer than half the
            frames (S4b then takes `keyframe` there)
  unstable  cubic Hermite position (zero end velocity) + slerp between the neighbouring end poses
            (consecutive unsolved segments as one bridge; at frame 0, s3a2_register's pose),
            then <= UNSTABLE_STEPS least-squares steps per frame on (a) where the mask covers >= 1 % of
            the image (observe.area_bar), held near the interpolation by a prior and bounded (5 cm,
            20 degrees), kept only when it explains the frame better
Order: segments in clip order (statics and grasps), then the grasps again with both neighbouring
poses as end terms, then the unstable bridges. Rendered depth: Open3D's RaycastingScene (CPU).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from . import scan as S
from .observe import area_bar

SIG_A = 0.01  # m: depth residual scale (soft-L1)
SIG_B = 0.005  # m: fingertip signed distance scale
SIG_C = 0.005  # m: end-pose scale
ROT_LEN = 0.05  # m: a radian of end-pose rotation counts as this much translation
W_END = 10.0  # spec (c)
MISS = 0.03  # m: residual of a mask pixel whose ray misses the mesh
STEP_M, STEP_RAD = 0.005, np.radians(3.0)  # spec: per-frame change of the in-hand correction
SLIP_M, SLIP_FRAMES = 0.015, 10  # spec slip guard
FAIL_M, FAIL_SHARE = 0.01, 0.5  # spec: grasp residual (a) median <= 1 cm on >= 50 % of its frames
STATIC_BAR_M = 0.005  # spec
TIE_M = 0.001  # default: static fits this close count as the same fit
UNSTABLE_STEPS = 5  # spec
UNSTABLE_PRIOR_M, UNSTABLE_PRIOR_RAD = 0.02, np.radians(10)  # default: how far (a) may pull a bridge frame
UNSTABLE_REACH_M, UNSTABLE_REACH_RAD = 0.05, np.radians(20)  # default: the refinement's bound (a bridge frame
# whose guess is far off otherwise wandered 0.6 m on the synthetic slip)
FIT_FRAMES, FIT_RAYS, RAYS = 60, 150, 300  # default: pass 1 subsample; rays per frame (pass 1, pass 2)
MIN_GRASP = 5
SIL_POINTS = 200  # default: canonical surface points projected for the silhouette term
SIL_MARGIN = 80  # px: the distance map's crop around the allowed mask
SIL_CAP = 80.0  # px: distances beyond the crop


def T_of(x: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rotation.from_rotvec(x[:3]).as_matrix(), x[3:]
    return T


def x_of(T: np.ndarray) -> np.ndarray:
    return np.concatenate([Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])


def pose_diff(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """6-vector: rotation of inv(B) A (rotvec, rad) and the translation difference (m)."""
    return np.concatenate([Rotation.from_matrix(B[:3, :3].T @ A[:3, :3]).as_rotvec(), A[:3, 3] - B[:3, 3]])


@dataclass
class Clip:
    """One object's inputs. obs[t] = (uv (M, 2) at depth resolution, observed depth (M,), mask share)
    or None; tips (N, 2, 5, 3), wrist (N, 2, 4, 4), valid (N, 2) as pathb/hands.py writes them;
    target(a, b) -> the static segment's fused surface (open3d point cloud with normals, world) or
    None; group: the symmetry group (k, 4, 4) in the object frame, identity first."""
    K: np.ndarray
    c2w: np.ndarray
    obs: list
    tips: np.ndarray
    wrist: np.ndarray
    valid: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    group: np.ndarray
    target: object
    silhouette: list | None = None  # per frame: silhouette_map(...) of the object's and hands' masks, or None
    seed: int = 0
    scene: o3d.t.geometry.RaycastingScene = field(init=False)
    surface: o3d.geometry.PointCloud = field(init=False)

    def __post_init__(self):
        mesh = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(np.asarray(self.vertices, np.float32)),
                                           o3d.core.Tensor(np.asarray(self.faces, np.int32)))
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(mesh)
        legacy = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(self.vertices),
                                           o3d.utility.Vector3iVector(np.asarray(self.faces, np.int32)))
        legacy.compute_vertex_normals()
        self.surface = S.mesh_points(legacy, 20000)
        self.closed = bool(legacy.is_watertight()) or _trimesh_watertight(self.vertices, self.faces)
        self.rng = np.random.default_rng(self.seed)
        pts = np.asarray(self.surface.points)
        self.samples = pts[np.random.default_rng(self.seed).choice(len(pts), min(SIL_POINTS, len(pts)), replace=False)]
        self.min_share = area_bar([o[2] if o is not None else 0 for o in self.obs])  # an unstable frame's mask bar

    @property
    def n(self) -> int:
        return len(self.obs)

    def rays(self, t: int, limit: int):
        o = self.obs[t]
        if o is None:
            return None
        uv, d, share = o
        if len(d) > limit:
            i = np.random.default_rng(t).choice(len(d), limit, replace=False)
            uv, d = uv[i], d[i]
        K = self.K
        dirs = np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(d))], axis=1)
        return dirs, d, share

    def depth_residual(self, pose: np.ndarray, t: int, limit: int = RAYS) -> np.ndarray:
        """(M,) rendered minus observed depth (m) on frame t's mask pixels; MISS where a ray misses."""
        r = self.rays(t, limit)
        if r is None:
            return np.zeros(0)
        dirs, d, _ = r
        c2w = self.c2w[t]
        R_o, t_o = pose[:3, :3], pose[:3, 3]
        origin = R_o.T @ (c2w[:3, 3] - t_o)
        direction = (dirs @ c2w[:3, :3].T) @ R_o  # world, then into the object frame (R_o^T x)
        rays = np.concatenate([np.repeat(origin[None], len(d), 0), direction], axis=1).astype(np.float32)
        hit = self.scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        return np.where(np.isfinite(hit), hit - d, MISS)

    def silhouette_residual(self, pose: np.ndarray, t: int) -> np.ndarray:
        """(M,) metres: how far outside the allowed mask (the object's and the hands') each projected
        canonical surface point falls (0 inside), at its depth. The depth term (a) looks only inside
        the observed mask, so an estimate that overhangs it, or slides along a flat face, costs
        nothing there; this term is the other direction."""
        if self.silhouette is None or self.silhouette[t] is None:
            return np.zeros(0)
        (x0, y0), dist = self.silhouette[t]
        world = self.samples @ pose[:3, :3].T + pose[:3, 3]
        cam = (world - self.c2w[t, :3, 3]) @ self.c2w[t, :3, :3]
        z = np.maximum(cam[:, 2], 0.05)  # a point behind the camera counts as far outside (fixed length)
        u = self.K[0, 0] * cam[:, 0] / z + self.K[0, 2] - x0
        v = self.K[1, 1] * cam[:, 1] / z + self.K[1, 2] - y0
        inside = (u >= 0) & (v >= 0) & (u < dist.shape[1] - 1) & (v < dist.shape[0] - 1) & (cam[:, 2] > 0.05)
        px = np.full(len(u), SIL_CAP)
        px[inside] = dist[v[inside].astype(int), u[inside].astype(int)]
        return px * z / self.K[0, 0]

    def appearance(self, pose: np.ndarray, t: int, limit: int = RAYS) -> np.ndarray:
        """Residual (a), both ways: rendered vs observed depth inside the mask, and the silhouette."""
        return np.concatenate([self.depth_residual(pose, t, limit), self.silhouette_residual(pose, t)])

    def tip_distance(self, pose: np.ndarray, t: int, hand: int, fingers: list[int]) -> np.ndarray:
        if not fingers or not self.valid[t, hand]:
            return np.zeros(0)
        p = self.tips[t, hand, fingers]
        local = (p - pose[:3, 3]) @ pose[:3, :3]
        q = o3d.core.Tensor(local.astype(np.float32))
        sd = self.scene.compute_signed_distance(q) if self.closed else self.scene.compute_distance(q)
        return sd.numpy().astype(np.float64)


def silhouette_map(allowed: np.ndarray) -> tuple[tuple[int, int], np.ndarray] | None:
    """The crop offset and the pixel distance to the allowed mask, around it (SIL_MARGIN px)."""
    from scipy.ndimage import distance_transform_edt
    ys, xs = np.nonzero(allowed)
    if len(ys) == 0:
        return None
    y0, x0 = max(0, ys.min() - SIL_MARGIN), max(0, xs.min() - SIL_MARGIN)
    y1, x1 = min(allowed.shape[0], ys.max() + SIL_MARGIN + 1), min(allowed.shape[1], xs.max() + SIL_MARGIN + 1)
    crop = allowed[y0:y1, x0:x1]
    return (int(x0), int(y0)), np.minimum(distance_transform_edt(~crop), SIL_CAP).astype(np.float32)


def _trimesh_watertight(v, f) -> bool:
    import trimesh
    return bool(trimesh.Trimesh(v, f, process=False).is_watertight)


# ---------------------------------------------------------------- static
def solve_static(clip: Clip, a: int, b: int, init: np.ndarray) -> dict:
    """One pose over [a, b] by ICP onto the segment's fused surface, best over the symmetry group."""
    target = clip.target(a, b)
    if target is None or len(target.points) < 50:
        return {"pose": init, "residual_m": None, "candidates": 0, "why": "no fused surface: the start pose kept"}
    pts = np.asarray(target.points)
    centred = np.eye(4)  # the start pose's rotation, the mesh centred on the fused surface
    centred[:3, :3] = init[:3, :3]
    centred[:3, 3] = pts.mean(axis=0) - init[:3, :3] @ np.asarray(clip.surface.points).mean(axis=0)

    def fit(starts, label):
        found = []
        for k, G in enumerate(clip.group):
            for name, s0 in starts:
                T = S.icp(clip.surface, target, s0 @ G, max_dist=0.02, iterations=60, point_to_point=True)["T"]
                T = S.icp(clip.surface, target, T, max_dist=0.005, iterations=60)["T"]
                local = (pts - T[:3, 3]) @ T[:3, :3]
                res = float(np.median(np.abs(clip.scene.compute_distance(o3d.core.Tensor(local.astype(np.float32))).numpy())))
                found.append({"pose": T, "residual_m": res, "symmetry_index": k, "start_from": name or label})
        # a shape that looks the same turned (a box's half turn) fits as well both ways: among fits within
        # TIE_M of the best, the one turned least from the start pose
        floor = min(f["residual_m"] for f in found)
        tied = [f for f in found if f["residual_m"] <= floor + TIE_M]
        return min(tied, key=lambda f: Rotation.from_matrix(f["pose"][:3, :3] @ init[:3, :3].T).magnitude())
    best = fit([("previous pose", init), ("centred on the fused surface", centred)], "")
    if best["residual_m"] > STATIC_BAR_M:  # the start was too far (a bad grasp before it): global starts too
        alt = fit([("", S.global_start(clip.surface, target, seed=seed)) for seed in range(3)], "global (FPFH + RANSAC)")
        best = alt if alt["residual_m"] < best["residual_m"] else best
    return best | {"candidates": len(clip.group), "points": len(pts)}


# ---------------------------------------------------------------- grasp
def end_terms(pose_a: np.ndarray, pose_b: np.ndarray, start: np.ndarray | None, end: np.ndarray | None) -> list:
    out = []
    for P, ref in ((pose_a, start), (pose_b, end)):
        if ref is not None:
            d = pose_diff(P, ref)
            out.append(W_END * np.concatenate([d[:3] * ROT_LEN, d[3:]]) / SIG_C)
    return out


def pass1(clip: Clip, a: int, b: int, hand: int, fingers: list[int], start, end, T0: np.ndarray) -> dict:
    frames = [t for t in range(a, b + 1) if clip.valid[t, hand]]
    fit = frames if len(frames) <= FIT_FRAMES else [frames[i] for i in np.linspace(0, len(frames) - 1, FIT_FRAMES).astype(int)]
    W = clip.wrist[:, hand]

    def residual(x):
        T = T_of(x)
        parts = [clip.appearance(W[t] @ T, t, FIT_RAYS) / SIG_A for t in fit]
        parts += [clip.tip_distance(W[t] @ T, t, hand, fingers) / SIG_B for t in fit]
        parts += end_terms(W[frames[0]] @ T, W[frames[-1]] @ T, start if frames[0] == a else None,
                           end if frames[-1] == b else None)
        r = np.concatenate([p for p in parts if len(p)])
        return r if len(r) else np.zeros(1)

    r = least_squares(residual, x_of(T0), loss="soft_l1", f_scale=1.0, max_nfev=80, x_scale=[0.1] * 3 + [0.01] * 3)
    return {"T": T_of(r.x), "cost": float(r.cost), "nfev": int(r.nfev), "frames": len(fit)}


def pass2(clip: Clip, a: int, b: int, hand: int, fingers: list[int], T: np.ndarray) -> tuple[np.ndarray, dict]:
    """Per frame wrist(t) T D(t), D refined on (a) + (b) with a soft prior on D's change per frame."""
    W = clip.wrist[:, hand]
    poses = np.full((b - a + 1, 4, 4), np.nan)
    med, q75 = np.full(b - a + 1, np.nan), np.full(b - a + 1, np.nan)
    D_prev = np.zeros(6)
    drift = []
    for i, t in enumerate(range(a, b + 1)):
        if not clip.valid[t, hand]:
            continue
        base = W[t] @ T

        def residual(x):
            P = base @ T_of(x)
            r = [clip.appearance(P, t) / SIG_A, clip.tip_distance(P, t, hand, fingers) / SIG_B,
                 np.concatenate([(x[:3] - D_prev[:3]) / STEP_RAD, (x[3:] - D_prev[3:]) / STEP_M])]
            return np.concatenate([p for p in r if len(p)])

        step = np.array([STEP_RAD] * 3 + [STEP_M] * 3)  # the spec's per-frame limit, per component
        lo, hi = D_prev - step, D_prev + step
        r = least_squares(residual, np.clip(D_prev, lo + 1e-12, hi - 1e-12), loss="soft_l1", f_scale=1.0,
                          max_nfev=30, x_scale=[0.05] * 3 + [0.005] * 3, bounds=(lo, hi))
        D_prev = r.x
        poses[i] = base @ T_of(r.x)
        res = np.abs(clip.appearance(poses[i], t))
        med[i] = float(np.median(res)) if len(res) else np.nan
        q75[i] = float(np.percentile(res, 75)) if len(res) else np.nan
        drift.append(r.x)
    drift = np.array(drift) if drift else np.zeros((1, 6))
    return poses, {"residual_a_median_m": med, "residual_a_q75_m": q75, "drift_max_m": float(np.abs(drift[:, 3:]).max()),
                   "drift_max_deg": float(np.degrees(np.linalg.norm(drift[:, :3], axis=1).max()))}


def slips(q75: np.ndarray) -> list[tuple[int, int]]:
    """Runs (relative, inclusive) of >= SLIP_FRAMES frames whose residual (a) exceeds SLIP_M at its 75th
    percentile (a median let a lagging 6 cm slip through: up to half of the residuals can be large)."""
    bad = np.nan_to_num(q75, nan=0.0) > SLIP_M
    out, start = [], None
    for i, x in enumerate(list(bad) + [False]):
        if x and start is None:
            start = i
        if not x and start is not None:
            if i - start >= SLIP_FRAMES:
                out.append((start, i - 1))
            start = None
    return out


def solve_grasp(clip: Clip, a: int, b: int, hand: int, fingers: list[int], start, end, split: bool = True,
                T_hint: np.ndarray | None = None, near: tuple | None = None) -> list[dict]:
    """The grasp [a, b] (and, after a slip, its parts and the bridge between them). start / end: the
    adjacent segments' poses (end terms (c)). Where to start pass 1 from, in order: start, end, T_hint
    (a part after a slip: the hand offset before it), near = (pose, before?) (the nearest known pose
    across an unstable bridge: the object rested there just before or after), identity."""
    W = clip.wrist[:, hand]
    first = next((t for t in range(a, b + 1) if clip.valid[t, hand]), None)
    if first is None:
        return [{"kind": "unstable", "start": a, "end": b, "why": "no wrist in the grasp"}]
    last = next(t for t in range(b, a - 1, -1) if clip.valid[t, hand])
    if start is not None:
        T0 = np.linalg.inv(W[first]) @ start
    elif end is not None:
        T0 = np.linalg.inv(W[last]) @ end
    elif T_hint is not None:
        T0 = T_hint
    elif near is not None:
        T0 = np.linalg.inv(W[first] if near[1] else W[last]) @ near[0]
    else:
        T0 = np.eye(4)
    p1 = pass1(clip, a, b, hand, fingers, start, end, T0)
    poses, p2 = pass2(clip, a, b, hand, fingers, p1["T"])
    med = p2["residual_a_median_m"]
    runs = slips(p2["residual_a_q75_m"]) if split else []
    if runs:
        out, cursor = [], a
        for s0, s1 in runs:
            if a + s0 - cursor >= MIN_GRASP:
                out += solve_grasp(clip, cursor, a + s0 - 1, hand, fingers, start if cursor == a else None, None,
                                   split=False, T_hint=p1["T"])
            elif a + s0 > cursor:
                out.append({"kind": "unstable", "start": cursor, "end": a + s0 - 1, "why": "short part before a slip"})
            out.append({"kind": "unstable", "start": a + s0, "end": a + s1, "why": "slip guard"})
            cursor = a + s1 + 1
        if b - cursor + 1 >= MIN_GRASP:
            out += solve_grasp(clip, cursor, b, hand, fingers, None, end, split=False, T_hint=p1["T"])
        elif cursor <= b:
            out.append({"kind": "unstable", "start": cursor, "end": b, "why": "short part after a slip"})
        return out
    ok = np.nan_to_num(med, nan=np.inf) <= FAIL_M
    return [{"kind": "grasp", "start": a, "end": b, "hand": hand, "fingers": fingers, "poses": poses,
             "T": p1["T"], "pass1": {k: v for k, v in p1.items() if k != "T"},
             "residual_a_median_m": float(np.nanmedian(med)) if np.isfinite(med).any() else None,
             "share_within_1cm": float(ok.mean()), "failed": bool(ok.mean() < FAIL_SHARE),
             "drift_max_m": p2["drift_max_m"], "drift_max_deg": p2["drift_max_deg"]}]


# ---------------------------------------------------------------- unstable
def interpolate(P0: np.ndarray, P1: np.ndarray, n: int) -> np.ndarray:
    """n poses strictly between P0 and P1: cubic Hermite position, zero end velocity; slerp rotation."""
    s = np.arange(1, n + 1) / (n + 1)
    h = 3 * s ** 2 - 2 * s ** 3
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, 3] = P0[:3, 3] + h[:, None] * (P1[:3, 3] - P0[:3, 3])
    out[:, :3, :3] = Slerp([0, 1], Rotation.from_matrix([P0[:3, :3], P1[:3, :3]]))(s).as_matrix()
    return out


def solve_unstable(clip: Clip, a: int, b: int, before: np.ndarray | None, after: np.ndarray | None) -> dict:
    P0 = before if before is not None else after
    P1 = after if after is not None else before
    if P0 is None:
        return {"poses": np.full((b - a + 1, 4, 4), np.nan), "refined": np.zeros(b - a + 1, bool),
                "why": "no neighbouring pose"}
    guess = interpolate(P0, P1, b - a + 1)
    poses, refined = guess.copy(), np.zeros(b - a + 1, bool)
    for i, t in enumerate(range(a, b + 1)):
        o = clip.obs[t]
        if o is None or o[2] < clip.min_share:
            continue

        def residual(x, i=i, t=t):
            P = guess[i] @ T_of(x)
            return np.concatenate([clip.appearance(P, t) / SIG_A,
                                   np.concatenate([x[:3] / UNSTABLE_PRIOR_RAD, x[3:] / UNSTABLE_PRIOR_M])])

        bound = np.array([UNSTABLE_REACH_RAD] * 3 + [UNSTABLE_REACH_M] * 3)
        r = least_squares(residual, np.zeros(6), loss="soft_l1", f_scale=1.0, max_nfev=7 * UNSTABLE_STEPS,
                          bounds=(-bound, bound))
        before = np.median(np.abs(clip.appearance(guess[i], t)))
        if np.median(np.abs(clip.appearance(guess[i] @ T_of(r.x), t))) < before:  # kept only if it explains more
            poses[i] = guess[i] @ T_of(r.x)
            refined[i] = True
    return {"poses": poses, "refined": refined}


# ---------------------------------------------------------------- the clip
def solve(clip: Clip, segments: list[dict], start_pose: np.ndarray, fps: float,
          fixed: dict[int, np.ndarray] | None = None) -> tuple[np.ndarray, list[dict]]:
    """(object_to_world (N, 4, 4), per-segment report). start_pose: s3a2_register's canonical_to_world
    (the first static segment's start). fixed: static segment start -> its pose, used as it is (S3d's
    settled poses: the grasps and bridges are solved again against them)."""
    n = clip.n
    P = np.full((n, 4, 4), np.nan)
    HANDS = ("L", "R")
    work = [dict(s) for s in segments]
    report = []

    def known(t):
        return 0 <= t < n and np.isfinite(P[t]).all()

    def run_grasp(s, both: bool):
        a, b = s["start"], s["end"]
        start = P[a - 1] if known(a - 1) else (start_pose if a == 0 else None)
        end = P[b + 1] if both and known(b + 1) else None
        before = next((t for t in range(a - 1, -1, -1) if known(t)), None)
        after = next((t for t in range(b + 1, n) if known(t)), None) if both else None
        near = (P[before], True) if before is not None else (P[after], False) if after is not None else (start_pose, True)
        hand = HANDS.index(s["hand"])
        return solve_grasp(clip, a, b, hand, [f for h, f in s["fingertips"] if h == s["hand"]], start, end, near=near)

    solved: dict[int, list[dict]] = {}
    for i, s in enumerate(work):  # sweep 1: statics and grasps in clip order
        a, b = s["start"], s["end"]
        if s["kind"] == "static" and fixed and a in fixed:
            P[a:b + 1] = fixed[a]
            solved[i] = [{"kind": "static", "start": a, "end": b, "residual_m": None, "start_from": "fixed (S3d)"}]
        elif s["kind"] == "static":
            last = next((t for t in range(a - 1, -1, -1) if known(t)), None)
            init = P[a - 1] if known(a - 1) else P[last] if last is not None else start_pose
            r = solve_static(clip, a, b, init)
            P[a:b + 1] = r["pose"]
            solved[i] = [{"kind": "static", "start": a, "end": b, **{k: v for k, v in r.items() if k != "pose"}}]
        elif s["kind"] == "grasp":
            parts = run_grasp(s, both=False)
            for p in parts:
                if p["kind"] == "grasp":
                    P[p["start"]:p["end"] + 1] = p["poses"]
            solved[i] = parts
    for i, s in enumerate(work):  # sweep 2: grasps with both neighbouring poses
        if s["kind"] == "grasp" and known(s["end"] + 1):
            parts = run_grasp(s, both=True)
            P[s["start"]:s["end"] + 1] = np.nan
            for p in parts:
                if p["kind"] == "grasp":
                    P[p["start"]:p["end"] + 1] = p["poses"]
            solved[i] = parts
    rows = [p for i in sorted(solved) for p in solved[i]]
    rows += [{"kind": "unstable", "start": s["start"], "end": s["end"]} for s in work if s["kind"] == "unstable"]
    refined = np.zeros(n, bool)
    for row in sorted(rows, key=lambda r: r["start"]):  # bridges: unstable segments, slip runs, frames left unsolved
        if row["kind"] == "unstable":
            a, b = row["start"], row["end"]
            if not known(a):  # one bridge over the whole unsolved run (ep18's dust pan: 0-6, then a 7-205 slip)
                while b + 1 < n and not known(b + 1):
                    b += 1
                before = P[a - 1] if known(a - 1) else start_pose if a == 0 else None  # frame 0: S3a's pose
                u = solve_unstable(clip, a, b, before, P[b + 1] if b + 1 < n and known(b + 1) else None)
                P[a:b + 1], refined[a:b + 1] = u["poses"], u["refined"]
                row.setdefault("why", u.get("why", ""))
            row.update(refined=int(refined[row["start"]:row["end"] + 1].sum()), why=row.get("why", ""))
        report.append({k: v for k, v in row.items() if k not in ("poses", "T")})
    for t in range(n):  # a wrist gap inside a grasp left a frame unsolved: hold the last pose
        if not known(t) and t > 0 and known(t - 1):
            P[t] = P[t - 1]
    for t in range(n - 2, -1, -1):
        if not known(t) and known(t + 1):
            P[t] = P[t + 1]
    jumps = np.linalg.norm(np.diff(P[:, :3, 3], axis=0), axis=1)
    speed = jumps * fps
    for row in report:
        a, b = row["start"], row["end"]
        row["max_jump_m"] = float(np.nanmax(jumps[a:b])) if b > a else 0.0
        if row["kind"] == "unstable":
            row["max_speed_m_s"] = float(np.nanmax(speed[a:b])) if b > a else 0.0
    return P, report
