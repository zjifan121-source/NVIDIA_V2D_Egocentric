"""Segment solving (V4_BUILD_SPEC.md S3c): one object's track from S3b's segments, each solved by the rule
that holds there. Poses map the canonical mesh frame to the world (V4 §3).

  static    one pose: ICP of the canonical mesh onto the segment's fused surface (scale fixed), from
            the previous segment's end pose (the registration's for the first) and its pose EARLIER
            frames before its end, each times each element of the object's symmetry group; the lowest
            median point-to-mesh residual wins (bar <= 5 mm), among fits within TIE_M of it the one
            whose posed mesh the fewest mask rays miss (rays_missed), then the one hanging out of the mask
            least (overhang). Global starts (FPFH + RANSAC) and the best fit so far turned about gravity
            (YAW_STEP) too when the fit misses that bar or rays missed plus overhang exceed MISSED_BAR
  grasp     pass 1: one hand->object offset T for the grasping hand, object(t) = wrist(t) T, fitted
            over the segment on (a) the canonical mesh's rendered depth vs the stereo depth inside the
            mask, with its silhouette against the mask the other way (Clip.silhouette_residual), (b)
            the contact fingertips' signed distance to the mesh, (c) the end poses equal to
            the neighbouring static poses (weight W_END; across S3b's unstable run of at most ANCHOR_GAP
            frames to a rest or the clip's start, that rest's pose); pass 2: per frame, object(t) = wrist(t) T
            D(t). D starts where ICP of the frame's depth points onto the mesh puts it, from the
            previous frame's D carried by the wrist (icp_start: 2 cm, then 1 cm; kept when it pairs
            >= ICP_FIT of the points, else that previous D), clipped into 5 mm / 3 degrees per
            component of the previous D, and is refined on (a) + (b) within that box (pulled towards
            the start by a soft prior): D may not jump (a soft prior alone let a 6 cm jump through in
            one frame, the depth outweighs it, and the slip guard never saw it; an unclipped ICP start
            turned ep41's near-round pot 165 degrees in the hand at 2 mm). Started from the previous D alone
            (until 2026-10-27), ep40's carries sat 3-5 cm / 20-40 degrees off the truth with residual
            (a) above the truth's on 75-85 % of frames: the hand tracker's wrist drifts 6-9 cm from
            the object over a carry, and (a)'s misses and silhouette are flat there; ICP from the
            same start tracked to 1.6 cm / 10 degrees. Pass 2 runs from the end backwards when the
            only known neighbour is after the grasp (the part after a slip: T fits that end, and from
            the start ep41's 91-frame carry never locked on, ICP pairing 12 %). Slip guard: (a) above
            SLIP_M for SLIP_FRAMES frames in a row splits the grasp there; that run becomes an unstable
            bridge and the parts are solved again. `failed`: (a) within 1 cm on fewer than half the
            frames: its poses anchor nothing, it is bridged as an unstable run (S4b then takes
            `keyframe` there)
  unstable  cubic Hermite position (zero end velocity) + slerp between the neighbouring end poses
            (consecutive unsolved segments as one bridge; at frame 0, s3a2_register's pose); over frames S3b
            saw a hand holding, also the object moved with that wrist from both ends (`carried`), whichever
            explains the frames better (`explains`),
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
MISSED_BAR = 0.05  # default: a static pose's mask rays that may miss the mesh (mask edges; the closed meshes 0-4 %),
# and its retry bar on rays missed plus overhang (the right fits 1-5 % on ep13/42/43/44)
TIE_M = 0.001  # default: static fits this close count as the same fit
RAYS_TIE = 0.02  # default: tied static fits whose missed-ray shares differ by this little count as alike (mask edges)
OVER_M, OVER_TIE = 0.005, 0.005  # default: how far outside the mask a projected surface point hangs out (mask
# edges); overhang shares this close count as alike (ep42's frame-0 rest: 1.0 % at the true turn, 2.1 % kept)
YAW_STEP = 45  # default: degrees between the start's turns about gravity in a static retry
UNSTABLE_STEPS = 5  # spec
EARLIER = (5, 10)  # default: a static fit also starts from the previous segment this many frames before its end
# (a grasp's last frames, the hand letting go, are its least sure: ep40's track flipped 140 degrees at the release
# and the rest after it fitted upside down, 2.4 mm against 1.0 from five frames earlier)
UNSTABLE_PRIOR_M, UNSTABLE_PRIOR_RAD = 0.02, np.radians(10)  # default: how far (a) may pull a bridge frame
UNSTABLE_REACH_M, UNSTABLE_REACH_RAD = 0.05, np.radians(20)  # default: the refinement's bound (a bridge frame
# whose guess is far off otherwise wandered 0.6 m on the synthetic slip)
FIT_FRAMES, FIT_RAYS, RAYS = 60, 150, 300  # default: pass 1 subsample; rays per frame (pass 1, pass 2)
MIN_GRASP = 5
RELEASE_M, RELEASE_FRAMES = 0.01, 10  # default: a grasp's frames next to a rest whose wrist moves more than this per
# frame (at most this many per end) are the hand arriving or letting go: bridged, not tracked (ep41's left wrist
# moved 2-12 cm per frame over the carry's last 6 frames, 0.1-0.4 before; the release frame anchored the whole carry)
BRIDGE_FAILED = True  # default: a failed grasp's poses anchor nothing; its frames are bridged (ep26's carry)
ANCHOR_GAP = RELEASE_FRAMES  # default: a grasp whose S3b neighbour is an unstable run at most this long with a rest
# (or the clip's start) beyond it takes that rest's pose as its end term, as across its own trimmed frames (ep43:
# 0-5 and 272-274 left its carry 6-271 without either end, pass 1 drifted and the slip guard took all of it). Public,
# 20 episodes: best of three 0.536 -> 0.546 (ep31 +0.115, ep43 +0.080, ep39 +0.017; ep26 -0.012)
CARRIED_BRIDGES = True  # default: a bridge over frames S3b saw a hand holding also tries the object moved with that
# wrist (`carried`) and keeps whichever guess explains the frames better (`explains`). Public, 20 episodes, on top of
# the gap anchor: best of three 0.546 -> 0.554 (12, 13, 26, 41, 42 up 0.02-0.05; ep45 -0.008)
ICP_FIT = 0.3  # default: share of a frame's depth points ICP pairs (within 1 cm) for its pose to start pass 2
ICP_REACH = (0.02, 0.01)  # default: ICP's pairing distance, coarse then fine (m)
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
def solve_static(clip: Clip, a: int, b: int, init: np.ndarray, earlier: list[np.ndarray] = ()) -> dict:
    """One pose over [a, b] by ICP onto the segment's fused surface, best over the symmetry group."""
    target = clip.target(a, b)
    if target is None or len(target.points) < 50:
        return {"pose": init, "residual_m": None, "candidates": 0, "why": "no fused surface: the start pose kept"}
    pts = np.asarray(target.points)
    centred = np.eye(4)  # the start pose's rotation, the mesh centred on the fused surface
    centred[:3, :3] = init[:3, :3]
    centred[:3, 3] = pts.mean(axis=0) - init[:3, :3] @ np.asarray(clip.surface.points).mean(axis=0)

    def fit(starts, label, group=None):
        found = []
        for k, G in enumerate(clip.group if group is None else group):
            for name, s0 in starts:
                T = S.icp(clip.surface, target, s0 @ G, max_dist=0.02, iterations=60, point_to_point=True)["T"]
                T = S.icp(clip.surface, target, T, max_dist=0.005, iterations=60)["T"]
                local = (pts - T[:3, 3]) @ T[:3, :3]
                res = float(np.median(np.abs(clip.scene.compute_distance(o3d.core.Tensor(local.astype(np.float32))).numpy())))
                found.append({"pose": T, "residual_m": res, "symmetry_index": k, "start_from": name or label})
        return found

    def pick(found):
        # fits within TIE_M of the best fit the fused surface alike (a box's half turn, a pot turned about its
        # axis: the fused surface is one side of it): among them the fewest mask rays missing the posed mesh
        # (within RAYS_TIE), then the least mesh hanging out of the mask (within OVER_TIE), then the least turned
        # from the start pose. Overhang only breaks ties: an occluded or thinly masked part hangs out at the
        # true pose too (ep31's sweeper: 3 % missed, 12 % hanging out; turned 180 degrees 6 % and 0.3 %)
        floor = min(f["residual_m"] for f in found)
        near = [f for f in found if f["residual_m"] <= floor + TIE_M]
        for f in near:
            if "rays_missed" not in f:
                f["rays_missed"], f["overhang"] = rays_missed(clip, a, b, f["pose"]), overhang(clip, a, b, f["pose"])
        for key, tie in (("rays_missed", RAYS_TIE), ("overhang", OVER_TIE)):
            least = min((f[key] for f in near if f[key] is not None), default=None)
            near = [f for f in near if least is None or f[key] is None or f[key] <= least + tie]
        return min(near, key=lambda f: Rotation.from_matrix(f["pose"][:3, :3] @ init[:3, :3].T).magnitude())

    found = fit([("previous pose", init), ("centred on the fused surface", centred)]
                + [(f"the previous segment {k} frames before its end", E) for k, E in zip(EARLIER, earlier)], "")
    best = pick(found)
    if best["residual_m"] > STATIC_BAR_M or (disagreement(best) or 0) > MISSED_BAR:
        # the start was too far (a bad grasp before it), or the mask disagrees with the fit (a turn the fused
        # surface cannot tell: ep45's pot, 81 degrees off at 1.4 mm, 30 % missed; ep13's cup upside down, 2 %
        # missed but 16 % hanging out of the mask): global starts, then the best fit so far, global ones included,
        # turned about gravity (its tilt and place are sure, its turn is not; turning the start instead missed
        # ep26's, ep27's and ep43's final rests, 2-3 % missed and 4-6 degrees off against the 13-14 % kept)
        found += fit([("", S.global_start(clip.surface, target, seed=seed)) for seed in range(3)], "global (FPFH + RANSAC)")
        best, mid, turned = pick(found), pts.mean(axis=0), []
        for deg in range(YAW_STEP, 360, YAW_STEP):
            Rz = Rotation.from_euler("z", deg, degrees=True).as_matrix()
            E = best["pose"].copy()
            E[:3, :3], E[:3, 3] = Rz @ E[:3, :3], Rz @ (E[:3, 3] - mid) + mid
            turned.append((f"the best fit turned {deg} degrees about gravity", E))
        best = pick(found + fit(turned, "", clip.group[:1]))  # a cup's own group turns it about its axis
    return best | {"candidates": len(clip.group), "points": len(pts)}


def disagreement(f: dict) -> float | None:
    """A static fit's disagreement with the mask both ways: rays missed plus overhang (None without a mask)."""
    if f.get("rays_missed") is None:
        return None
    return f["rays_missed"] + (f.get("overhang") or 0.0)


def overhang(clip: Clip, a: int, b: int, pose: np.ndarray, frames: int = 10) -> float | None:
    """The share of the posed mesh's projected surface points more than OVER_M outside the allowed mask (the
    object's and the hands'), over up to `frames` frames of [a, b]: rays_missed's other direction. A cup
    upside down on its rest misses no more rays than upright (the mask is its side) but hangs out of it."""
    ts = [t for t in range(a, b + 1) if clip.obs[t] is not None]
    r = [clip.silhouette_residual(pose, t) for t in ts[:: max(1, len(ts) // frames)]]
    r = np.concatenate(r) if r else np.zeros(0)
    return float(np.mean(r > OVER_M)) if len(r) else None


def rays_missed(clip: Clip, a: int, b: int, pose: np.ndarray, frames: int = 10) -> float | None:
    """The share of the mask's rays that miss the mesh at a static pose (up to `frames` frames of [a, b]):
    a see-through mesh (ep40's dust pan, a voxel net: 27 %) fits the fused surface to 1 mm all the same,
    and every grasp then trips the slip guard on the misses alone."""
    ts = [t for t in range(a, b + 1) if clip.obs[t] is not None]
    r = [clip.depth_residual(pose, t) for t in ts[:: max(1, len(ts) // frames)]]
    r = np.concatenate(r) if r else np.zeros(0)
    return float(np.mean(r == MISS)) if len(r) else None


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


def frame_cloud(clip: Clip, t: int) -> o3d.geometry.PointCloud | None:
    """Frame t's observed mask pixels as world points, normals towards the camera (None under 50)."""
    o = clip.obs[t]
    if o is None or len(o[1]) < 50:
        return None
    uv, d, _ = o
    K, c2w = clip.K, clip.c2w[t]
    cam = np.stack([(uv[:, 0] - K[0, 2]) * d / K[0, 0], (uv[:, 1] - K[1, 2]) * d / K[1, 1], d], axis=1)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cam @ c2w[:3, :3].T + c2w[:3, 3]))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pc.orient_normals_towards_camera_location(c2w[:3, 3])
    return pc


def icp_start(clip: Clip, t: int, guess: np.ndarray) -> tuple[np.ndarray, float]:
    """Frame t's depth points ICP'd onto the mesh posed at `guess` (point-to-plane, ICP_REACH): the
    corrected pose and the share of points paired; `guess` itself when fewer than ICP_FIT pair."""
    pc = frame_cloud(clip, t)
    if pc is None:
        return guess, 0.0
    pts, nrm = np.asarray(clip.surface.points), np.asarray(clip.surface.normals)
    target = S.cloud(pts @ guess[:3, :3].T + guess[:3, 3], nrm @ guess[:3, :3].T)
    T, fitness = np.eye(4), 0.0
    for reach in ICP_REACH:
        r = S.icp(pc, target, T, max_dist=reach, iterations=30)
        T, fitness = r["T"], r["fitness"]
    return (np.linalg.inv(T) @ guess, fitness) if fitness >= ICP_FIT else (guess, fitness)


def pass2(clip: Clip, a: int, b: int, hand: int, fingers: list[int], T: np.ndarray,
          backward: bool = False) -> tuple[np.ndarray, dict]:
    """Per frame wrist(t) T D(t): D starts from ICP (icp_start, from the previous D) clipped into the per-frame
    step around the previous D, and is refined on (a) + (b) within that step. backward: from b down to a (a
    part whose only known neighbour is after it: T holds there, the wrist drifts away from it before)."""
    W = clip.wrist[:, hand]
    poses = np.full((b - a + 1, 4, 4), np.nan)
    med, q75, paired = np.full(b - a + 1, np.nan), np.full(b - a + 1, np.nan), np.full(b - a + 1, np.nan)
    D_prev = np.zeros(6)
    drift = []
    for t in (range(b, a - 1, -1) if backward else range(a, b + 1)):
        i = t - a
        if not clip.valid[t, hand]:
            continue
        base = W[t] @ T
        step = np.array([STEP_RAD] * 3 + [STEP_M] * 3)  # the spec's per-frame limit, per component
        start, paired[i] = icp_start(clip, t, base @ T_of(D_prev))
        lo, hi = D_prev - step, D_prev + step  # ICP's own step is unbounded (a near-round pot turned 165
        # degrees in ep41's hand at 2 mm): its start is clipped into the spec's box around the previous D
        D0 = np.clip(x_of(np.linalg.inv(base) @ start), lo + 1e-12, hi - 1e-12)

        def residual(x):
            P = base @ T_of(x)
            r = [clip.appearance(P, t) / SIG_A, clip.tip_distance(P, t, hand, fingers) / SIG_B,
                 np.concatenate([(x[:3] - D0[:3]) / STEP_RAD, (x[3:] - D0[3:]) / STEP_M])]
            return np.concatenate([p for p in r if len(p)])

        r = least_squares(residual, D0, loss="soft_l1", f_scale=1.0,
                          max_nfev=30, x_scale=[0.05] * 3 + [0.005] * 3, bounds=(lo, hi))
        D_prev = r.x
        poses[i] = base @ T_of(r.x)
        res = np.abs(clip.appearance(poses[i], t))
        med[i] = float(np.median(res)) if len(res) else np.nan
        q75[i] = float(np.percentile(res, 75)) if len(res) else np.nan
        drift.append(r.x)
    drift = np.array(drift) if drift else np.zeros((1, 6))
    return poses, {"residual_a_median_m": med, "residual_a_q75_m": q75, "drift_max_m": float(np.abs(drift[:, 3:]).max()),
                   "drift_max_deg": float(np.degrees(np.linalg.norm(drift[:, :3], axis=1).max())),
                   "icp_paired_median": float(np.nanmedian(paired)) if np.isfinite(paired).any() else None}


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


def trimmed(clip: Clip, a: int, b: int, hand: int, at_start: bool, at_end: bool) -> tuple[int, int]:
    """[a, b] without the end frames next to a rest whose wrist moves more than RELEASE_M per frame."""
    W, ok = clip.wrist[:, hand, :3, 3], clip.valid[:, hand]

    def fast(t, u):
        return ok[t] and ok[u] and np.linalg.norm(W[t] - W[u]) > RELEASE_M
    a2, b2 = a, b
    while at_end and b2 > a2 and b - b2 < RELEASE_FRAMES and fast(b2, b2 - 1):
        b2 -= 1
    while at_start and a2 < b2 and a2 - a < RELEASE_FRAMES and fast(a2, a2 + 1):
        a2 += 1
    return a2, b2


def solve_grasp(clip: Clip, a: int, b: int, hand: int, fingers: list[int], start, end, split: bool = True,
                T_hint: np.ndarray | None = None, near: tuple | None = None, trim: tuple = (False, False)) -> list[dict]:
    """The grasp [a, b] (and, after a slip, its parts and the bridge between them). start / end: the
    adjacent segments' poses (end terms (c)). Where to start pass 1 from, in order: start, end, T_hint
    (a part after a slip: the hand offset before it), near = (pose, before?) (the nearest known pose
    across an unstable bridge: the object rested there just before or after), identity. trim: (start,
    end) next to a rest: the hand's arriving / letting-go frames there are bridged (trimmed)."""
    a2, b2 = trimmed(clip, a, b, hand, *trim)
    if (a2, b2) != (a, b):
        rows = [{"kind": "unstable", "start": a, "end": a2 - 1, "why": "the hand arriving"}] if a2 > a else []
        tail = [{"kind": "unstable", "start": b2 + 1, "end": b, "why": "the hand letting go"}] if b2 < b else []
        if b2 - a2 + 1 < MIN_GRASP:
            return [{"kind": "unstable", "start": a, "end": b, "why": "the hand arriving or letting go"}]
        return rows + solve_grasp(clip, a2, b2, hand, fingers, start, end, split, T_hint, near) + tail
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
    poses, p2 = pass2(clip, a, b, hand, fingers, p1["T"], backward=start is None and end is not None)
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
             "drift_max_m": p2["drift_max_m"], "drift_max_deg": p2["drift_max_deg"],
             "icp_paired_median": p2["icp_paired_median"]}]


# ---------------------------------------------------------------- unstable
def interpolate(P0: np.ndarray, P1: np.ndarray, n: int) -> np.ndarray:
    """n poses strictly between P0 and P1: cubic Hermite position, zero end velocity; slerp rotation."""
    s = np.arange(1, n + 1) / (n + 1)
    h = 3 * s ** 2 - 2 * s ** 3
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, 3] = P0[:3, 3] + h[:, None] * (P1[:3, 3] - P0[:3, 3])
    out[:, :3, :3] = Slerp([0, 1], Rotation.from_matrix([P0[:3, :3], P1[:3, :3]]))(s).as_matrix()
    return out


def blend(A: np.ndarray, B: np.ndarray, w: float) -> np.ndarray:
    """The pose w of the way from A to B: translation linear, rotation slerp."""
    out = np.eye(4)
    out[:3, 3] = (1 - w) * A[:3, 3] + w * B[:3, 3]
    out[:3, :3] = Slerp([0, 1], Rotation.from_matrix([A[:3, :3], B[:3, :3]]))([w]).as_matrix()[0]
    return out


def carried(clip: Clip, a: int, b: int, before: np.ndarray | None, after: np.ndarray | None,
            carrier: np.ndarray) -> np.ndarray | None:
    """The bridge [a, b] moved with the hands S3b saw holding the object: from frame t-1 to t the object moves
    rigidly with the wrist of carrier[t] (>= 0, that wrist valid on both frames), else it stays. Forward from
    `before` (the pose at a-1; at frame 0 the pose of frame 0 itself), backward from `after` (the pose at b+1),
    the two blended by the frame's place in the bridge so both ends hold. None when no frame is carried."""
    n, W, ok = clip.n, clip.wrist, clip.valid

    def step(t):  # frame t-1 -> t: the carrying wrist's motion, or None
        c = carrier[t] if 0 < t < n else -1
        return W[t, c] @ np.linalg.inv(W[t - 1, c]) if c >= 0 and ok[t, c] and ok[t - 1, c] else None
    steps = {t: step(t) for t in range(max(a, 1), min(b + 1, n - 1) + 1)}
    if not any(s is not None for s in steps.values()):
        return None
    fwd = bwd = None
    if before is not None:
        fwd, Q = np.empty((b - a + 1, 4, 4)), before
        for t in range(a, b + 1):
            S_ = steps.get(t) if (t > a or a > 0) else None
            Q = S_ @ Q if S_ is not None else Q
            fwd[t - a] = Q
    if after is not None:
        bwd, Q = np.empty((b - a + 1, 4, 4)), after
        for t in range(b, a - 1, -1):
            S_ = steps.get(t + 1)
            Q = np.linalg.inv(S_) @ Q if S_ is not None else Q
            bwd[t - a] = Q
    if fwd is None or bwd is None:
        return fwd if bwd is None else bwd
    return np.stack([blend(fwd[i], bwd[i], (i + 1) / (b - a + 2)) for i in range(b - a + 1)])


def explains(clip: Clip, poses: np.ndarray, a: int) -> float | None:
    """Mean over the frames with a mask above the bar of the mean |residual (a)| of `poses` from frame a, each
    residual capped at MISS. A median saturates at MISS once half the rays miss (ep43's 0-274 bridge: 3 cm
    for both guesses), the share that misses still tells them apart."""
    r = [float(np.mean(np.minimum(np.abs(clip.appearance(P, a + i)), MISS))) for i, P in enumerate(poses)
         if clip.obs[a + i] is not None and clip.obs[a + i][2] >= clip.min_share]
    return float(np.mean(r)) if r else None


def solve_unstable(clip: Clip, a: int, b: int, before: np.ndarray | None, after: np.ndarray | None,
                   carrier: np.ndarray | None = None) -> dict:
    P0 = before if before is not None else after
    P1 = after if after is not None else before
    if P0 is None:
        return {"poses": np.full((b - a + 1, 4, 4), np.nan), "refined": np.zeros(b - a + 1, bool),
                "why": "no neighbouring pose"}
    guess, how = interpolate(P0, P1, b - a + 1), {"guess": "interpolated"}
    alt = carried(clip, a, b, before, after, carrier) if carrier is not None else None
    if alt is not None:
        how.update(explains_interpolated_m=explains(clip, guess, a), explains_carried_m=explains(clip, alt, a))
        if how["explains_carried_m"] is not None and (how["explains_interpolated_m"] is None
                                                      or how["explains_carried_m"] < how["explains_interpolated_m"]):
            guess, how["guess"] = alt, "carried"
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
    return {"poses": poses, "refined": refined, **how}


# ---------------------------------------------------------------- the clip
def solve(clip: Clip, segments: list[dict], start_pose: np.ndarray, fps: float,
          fixed: dict[int, np.ndarray] | None = None) -> tuple[np.ndarray, list[dict]]:
    """(object_to_world (N, 4, 4), per-segment report). start_pose: s3a2_register's canonical_to_world
    (the first static segment's start). fixed: static segment start -> its pose, used as it is (S3d's
    settled poses: the grasps and bridges are solved again against them). Open3D's OpenMP runs on one thread
    here: its multithreaded ICP sums in a different order each run, and the solver's thresholds (slip guard,
    kept-if-better, the static tie-breaks) turn that into different tracks (ep45: 0.637 and 0.482 from the same
    code and inputs; one thread: identical tracks, 0.632)."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1, user_api="openmp"):
        return _solve(clip, segments, start_pose, fps, fixed)


def _solve(clip: Clip, segments: list[dict], start_pose: np.ndarray, fps: float,
           fixed: dict[int, np.ndarray] | None) -> tuple[np.ndarray, list[dict]]:
    n = clip.n
    P = np.full((n, 4, 4), np.nan)
    HANDS = ("L", "R")
    work = [dict(s, i=i) for i, s in enumerate(segments)]
    report = []

    def known(t):
        return 0 <= t < n and np.isfinite(P[t]).all()

    def across(i: int, side: int) -> np.ndarray | None:
        """The rest pose beyond S3b's short unstable neighbour of segment i on `side` (-1 before, +1 after): the
        hand arriving or letting go, the object still at that rest. At the clip's start, s3a2_register's pose."""
        j = i + side
        if not (0 <= j < len(work)) or work[j]["kind"] != "unstable" or work[j]["end"] - work[j]["start"] + 1 > ANCHOR_GAP:
            return None
        if side < 0 and work[j]["start"] == 0:
            return start_pose
        k = j + side
        if 0 <= k < len(work) and work[k]["kind"] == "static":
            t = work[k]["end"] if side < 0 else work[k]["start"]
            return P[t] if known(t) else None
        return None

    def run_grasp(s, both: bool):
        a, b = s["start"], s["end"]
        start = P[a - 1] if known(a - 1) else (start_pose if a == 0 else across(s["i"], -1))
        end = P[b + 1] if both and known(b + 1) else across(s["i"], +1) if both else None
        before = next((t for t in range(a - 1, -1, -1) if known(t)), None)
        after = next((t for t in range(b + 1, n) if known(t)), None) if both else None
        near = (P[before], True) if before is not None else (P[after], False) if after is not None else (start_pose, True)
        hand = HANDS.index(s["hand"])
        rest = [work[j]["kind"] == "static" if 0 <= j < len(work) else False for j in (s["i"] - 1, s["i"] + 1)]
        return solve_grasp(clip, a, b, hand, [f for h, f in s["fingertips"] if h == s["hand"]], start, end, near=near,
                           trim=tuple(rest))

    solved: dict[int, list[dict]] = {}
    for i, s in enumerate(work):  # sweep 1: statics and grasps in clip order
        a, b = s["start"], s["end"]
        if s["kind"] == "static" and fixed and a in fixed:
            P[a:b + 1] = fixed[a]
            solved[i] = [{"kind": "static", "start": a, "end": b, "residual_m": None, "start_from": "fixed (S3d)"}]
        elif s["kind"] == "static":
            last = next((t for t in range(a - 1, -1, -1) if known(t)), None)
            init = P[a - 1] if known(a - 1) else P[last] if last is not None else start_pose
            earlier = [P[last - k] for k in EARLIER if last is not None and known(last - k)]
            r = solve_static(clip, a, b, init, earlier)
            P[a:b + 1] = r["pose"]
            solved[i] = [{"kind": "static", "start": a, "end": b, **{k: v for k, v in r.items() if k != "pose"}}]
        elif s["kind"] == "grasp":
            parts = run_grasp(s, both=False)
            for p in parts:
                if p["kind"] == "grasp" and not (BRIDGE_FAILED and p.get("failed")):  # bridged below
                    P[p["start"]:p["end"] + 1] = p["poses"]
            solved[i] = parts
    for i, s in enumerate(work):  # sweep 2: grasps with both neighbouring poses
        if s["kind"] == "grasp" and (known(s["end"] + 1) or across(i, +1) is not None):
            parts = run_grasp(s, both=True)
            P[s["start"]:s["end"] + 1] = np.nan
            for p in parts:
                if p["kind"] == "grasp" and not (BRIDGE_FAILED and p.get("failed")):
                    P[p["start"]:p["end"] + 1] = p["poses"]
            solved[i] = parts
    rows = [p for i in sorted(solved) for p in solved[i]]
    rows += [{"kind": "unstable", "start": s["start"], "end": s["end"]} for s in work if s["kind"] == "unstable"]
    refined = np.zeros(n, bool)
    carrier = np.full(n, -1)  # per frame: the hand S3b saw holding the object, except its arriving / letting-go frames
    for s in work if CARRIED_BRIDGES else ():
        if s["kind"] == "grasp":
            carrier[s["start"]:s["end"] + 1] = HANDS.index(s["hand"])
    for row in rows:
        if row["kind"] == "unstable" and ("arriving" in row.get("why", "") or "letting go" in row.get("why", "")):
            carrier[row["start"]:row["end"] + 1] = -1
    # bridges: unstable segments, slip runs, failed grasps, frames left unsolved. A failed grasp (its fit within
    # 1 cm on under half its frames) is bridged like an unstable run: ep26's carry, placed 56 cm off, pulled the
    # bridges on both sides to it, and S4b's interpolation over the grasp alone then ran between those
    for row in sorted(rows, key=lambda r: r["start"]):
        if row["kind"] == "unstable" or (BRIDGE_FAILED and row.get("failed")):
            a, b = row["start"], row["end"]
            if not known(a):  # one bridge over the whole unsolved run (ep18's dust pan: 0-6, then a 7-205 slip)
                while b + 1 < n and not known(b + 1):
                    b += 1
                before = P[a - 1] if known(a - 1) else start_pose if a == 0 else None  # frame 0: S3a's pose
                u = solve_unstable(clip, a, b, before, P[b + 1] if b + 1 < n and known(b + 1) else None,
                                   carrier if CARRIED_BRIDGES else None)
                P[a:b + 1], refined[a:b + 1] = u["poses"], u["refined"]
                row.setdefault("why", u.get("why", ""))
                row.update({k: u[k] for k in ("guess", "explains_interpolated_m", "explains_carried_m") if k in u},
                           bridge=[a, b])
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
