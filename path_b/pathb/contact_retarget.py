"""Contact-preserving retarget pass on Path B's stages 2-3 (handoff "Four hand failures", case ep1; James, 2026-11-02:
"try different attempts until the problem is solved"). OFF by default (ENABLED; the driver's request key "contact_retarget").

NVIDIA's IK (scripts/retarget/ego_recon_to_sharpa.py, params.SHARPA_TO_MANO_MAPPING) matches the Sharpa tip and MP sites to
wrist + 1.2 x (MANO joint - wrist), with no object term: on the reference's contact frames the elastomer pads keep the human
tip gap but do not close it (path_b/data/outputs/hand_audit_20261102/ep1). This pass runs after that IK, on the processed
Parquet, per hand:
- a finger is in contact on the frames where the reference labels its distal MANO link (mano_<side>_object_contact_part_ids,
  link_<finger>3: the labels FlashCHORD's contact objective keys on); gaps up to GAP_CLOSE frames are closed and runs shorter
  than MIN_RUN frames dropped;
- on those frames, Levenberg-Marquardt on that finger's joints only, bounded by MAX_DELTA from NVIDIA's solution and by the
  joint limits, puts the deepest point of the pad's palm side (the elastomer caps the whole fingertip) MARGIN outside the
  labelled object's own surface (signed distance to its sampled surface; a pad farther than PAD_NEAR_M goes to the contact's
  tangent plane), turns the palm side against the surface (W_NORMAL: the pad's palmar normal opposite the surface normal),
  weakly (W_TANGENT) at the human contact point (mano_<side>_object_contact_positions / _normals: the object surface point and
  its outward normal), with a penetration penalty on the moved fingers' collision points (W_COLL), a pull to NVIDIA's
  solution (W_IK) and to the previous frame's correction (W_TIME);
- the corrections ramp in BLEND frames before a contact run and out BLEND frames after it, then are smoothed over time
  (SMOOTH_SIGMA frames);
- the wrist stays NVIDIA's (WRIST_STEP, off: a bounded wrist translation, at most MAX_WRIST_M, toward the pads' residual).
apply() rewrites robot_<side>_finger_joints and robot_<side>_frames in the partition's Parquet (the original kept under
<out>/intermediate/contact_retarget/) and writes <out>/contact_retarget.json with metrics() before and after: pad gaps on the
contact frames (unsigned; signed; the palm side facing the surface; valid = the palm side within -TOUCH_IN_M..GAP_NEAR_M of
the surface and facing it), robot penetration (exact only where the mesh is watertight, else UNKNOWN, with an approximate
depth from the sampled surface normals beside it), finger joint steps and accelerations, frames changed.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ENABLED = False  # default: the driver runs the pass only when this (or the request's "contact_retarget") is True
WRIST_STEP = False  # default: a bounded wrist translation after the finger step
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
DISTAL_LINK = {"thumb": 3, "index": 6, "middle": 9, "ring": 12, "pinky": 15}  # mano_link_names: link_<finger>3
GAP_CLOSE = 2  # frames: a contact label gap this short is closed
MIN_RUN = 3  # frames: shorter contact runs are not corrected (label flicker)
MARGIN = 0.0005  # m: the pad's deepest point this far outside the object surface (touching, not pressing)
COLL_MARGIN = 0.0  # m: a collision point closer than this to the surface (or inside) is penalised
MAX_DELTA = 0.5  # rad: per finger joint, at most this far from NVIDIA's solution
MAX_WRIST_M = 0.01  # m: WRIST_STEP's bound
W_TANGENT = 0.04  # the pad point's tangent offset from the human contact point, against its normal offset (1)
W_COLL = 25.0
W_IK = 1e-6  # m^2/rad^2: a 0.3 rad change costs like 0.3 mm of contact error
W_TIME = 4e-6  # m^2/rad^2, on the change of the correction from the previous frame's
ITERS = 20
BLEND = 3  # frames
SMOOTH_SIGMA = 1.0  # frames
SURFACE_SAMPLES = 200_000
GAP_NEAR_M = 0.005  # metrics: a pad within this of the object counts as touching
TOUCH_IN_M = 0.002  # metrics: a pad deeper than this inside the object is penetrating, not touching
EXACT_NEAR_M = 0.02  # metrics: the exact inside test runs on points this close to the surface
SOLVE_NEAR_M = 0.02  # the penetration term looks this far from the surface (farther points: outside)
SOLVE_POINTS = 40  # points per finger link in the penetration term (pads: all; metrics: all)
PAD_NEAR_M = 0.05  # a pad this close to its object is put on the object's own surface, else on the contact's tangent plane
PALMAR_OFFSET_M = 0.002  # pad points this far palmward of the pad's centroid are its palm side (the contact term uses them)
W_NORMAL = 1e-4  # m^2 per (1 + cos): a 60 degree turn of the pad's palm side away from the surface costs like 5 mm
FACING_COS = 0.5  # metrics: the pad's palm side faces the object within 60 degrees
METRIC_NEAR_M = 0.05  # metrics: robot points farther than this from the surface count as outside


# ---------------------------------------------------------------- geometry
def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(np.asarray(q, np.float64), scalar_first=True).as_matrix()


class ObjectSurface:
    """An object's mesh in its own frame as a dense surface sample with outward normals: unsigned and (approximate) signed
    distances by the nearest sample; exact inside tests only where the mesh is watertight."""

    def __init__(self, mesh, samples: int = SURFACE_SAMPLES, seed: int = 0) -> None:
        import trimesh
        from scipy.spatial import cKDTree
        rng = np.random.default_rng(seed)
        area = mesh.area_faces / mesh.area_faces.sum()
        faces = rng.choice(len(mesh.faces), size=samples, p=area)
        uv = rng.random((samples, 2))
        flip = uv.sum(1) > 1
        uv[flip] = 1 - uv[flip]
        tri = mesh.triangles[faces]
        pts = tri[:, 0] + uv[:, :1] * (tri[:, 1] - tri[:, 0]) + uv[:, 1:] * (tri[:, 2] - tri[:, 0])
        self.points = np.vstack([pts, mesh.vertices])
        self.normals = np.vstack([mesh.face_normals[faces], mesh.vertex_normals])
        self.tree = cKDTree(self.points)
        self.mesh = mesh
        self.watertight = bool(mesh.is_watertight)
        self._trimesh = trimesh

    def query(self, p: np.ndarray, upper: float = np.inf) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Object-frame points (n, 3) -> unsigned distance (n,), approximate signed distance (n,; < 0 inside) and the
        nearest sample's outward normal (n, 3). Points farther than `upper` from the surface come back at inf, outside."""
        d, i = self.tree.query(p, distance_upper_bound=upper)
        far = ~np.isfinite(d)
        i = np.where(far, 0, i)
        n = np.where(far[:, None], 0.0, self.normals[i])
        sd = np.where(far, np.inf, np.einsum("ij,ij->i", p - self.points[i], n))
        return d, sd, n

    def inside(self, p: np.ndarray) -> np.ndarray | None:
        """Exact inside test (watertight meshes only), else None."""
        return self.mesh.contains(p) if self.watertight else None


class Placed:
    """An ObjectSurface at one frame's world pose."""

    def __init__(self, surface: ObjectSurface, position: np.ndarray, wxyz: np.ndarray) -> None:
        self.surface, self.p, self.R = surface, np.asarray(position, np.float64), quat_wxyz_to_matrix(wxyz)

    def to_object(self, x: np.ndarray) -> np.ndarray:
        return (x - self.p) @ self.R

    def query(self, x: np.ndarray, upper: float = np.inf) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d, sd, n = self.surface.query(self.to_object(x), upper)
        return d, sd, n @ self.R.T  # normals back to the world


# ---------------------------------------------------------------- kinematics
def point_jacobians(J: np.ndarray, r: np.ndarray) -> np.ndarray:
    """(6, k) joint Jacobian [linear; angular] of a body's origin in world-aligned axes and (n, 3) offsets r of points fixed
    in that body from its origin -> (n, 3, k): the points' velocity is v + w x r per joint column."""
    return J[None, :3, :] + np.cross(J[3:].T[None, :, :], r[:, None, :]).transpose(0, 2, 1)


class Geometry:
    """A collision geometry: local points (n, 3) in its parent joint's frame, the finger it moves with (None: the palm), the
    subset of its points the solver's penetration term uses (None: all); for a pad, its palm-side points (None: all) and its
    palmar normal in the joint frame."""

    def __init__(self, name: str, joint: int, local: np.ndarray, finger: str | None, pad: bool,
                 solve: np.ndarray | None = None, palmar: np.ndarray | None = None, axis: np.ndarray | None = None) -> None:
        self.name, self.joint, self.local, self.finger, self.pad, self.solve = name, joint, local, finger, pad, solve
        self.palmar, self.axis = palmar, axis


class SharpaHand:
    """One Sharpa Wave hand as NVIDIA's IK builds it (pinocchio RobotWrapper.BuildFromMJCF with a free-flyer root; q = wrist
    xyz, wrist quaternion xyzw, then the finger joints in model order = robot_<side>_finger_joint_names), with its MJCF
    collision geometry (the elastomer pads are the meshes named *elastomer.STL) sampled to at most `points` per geometry."""

    def __init__(self, side: str, xml: str | Path, points: int = 150) -> None:
        import pinocchio as pin
        import trimesh
        self.pin, self.side = pin, side
        rw = pin.RobotWrapper.BuildFromMJCF(filename=str(xml), root_joint=pin.JointModelFreeFlyer())
        self.model, self.data = rw.model, rw.model.createData()
        self.joint_names = [self.model.names[i] for i in range(2, self.model.njoints)]
        self.lower = np.asarray(self.model.lowerPositionLimit[7:], np.float64)
        self.upper = np.asarray(self.model.upperPositionLimit[7:], np.float64)
        self.finger_dofs = {f: [k for k, n in enumerate(self.joint_names) if n.startswith(f"{side}_{f}_")] for f in FINGERS}
        self.frame_names = [f.name for f in self.model.frames]
        rng = np.random.default_rng(0)
        self.geoms: list[Geometry] = []
        for g in rw.collision_model.geometryObjects:
            v = np.asarray(trimesh.load(g.meshPath, force="mesh").vertices, np.float64) * np.asarray(g.meshScale)
            if len(v) > points:
                v = v[rng.choice(len(v), points, replace=False)]
            local = v @ g.placement.rotation.T + g.placement.translation
            joint = self.model.names[g.parentJoint]
            finger = next((f for f in FINGERS if joint.startswith(f"{side}_{f}_")), None)
            pad = g.meshPath.endswith("elastomer.STL")
            solve = None if pad or len(local) <= SOLVE_POINTS else rng.choice(len(local), SOLVE_POINTS, replace=False)
            palmar = axis = None
            if pad:  # the elastomer caps the fingertip; the palm side is +y of the distal joint's frame (flexion about +z)
                axis = np.array([0.0, 1.0, 0.0])
                palmar = np.flatnonzero((local - local.mean(0)) @ axis > PALMAR_OFFSET_M)
            self.geoms.append(Geometry(g.name, int(g.parentJoint), local, finger, pad, solve, palmar, axis))
        self.pads = {g.finger: g for g in self.geoms if g.pad}
        self._q = None

    def q(self, wrist_position: np.ndarray, wrist_wxyz: np.ndarray, fingers: np.ndarray) -> np.ndarray:
        w = np.asarray(wrist_wxyz, np.float64)
        return np.concatenate([np.asarray(wrist_position, np.float64), w[[1, 2, 3, 0]], np.asarray(fingers, np.float64)])

    def update(self, q: np.ndarray) -> None:
        pin = self.pin
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q)
        self._q = q

    def points(self, g: Geometry) -> np.ndarray:
        M = self.data.oMi[g.joint]
        return g.local @ np.asarray(M.rotation).T + np.asarray(M.translation)

    def point_jacobians(self, g: Geometry, world: np.ndarray) -> np.ndarray:
        """(n, 3, n_finger_joints) d(world point)/d(finger joints) at the last update()."""
        pin = self.pin
        J = np.asarray(pin.getJointJacobian(self.model, self.data, g.joint, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))[:, 6:]
        r = world - np.asarray(self.data.oMi[g.joint].translation)
        return point_jacobians(J, r)

    def pad_normal(self, g: Geometry) -> np.ndarray:
        """The pad's palmar normal in the world at the last update()."""
        return np.asarray(self.data.oMi[g.joint].rotation) @ g.axis

    def normal_jacobian(self, g: Geometry, n: np.ndarray) -> np.ndarray:
        """(3, n_finger_joints) d(world direction n fixed in g's joint frame)/d(finger joints): w x n per joint column."""
        pin = self.pin
        J = np.asarray(pin.getJointJacobian(self.model, self.data, g.joint, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))[3:, 6:]
        return np.cross(J.T, n).T

    def frames(self) -> np.ndarray:
        """(n_frames, 7) every model frame's world pose [x y z qw qx qy qz], as NVIDIA's IK stores robot_<side>_frames."""
        from scipy.spatial.transform import Rotation
        out = np.empty((len(self.frame_names), 7))
        for i in range(len(self.frame_names)):
            M = self.data.oMf[i]
            out[i, :3] = np.asarray(M.translation)
            out[i, 3:] = Rotation.from_matrix(np.asarray(M.rotation)).as_quat(scalar_first=True)
        return out


# ---------------------------------------------------------------- the pass
def contact_runs(part_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Per finger, (T,) object index + 1 in contact with its distal MANO link after GAP_CLOSE / MIN_RUN, 0 = none."""
    out = {}
    for f, link in DISTAL_LINK.items():
        lab = np.asarray(part_ids[:, link], np.int64).copy()
        idx = np.flatnonzero(lab > 0)
        for a, b in zip(idx[:-1], idx[1:]):  # close short gaps with the earlier label
            if 1 < b - a <= GAP_CLOSE + 1:
                lab[a + 1:b] = lab[a]
        on = lab > 0
        t = 0
        while t < len(on):
            if on[t]:
                e = t
                while e + 1 < len(on) and on[e + 1]:
                    e += 1
                if e - t + 1 < MIN_RUN:
                    lab[t:e + 1] = 0
                t = e + 1
            else:
                t += 1
        out[f] = lab
    return out


def blend_weights(active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T,) bool contact frames -> (T,) weight in [0, 1] and (T,) index of the contact frame whose correction a ramp frame
    takes (BLEND frames before a run and after it; -1 where no correction)."""
    T = len(active)
    w, src = np.zeros(T), np.full(T, -1)
    w[active], src[active] = 1.0, np.flatnonzero(active)
    for t in np.flatnonzero(active):
        for d in range(1, BLEND + 1):
            for u in (t - d, t + d):
                if 0 <= u < T and not active[u]:
                    wt = 1.0 - d / (BLEND + 1)
                    if wt > w[u]:
                        w[u], src[u] = wt, t
    return w, src


def residuals(hand, q_wrist, x: np.ndarray, x_ik: np.ndarray, x_prev: np.ndarray | None, A: list[int],
              targets: dict, placed: list, moved: list) -> tuple[np.ndarray, np.ndarray]:
    """solve_frame's stacked residuals (m) and their Jacobian w.r.t. the joints A, at finger joints x."""
    hand.update(hand.q(*q_wrist, x))
    rows_r, rows_J = [], []
    for f, (y, n, oi) in targets.items():
        g = hand.pads[f]
        P = hand.points(g)
        if g.palmar is not None and len(g.palmar):
            P = P[g.palmar]
        _, sd, nw = placed[oi].query(P, PAD_NEAR_M)
        if np.isfinite(sd).any():  # the palm-side pad's deepest point onto the object's own surface, along its normal
            s = int(np.argmin(sd))
            normal, offset = nw[s], sd[s]
        else:  # far from the object: towards the human contact point's tangent plane
            s = int(np.argmin((P - y) @ n))
            normal, offset = n, float(n @ (P[s] - y))
        p = P[s:s + 1]
        J = hand.point_jacobians(g, p)[0][:, A]
        rows_r.append(np.array([offset - MARGIN]))
        rows_J.append((normal @ J)[None])
        Tn = np.eye(3) - np.outer(n, n)  # weakly: on the human's contact feature
        rows_r.append(np.sqrt(W_TANGENT) * (Tn @ (p[0] - y)))
        rows_J.append(np.sqrt(W_TANGENT) * (Tn @ J))
        if g.axis is not None and W_NORMAL > 0:  # the pad's palm side against the surface: n_pad = -n_surface
            npad = hand.pad_normal(g)
            rows_r.append(np.array([np.sqrt(W_NORMAL) * (1.0 + npad @ normal)]))
            rows_J.append(np.sqrt(W_NORMAL) * (normal @ hand.normal_jacobian(g, npad)[:, A])[None])
    for g in moved:
        P = hand.points(g)
        if g.solve is not None:
            P = P[g.solve]
        for o in placed:
            _, sd, nw = o.query(P, SOLVE_NEAR_M)
            hit = sd < COLL_MARGIN
            if g.pad:  # the pad's targeted point is held MARGIN outside by the contact term
                hit &= sd < COLL_MARGIN - 1e-4
            if not hit.any():
                continue
            J = hand.point_jacobians(g, P[hit])[:, :, A]
            rows_r.append(np.sqrt(W_COLL) * (sd[hit] - COLL_MARGIN))
            rows_J.append(np.sqrt(W_COLL) * np.einsum("ij,ijk->ik", nw[hit], J))
    rows_r.append(np.sqrt(W_IK) * (x[A] - x_ik[A]))
    rows_J.append(np.sqrt(W_IK) * np.eye(len(A)))
    if x_prev is not None:
        rows_r.append(np.sqrt(W_TIME) * (x[A] - x_prev[A]))
        rows_J.append(np.sqrt(W_TIME) * np.eye(len(A)))
    return np.concatenate(rows_r), np.vstack(rows_J)


def solve_frame(hand, q_wrist: tuple[np.ndarray, np.ndarray], x_ik: np.ndarray, x_prev: np.ndarray | None,
                targets: dict[str, tuple[np.ndarray, np.ndarray]], placed: list) -> tuple[np.ndarray, dict]:
    """One frame: finger joints x (n_finger_joints,) moving only the targeted fingers, by Levenberg-Marquardt (a step is
    taken only if it lowers the cost; the joints stay in their box). targets: finger -> (world contact point, world outward
    normal, index of the labelled object in `placed`). placed: the frame's objects (Placed). x_prev: NVIDIA's joints plus the
    previous frame's correction (W_TIME), or None."""
    A = sorted(k for f in targets for k in hand.finger_dofs[f])
    moved = [g for g in hand.geoms if g.finger in targets]
    lo = np.maximum(hand.lower[A], x_ik[A] - MAX_DELTA)
    hi = np.minimum(hand.upper[A], x_ik[A] + MAX_DELTA)
    x = x_ik.copy()
    if x_prev is not None:
        x[A] = x_prev[A]
    x[A] = np.clip(x[A], lo, hi)
    args = (x_ik, x_prev, A, targets, placed, moved)
    r, J = residuals(hand, q_wrist, x, *args)
    cost, mu, it = float(r @ r), 1e-3, 0
    for it in range(1, ITERS + 1):
        H, grad = J.T @ J, J.T @ r
        for _ in range(10):
            step = np.linalg.solve(H + mu * (np.diag(np.diag(H)) + 1e-12 * np.eye(len(A))), -grad)
            x_new = x.copy()
            x_new[A] = np.clip(x[A] + step, lo, hi)
            r_new, J_new = residuals(hand, q_wrist, x_new, *args)
            if float(r_new @ r_new) < cost:
                break
            mu *= 4.0
        else:
            break  # no step lowers the cost
        moved_by = float(np.abs(x_new[A] - x[A]).max())
        x, r, J, cost, mu = x_new, r_new, J_new, float(r_new @ r_new), max(mu / 3.0, 1e-9)
        if moved_by < 1e-5:
            break
    return x, {"iterations": it, "cost": cost}


def correct_hand(hand, row: dict, side: str, surfaces: list[ObjectSurface]) -> tuple[np.ndarray, dict]:
    """The pass on one hand of one processed row: returns the corrected finger joints (T, n) and a summary."""
    X_ik = np.asarray(row[f"robot_{side}_finger_joints"], np.float64)
    Wp = np.asarray(row[f"robot_{side}_wrist_position"], np.float64)
    Wq = np.asarray(row[f"robot_{side}_wrist_wxyz"], np.float64)
    pid = np.asarray(row[f"mano_{side}_object_contact_part_ids"], np.int64)
    cpos = np.asarray(row[f"mano_{side}_object_contact_positions"], np.float64)
    cnrm = np.asarray(row[f"mano_{side}_object_contact_normals"], np.float64)
    P_obj, Q_obj = np.asarray(row["object_body_position"], np.float64), np.asarray(row["object_body_wxyz"], np.float64)
    T = len(X_ik)
    runs = contact_runs(pid)
    delta = np.zeros_like(X_ik)
    solved, prev = 0, None
    for t in range(T):
        targets = {}
        for f, lab in runs.items():
            link = DISTAL_LINK[f]
            if lab[t] <= 0 or f not in hand.pads:
                continue
            y, n = cpos[t, link], cnrm[t, link]
            if np.linalg.norm(y) < 1e-3 or np.linalg.norm(n) < 0.5:  # the loader zeroes invalid contacts
                continue
            targets[f] = (y, n / np.linalg.norm(n), int(lab[t]) - 1)
        if not targets:
            prev = None
            continue
        placed = [Placed(s, P_obj[t, i], Q_obj[t, i]) for i, s in enumerate(surfaces)]
        x_prev = None if prev is None else X_ik[t] + prev
        x, _ = solve_frame(hand, (Wp[t], Wq[t]), X_ik[t], x_prev, targets, placed)
        delta[t] = x - X_ik[t]
        prev, solved = delta[t], solved + 1
    # ramp in and out per finger, then smooth over time
    out = np.zeros_like(delta)
    for f in FINGERS:
        dofs = hand.finger_dofs.get(f, [])
        if not dofs:
            continue
        active = np.abs(delta[:, dofs]).max(1) > 0
        w, src = blend_weights(active)
        for t in np.flatnonzero(src >= 0):
            out[t, dofs] = w[t] * delta[src[t], dofs]
    if SMOOTH_SIGMA > 0:
        from scipy.ndimage import gaussian_filter1d
        out = gaussian_filter1d(out, SMOOTH_SIGMA, axis=0, mode="nearest")
        out[np.abs(out) < 1e-9] = 0.0
    X = np.clip(X_ik + np.clip(out, -MAX_DELTA, MAX_DELTA), hand.lower, hand.upper)
    changed = np.abs(X - X_ik).max(1) > 1e-7
    return X, {"frames_solved": int(solved), "frames_changed": int(changed.sum()),
               "contact_frames": {f: int((lab > 0).sum()) for f, lab in runs.items()},
               "max_change_deg": round(float(np.degrees(np.abs(X - X_ik).max())), 2) if changed.any() else 0.0}


def wrist_step(hand, row: dict, side: str, X: np.ndarray, surfaces: list[ObjectSurface]) -> np.ndarray:
    """WRIST_STEP (off by default): per corrected frame, the wrist translated (at most MAX_WRIST_M) by the mean normal residual
    of its contact pads; returns the new wrist positions (T, 3)."""
    Wp = np.asarray(row[f"robot_{side}_wrist_position"], np.float64).copy()
    Wq = np.asarray(row[f"robot_{side}_wrist_wxyz"], np.float64)
    pid = np.asarray(row[f"mano_{side}_object_contact_part_ids"], np.int64)
    cpos = np.asarray(row[f"mano_{side}_object_contact_positions"], np.float64)
    cnrm = np.asarray(row[f"mano_{side}_object_contact_normals"], np.float64)
    runs = contact_runs(pid)
    for t in range(len(X)):
        res = []
        for f, lab in runs.items():
            link = DISTAL_LINK[f]
            if lab[t] <= 0 or f not in hand.pads or np.linalg.norm(cnrm[t, link]) < 0.5:
                continue
            hand.update(hand.q(Wp[t], Wq[t], X[t]))
            n = cnrm[t, link] / np.linalg.norm(cnrm[t, link])
            P = hand.points(hand.pads[f])
            res.append(-(min((P - cpos[t, link]) @ n) - MARGIN) * n)
        if res:
            step = np.mean(res, axis=0)
            norm = np.linalg.norm(step)
            Wp[t] += step * min(1.0, MAX_WRIST_M / norm) if norm > 0 else 0.0
    return Wp


# ---------------------------------------------------------------- metrics
def metrics(row: dict, hands: dict, surfaces: list[ObjectSurface], fingers: dict | None = None) -> dict:
    """Pad gaps (unsigned, the pad's nearest vertex to the labelled object) on the frames where the reference labels the
    finger's distal link (raw labels); robot penetration of every collision point of the hand into every object on the
    contact frames (exact count where the mesh is watertight, else UNKNOWN; approximate depth from the sampled normals);
    finger joint steps and accelerations over the clip. `fingers` overrides robot_<side>_finger_joints (side -> (T, n))."""
    P_obj, Q_obj = np.asarray(row["object_body_position"], np.float64), np.asarray(row["object_body_wxyz"], np.float64)
    out = {}
    for side, hand in hands.items():
        X = np.asarray(fingers[side] if fingers else row[f"robot_{side}_finger_joints"], np.float64)
        Wp = np.asarray(row[f"robot_{side}_wrist_position"], np.float64)
        Wq = np.asarray(row[f"robot_{side}_wrist_wxyz"], np.float64)
        pid = np.asarray(row[f"mano_{side}_object_contact_part_ids"], np.int64)
        gaps = {f: [] for f in FINGERS}
        contact_t = np.flatnonzero((pid[:, list(DISTAL_LINK.values())] > 0).any(1))
        pen = {"all": [0, 0.0, 0], "fingers": [0, 0.0, 0]}  # exact inside points, approx max depth (m), frames over 2 mm
        exact_known = all(s.watertight for s in surfaces)
        for t in contact_t:
            hand.update(hand.q(Wp[t], Wq[t], X[t]))
            placed = [Placed(s, P_obj[t, i], Q_obj[t, i]) for i, s in enumerate(surfaces)]
            for f, link in DISTAL_LINK.items():
                if pid[t, link] > 0 and f in hand.pads:
                    o, g = placed[pid[t, link] - 1], hand.pads[f]
                    P = hand.points(g)
                    d, sd, nw = o.query(P)
                    inside = o.surface.inside(o.to_object(P)) if o.surface.watertight else sd < 0
                    signed = np.where(inside, -d, d)
                    pal = g.palmar if g.palmar is not None and len(g.palmar) else np.arange(len(P))
                    s = pal[int(np.argmin(sd[pal]))]  # the palm side's deepest point: does the palm side face the surface?
                    facing = float(hand.pad_normal(g) @ -nw[s]) if g.axis is not None else np.nan
                    gaps[f].append((float(d.min()), float(signed.min()), float(signed[pal].min()), facing))
            deepest = {"all": 0.0, "fingers": 0.0}
            for g in hand.geoms:
                P = hand.points(g)
                for o in placed:
                    d, sd, _ = o.query(P, METRIC_NEAR_M)
                    n_in = 0
                    if exact_known:  # the exact ray test near the surface; farther points keep the sampled normal's sign
                        near = d < EXACT_NEAR_M
                        n_in = int(o.surface.inside(o.to_object(P[near])).sum()) + int((sd[~near] < 0).sum()) if near.any() \
                            else int((sd < 0).sum())
                    for k in ("all", "fingers") if g.finger else ("all",):
                        deepest[k] = max(deepest[k], float(-sd.min()))
                        pen[k][0] += n_in
            for k, v in pen.items():
                v[1] = max(v[1], deepest[k])
                v[2] += int(deepest[k] > 0.002)
        steps = np.abs(np.diff(X, axis=0))
        acc = np.abs(np.diff(X, n=2, axis=0))
        e = {"contact_frames": int(len(contact_t))}
        for k, (n_in, depth, over) in pen.items():
            e[f"penetration_{k}_points_exact"] = n_in if exact_known else "UNKNOWN (mesh not watertight)"
            e[f"penetration_{k}_approx_max_mm"] = round(depth * 1e3, 2)
            e[f"penetration_{k}_approx_frames_over_2mm"] = over
        e.update({
                  "joint_step_max_deg": round(float(np.degrees(steps.max())), 2) if len(steps) else 0.0,
                  "joint_step_p99_deg": round(float(np.degrees(np.percentile(steps.max(1), 99))), 2) if len(steps) else 0.0,
                  "joint_acc_max_deg": round(float(np.degrees(acc.max())), 2) if len(acc) else 0.0})
        for f, g in gaps.items():
            if g:
                g, s, sp, face = np.asarray(g).T  # unsigned gap; signed (inside < 0: exact where watertight); palm side's; cos
                g, s, sp = g * 1e3, s * 1e3, sp * 1e3
                faces = face > FACING_COS
                valid = faces & (sp >= -TOUCH_IN_M * 1e3) & (sp <= GAP_NEAR_M * 1e3)
                e[f] = {"frames": len(g), "pad_gap_median_mm": round(float(np.median(g)), 1),
                        "share_palm_facing": round(float(faces.mean()), 3), "share_valid_contact": round(float(valid.mean()), 3),
                        "pad_gap_p90_mm": round(float(np.percentile(g, 90)), 1),
                        f"share_lt_{GAP_NEAR_M * 1e3:.0f}mm": round(float((g < GAP_NEAR_M * 1e3).mean()), 3),
                        "pad_signed_median_mm": round(float(np.median(s)), 1),
                        "share_touching": round(float(((s >= -TOUCH_IN_M * 1e3) & (s <= GAP_NEAR_M * 1e3)).mean()), 3),
                        "share_penetrating": round(float((s < -TOUCH_IN_M * 1e3).mean()), 3)}
        out[side] = e
    return out


# ---------------------------------------------------------------- the processed Parquet
def surfaces_for(row: dict, processed: Path) -> list[ObjectSurface]:
    """The scene's object meshes: processed/<body>.obj (NVIDIA's metric mesh per body), else the Parquet's mesh path."""
    import trimesh
    out = []
    for body, path in zip(row["object_body_names"], row["object_mesh_paths"]):
        local = Path(processed) / f"{body}.obj"
        out.append(ObjectSurface(trimesh.load(local if local.is_file() else path, force="mesh")))
    return out


def frames_for(hand, wrist_position: np.ndarray, wrist_wxyz: np.ndarray, X: np.ndarray, old: np.ndarray,
               changed: np.ndarray) -> np.ndarray:
    """robot_<side>_frames with the changed frames recomputed by FK."""
    F = np.array(old, np.float64)
    for t in np.flatnonzero(changed):
        hand.update(hand.q(wrist_position[t], wrist_wxyz[t], X[t]))
        F[t] = hand.frames()
    return F


def apply(partition: Path, out: Path, xml_dir: Path) -> dict:
    """The pass on every Parquet of a processed partition (one row each); keeps the originals under
    <out>/intermediate/contact_retarget/ and starts from them on a rerun; writes <out>/contact_retarget.json."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    keep = Path(out) / "intermediate/contact_retarget"
    keep.mkdir(parents=True, exist_ok=True)
    hands = {s: SharpaHand(s, Path(xml_dir) / f"{s}_sharpawave.xml") for s in ("right", "left")}
    report = {"settings": {k: globals()[k] for k in ("GAP_CLOSE", "MIN_RUN", "MARGIN", "COLL_MARGIN", "MAX_DELTA", "W_TANGENT",
                                                       "W_COLL", "W_IK", "W_TIME", "ITERS", "BLEND", "SMOOTH_SIGMA",
                                                       "WRIST_STEP", "MAX_WRIST_M")}, "files": {}}
    for f in sorted(Path(partition).glob("*.parquet")):
        orig = keep / f.name
        if not orig.is_file():
            shutil.copy2(f, orig)
        table = pq.read_table(orig)
        assert table.num_rows == 1, (f, table.num_rows)
        row = table.to_pylist()[0]
        surfaces = surfaces_for(row, Path(partition).parents[1])
        before = metrics(row, hands, surfaces)
        new, summary = {}, {}
        for side, hand in hands.items():
            X, summary[side] = correct_hand(hand, row, side, surfaces)
            Wp_old = np.asarray(row[f"robot_{side}_wrist_position"], np.float64)
            Wp = wrist_step(hand, row, side, X, surfaces) if WRIST_STEP else Wp_old
            X_old = np.asarray(row[f"robot_{side}_finger_joints"], np.float64)
            changed = (np.abs(X - X_old).max(1) > 1e-7) | (np.abs(Wp - Wp_old).max(1) > 1e-9)
            new[f"robot_{side}_finger_joints"] = X
            if WRIST_STEP:
                new[f"robot_{side}_wrist_position"] = Wp
            else:
                assert np.array_equal(Wp, Wp_old)
            new[f"robot_{side}_frames"] = frames_for(hand, Wp, np.asarray(row[f"robot_{side}_wrist_wxyz"], np.float64), X,
                                                     row[f"robot_{side}_frames"], changed)
        for key, value in new.items():
            i = table.schema.get_field_index(key)
            table = table.set_column(i, table.schema.field(i), pa.array([value.tolist()], type=table.schema.field(i).type))
        pq.write_table(table, f)
        after_row = dict(row, **{k: v.tolist() for k, v in new.items()})
        report["files"][f.name] = {"summary": summary, "before": before, "after": metrics(after_row, hands, surfaces)}
    (Path(out) / "contact_retarget.json").write_text(json.dumps(report, indent=1) + "\n")
    return report
