set -u; D=/data/user_data/haisenk/v2d/refine1104/ep6_d2; W=$D/work; hostname; date
export CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cat > $W/kit_AUC.py << '__KIT_EOF__'
"""Kaggle metric scoring code. Reference arrays and embedded model or vertex assets are omitted from this participant copy."""

import base64
import io
import re

import numpy as np
import pandas as pd
import pandas.api.types


class ParticipantVisibleError(Exception):
    pass


                                                                               
                                                                    
def _mesh_faces(faces: np.ndarray, vertex_count: int) -> np.ndarray:

    rounded = np.rint(faces)
    if np.abs(faces - rounded).max() > 1e-6:
        raise ValueError("object_mesh_faces must hold integer vertex indices")
    rounded = rounded.astype(np.int64)
    if rounded.min() < 0 or rounded.max() >= vertex_count:
        raise ValueError(f"object_mesh_faces index outside [0, {vertex_count})")
    return rounded


def _sample_mesh_surface(vertices: np.ndarray, faces: np.ndarray, count: int,
                         seed: int) -> np.ndarray:

    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = area.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("object mesh has no surface area")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(faces), size=count, p=area / total)
    r1, r2 = rng.random(count), rng.random(count)
    s = np.sqrt(r1)[:, None]
    return (1 - s) * a[chosen] + s * (1 - r2[:, None]) * b[chosen] + s * r2[:, None] * c[chosen]


                                                                               
                                                                    
                                       
_METRIC_SAMPLES = 10_000
_ALIGNMENT_SAMPLES = 3_000
_ICP_ITERATIONS = 40
_ALIGNMENT_STARTS = 25
_TRIM_FRACTION = 0.9


def _to_meshes(block, value_columns: list[str]) -> dict[int, tuple[np.ndarray, np.ndarray]]:

    meshes = {}
    for slot, rows in block.groupby("object_slot", sort=True):
        parts = {}
        for name in ("object_mesh_vertices", "object_mesh_faces"):
            part = rows[rows["array_name"] == name].sort_values("point_index")
            if part.empty:
                raise ValueError(f"object slot {slot}: no rows for {name!r}")
            if not np.array_equal(part["point_index"].to_numpy(), np.arange(len(part))):
                raise ValueError(f"object slot {slot}: {name} point_index must run 0..{len(part) - 1}")
            parts[name] = part[value_columns].to_numpy(dtype=np.float64)
        vertices = parts["object_mesh_vertices"]
        meshes[int(slot)] = (vertices, _mesh_faces(parts["object_mesh_faces"], len(vertices)))
    return meshes


def _pca_basis(points):
    centered = points - points.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered / max(len(centered), 1))
    basis = vectors[:, ::-1]
    if np.linalg.det(basis) < 0.0:
        basis[:, -1] *= -1.0
    return basis


def _initial_rotations(reference, candidate, maximum):

    import itertools

    ref_basis, cand_basis = _pca_basis(reference), _pca_basis(candidate)
    rotations = [np.eye(3)]
    seen = {tuple(np.round(rotations[0], 10).ravel())}
    for permutation in itertools.permutations(range(3)):
        permuted = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            mapping = permuted @ np.diag(signs)
            if np.linalg.det(mapping) <= 0.0:
                continue
            rotation = ref_basis @ mapping @ cand_basis.T
            if np.linalg.det(rotation) < 0.0:
                continue
            key = tuple(np.round(rotation, 10).ravel())
            if key in seen:
                continue
            seen.add(key)
            rotations.append(rotation)
            if len(rotations) >= maximum:
                return rotations
    return rotations


def _kabsch(source, target):
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    left, _, right_t = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1, :] *= -1.0
        rotation = right_t.T @ left.T
    return rotation, target_center - rotation @ source_center


def _symmetric_mean_distance(reference, candidate):
    from scipy.spatial import cKDTree

    to_reference = cKDTree(reference).query(candidate, k=1)[0]
    to_candidate = cKDTree(candidate).query(reference, k=1)[0]
    return 0.5 * (float(np.mean(to_reference)) + float(np.mean(to_candidate)))


def _align_rigid(reference, candidate):

    import math

    from scipy.spatial import cKDTree

    tree = cKDTree(reference)
    best = None
    for start in _initial_rotations(reference, candidate, _ALIGNMENT_STARTS):
        rotation = start.copy()
        translation = reference.mean(axis=0) - rotation @ candidate.mean(axis=0)
        previous = math.inf
        for _ in range(_ICP_ITERATIONS):
            moved = candidate @ rotation.T + translation
            distances, indices = tree.query(moved, k=1)
            keep = max(3, int(math.ceil(len(distances) * _TRIM_FRACTION)))
            chosen = (np.argpartition(distances, keep - 1)[:keep] if keep < len(distances)
                      else np.arange(len(distances)))
            step_rotation, step_translation = _kabsch(moved[chosen], reference[indices[chosen]])
            rotation = step_rotation @ rotation
            translation = step_rotation @ translation + step_translation
            error = float(np.mean(distances[chosen]))
            if math.isfinite(previous) and abs(previous - error) <= 1e-8 * max(previous, 1.0):
                break
            previous = error
        score = _symmetric_mean_distance(reference, candidate @ rotation.T + translation)
        if best is None or score < best[2]:
            best = (rotation, translation, score)
    return best[0], best[1]


def _surface_moments(vertices, faces):

    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                          triangles[:, 2] - triangles[:, 0]), axis=1)
    keep = np.isfinite(areas) & (areas > 1e-12)
    if not keep.any():
        raise ValueError("object mesh has no surface area")
    triangles, areas = triangles[keep], areas[keep]
    total = float(areas.sum())
    centroid = (triangles.mean(axis=1) * areas[:, None]).sum(axis=0) / total
    centered = triangles - centroid
    squared = np.einsum("tij,tij->ti", centered, centered).sum(axis=1)
    cross = (np.einsum("ti,ti->t", centered[:, 0], centered[:, 1])
             + np.einsum("ti,ti->t", centered[:, 0], centered[:, 2])
             + np.einsum("ti,ti->t", centered[:, 1], centered[:, 2]))
    radius = float(np.sqrt(max(float((areas * (squared + cross) / 6.0).sum() / total), 0.0)))
    if not np.isfinite(radius) or radius <= 1e-12:
        raise ValueError("object mesh has zero or invalid extent")
    return centroid, radius


def _object_chamfer_cm(reference_mesh, candidate_mesh, seed: int) -> float:

    reference_v, reference_f = reference_mesh
    candidate_v, candidate_f = candidate_mesh
    _, reference_radius = _surface_moments(reference_v, reference_f)
    candidate_center, candidate_radius = _surface_moments(candidate_v, candidate_f)
    ratio = candidate_radius / reference_radius
    candidate_v = (candidate_v - candidate_center) / ratio + candidate_center
    reference_metric = _sample_mesh_surface(reference_v, reference_f, _METRIC_SAMPLES, seed)
    candidate_metric = _sample_mesh_surface(candidate_v, candidate_f, _METRIC_SAMPLES, seed)
    reference_align = _sample_mesh_surface(reference_v, reference_f, _ALIGNMENT_SAMPLES, seed + 1)
    candidate_align = _sample_mesh_surface(candidate_v, candidate_f, _ALIGNMENT_SAMPLES, seed + 1)
    rotation, translation = _align_rigid(reference_align, candidate_align)
    return 100.0 * _symmetric_mean_distance(reference_metric,
                                            candidate_metric @ rotation.T + translation)


def _object_seed(episode: int, slot: int) -> int:

    return 1_000_003 * int(episode) + 101 * int(slot)


                                                                               
                                                                    
                                               
_ADD_THRESHOLDS_M = np.arange(0.01, 0.10, 0.01)
_SPIDER_POSITION_THRESHOLD_M = 0.1
_SPIDER_ORIENTATION_THRESHOLD_RAD = 0.5
_MANIPTRANS_POSITION_THRESHOLD_M = 0.03
_MANIPTRANS_ORIENTATION_THRESHOLD_DEG = 30.0
                                                                           
                                                                                              
                                                                                                 
                                                                                                
                                                                                           
                                                                               
_MPPE_KEYPOINT_RADIUS_M = 0.05

                                                                                            
                                                                                            
_MPPE_KEYPOINT_DIRECTIONS = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
     [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float64,
)

                                                         
_POSE_COLUMNS = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
                                                                          
_REF_COLUMNS = ["ref_" + _c for _c in _POSE_COLUMNS]


                                                                                        

def _rotation_matrix(quat_xyzw: np.ndarray) -> np.ndarray:

    x, y, z, w = (quat_xyzw[..., index] for index in range(4))
    return np.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(*quat_xyzw.shape[:-1], 3, 3)


def _quaternion_geodesic_angle(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> np.ndarray:

                                                                           
    dot = np.clip(np.abs(np.sum(a_xyzw * b_xyzw, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arctan2(np.sqrt(np.maximum(0.0, 1.0 - dot * dot)), dot)


def _object_add(achieved: np.ndarray, reference: np.ndarray, vertices: np.ndarray) -> np.ndarray:

    delta_rotation = _rotation_matrix(achieved[..., 3:]) - _rotation_matrix(reference[..., 3:])
    delta_position = achieved[..., :3] - reference[..., :3]
    steps, worlds, bodies, _ = achieved.shape
    add = np.empty((steps, worlds, bodies), dtype=np.float64)
    for body in range(bodies):
        offsets = np.einsum("twij,vj->twvi", delta_rotation[:, :, body], vertices[body])
        offsets += delta_position[:, :, body, None, :]
        add[:, :, body] = np.linalg.norm(offsets, axis=-1).mean(axis=-1)
    return add


def _add_auc(add: np.ndarray) -> float:

    axis = np.linspace(0.0, 1.0, _ADD_THRESHOLDS_M.size)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    per_body = [
        float(trapezoid(np.array([np.mean(add[..., body] < t) for t in _ADD_THRESHOLDS_M]), x=axis))
        for body in range(add.shape[2])
    ]
    return float(np.mean(per_body))


def _quaternion_multiply(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> np.ndarray:

    ax, ay, az, aw = (a_xyzw[..., i] for i in range(4))
    bx, by, bz, bw = (b_xyzw[..., i] for i in range(4))
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def _warp_to_reference_start(achieved: np.ndarray, reference: np.ndarray) -> np.ndarray:

    reference_start = reference[0, 0, 0]                                                      
    achieved_start = achieved[0, :, 0]                                

    conjugate = achieved_start[:, 3:] * np.array([-1.0, -1.0, -1.0, 1.0])
    q_align = _quaternion_multiply(reference_start[None, 3:], conjugate)                  
    q_align = q_align / np.linalg.norm(q_align, axis=-1, keepdims=True)
    rotation = _rotation_matrix(q_align)                                                     
    offset = reference_start[None, :3] - np.einsum("wij,wj->wi", rotation, achieved_start[:, :3])

    warped = np.empty_like(achieved)
    warped[..., :3] = np.einsum("wij,twbj->twbi", rotation, achieved[..., :3]) + offset[None, :, None, :]
    warped[..., 3:] = _quaternion_multiply(q_align[None, :, None, :], achieved[..., 3:])
    return warped


def _per_world_pose_error(achieved: np.ndarray, reference: np.ndarray):

    position = np.linalg.norm(achieved[..., :3] - reference[..., :3], axis=-1).mean(axis=0)
    orientation = _quaternion_geodesic_angle(achieved[..., 3:], reference[..., 3:]).mean(axis=0)
    return position, orientation


def _spider_success(achieved: np.ndarray, reference: np.ndarray) -> np.ndarray:

    position, orientation = _per_world_pose_error(achieved, reference)
    return (position.mean(axis=-1) <= _SPIDER_POSITION_THRESHOLD_M) & (
        orientation.mean(axis=-1) <= _SPIDER_ORIENTATION_THRESHOLD_RAD
    )


def _maniptrans_success(achieved: np.ndarray, reference: np.ndarray, object_ids: np.ndarray) -> np.ndarray:

    position, orientation = _per_world_pose_error(achieved, reference)
    orientation_deg = np.degrees(orientation)
    success = np.ones(achieved.shape[1], dtype=np.bool_)
    for object_id in np.unique(object_ids):
        members = object_ids == object_id
        success &= (position[:, members].mean(axis=-1) < _MANIPTRANS_POSITION_THRESHOLD_M) & (
            orientation_deg[:, members].mean(axis=-1) < _MANIPTRANS_ORIENTATION_THRESHOLD_DEG
        )
    return success


def _relative_position_error_cm(achieved: np.ndarray, reference: np.ndarray) -> float:

    bodies = achieved.shape[2]
    if bodies < 2:
        return 0.0
    errors = []
    for first in range(bodies):
        for second in range(first + 1, bodies):
            rotation_a = _rotation_matrix(achieved[:, :, first, 3:])
            rotation_r = _rotation_matrix(reference[:, :, first, 3:])
            delta_a = achieved[:, :, second, :3] - achieved[:, :, first, :3]
            delta_r = reference[:, :, second, :3] - reference[:, :, first, :3]
            local_a = np.einsum("twji,twj->twi", rotation_a, delta_a)
            local_r = np.einsum("twji,twj->twi", rotation_r, delta_r)
            errors.append(np.linalg.norm(local_a - local_r, axis=-1))
    return float(np.mean(np.stack(errors)) * 100.0)


def _mean_per_point_error_cm(achieved: np.ndarray, reference: np.ndarray) -> float:

    offsets = _MPPE_KEYPOINT_DIRECTIONS * _MPPE_KEYPOINT_RADIUS_M             
    rotation_a = _rotation_matrix(achieved[..., 3:])                                   
    rotation_r = _rotation_matrix(reference[..., 3:])                                  
                                                                                      
    keypoints_a = achieved[..., None, :3] + np.einsum("...ij,kj->...ki", rotation_a, offsets)
    keypoints_r = reference[..., None, :3] + np.einsum("...ij,kj->...ki", rotation_r, offsets)
    return float(np.linalg.norm(keypoints_a - keypoints_r, axis=-1).mean() * 100.0)


                                                                                        

def _episode_tensors(frame: pd.DataFrame):

    frame = frame.sort_values(["frame_index", "world_index", "object_slot"], kind="mergesort")
    frames = np.unique(frame["frame_index"].to_numpy())
    worlds = np.unique(frame["world_index"].to_numpy())
    slots = np.unique(frame["object_slot"].to_numpy())
    expected = len(frames) * len(worlds) * len(slots)
    if len(frame) != expected:
        raise ValueError(
            f"episode grid is not rectangular: {len(frame)} rows for "
            f"{len(frames)} frames x {len(worlds)} worlds x {len(slots)} objects"
        )
    shape = (len(frames), len(worlds), len(slots))
    achieved = frame[_POSE_COLUMNS].to_numpy(dtype=np.float64).reshape(*shape, 7)
    reference = frame[_REF_COLUMNS].to_numpy(dtype=np.float64).reshape(*shape, 7)
                                                                               
    names = frame["object_name"].to_numpy().reshape(*shape)[0, 0]
    return achieved, reference[:, :1], slots.astype(np.int64), names


def _normalise_quaternions(pose: np.ndarray, who: str, error_type) -> np.ndarray:

    pose = pose.copy()
    norms = np.linalg.norm(pose[..., 3:], axis=-1)
    if np.any(norms < 1e-8):
        raise error_type(f"{who} contains a degenerate (near-zero-norm) quaternion")
    pose[..., 3:] /= norms[..., None]
    return pose


                                                                               
                                                                    
_MESH_PAYLOAD_COLUMN = "object_mesh_b64"
_HOST_PAYLOAD_COLUMN = "reference_geometry_b64"
_MESH_VERTEX_BUDGET = 4096
_MESH_FACE_BUDGET = 4096
_SYMMETRY_ANGLES = 360


def _encode_archive(**arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _decode_archive(text, maximum_bytes=2_000_000):
    import zipfile
    if not isinstance(text, str) or len(text) > maximum_bytes * 2:
        raise ValueError("mesh payload is missing or exceeds its size limit")
    raw = base64.b64decode(text, validate=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if sum(item.file_size for item in archive.infolist()) > maximum_bytes:
            raise ValueError("mesh payload exceeds its decoded size limit")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _decode_object_mesh(text):
    arrays = _decode_archive(text, 400_000)
    if set(arrays) != {"vertices", "faces"}:
        raise ValueError("mesh payload must contain vertices and faces")
    vertices = np.asarray(arrays["vertices"], dtype=np.float64)
    faces = np.asarray(arrays["faces"], dtype=np.float64)
    if vertices.shape != (_MESH_VERTEX_BUDGET, 3) or faces.shape != (_MESH_FACE_BUDGET, 3):
        raise ValueError("object mesh must contain 4096 vertex rows and 4096 face rows")
    if not np.isfinite(vertices).all() or not np.isfinite(faces).all():
        raise ValueError("object mesh contains a non-finite value")
    faces = _mesh_faces(faces, len(vertices))
    _surface_moments(vertices, faces)
    return vertices, faces


def _register_object_body(candidate, reference, anchor, seed):

    cv, cf = candidate
    rv, rf = reference
    cc, cr = _surface_moments(cv, cf)
    rc, rr = _surface_moments(rv, rf)
    scale = rr / cr
    cp = (_sample_mesh_surface(cv, cf, 3000, seed) - cc) * scale
    rp = _sample_mesh_surface(rv, rf, 3000, seed) - rc
    rotation, offset = _align_rigid(rp, cp)
    candidate_anchor = cc + rotation.T @ (anchor - rc - offset) / scale
    return candidate_anchor, rotation


def _convert_object_body(poses, candidate_anchor, body_rotation):
    from scipy.spatial.transform import Rotation
    result = poses.copy()
    rotation = _rotation_matrix(poses[..., 3:])
    result[..., :3] += np.einsum("...ij,j->...i", rotation, candidate_anchor)
    result[..., 3:] = Rotation.from_matrix((rotation @ body_rotation.T).reshape(-1, 3, 3)).as_quat().reshape(poses[..., 3:].shape)
    return result


def _axis_basis(axis, direction):
    perpendicular = direction - axis * np.dot(axis, direction)
    if np.linalg.norm(perpendicular) < 1e-8:
        perpendicular = np.eye(3)[np.argmin(np.abs(axis))]
        perpendicular -= axis * np.dot(axis, perpendicular)
    perpendicular /= np.linalg.norm(perpendicular)
    return np.column_stack((perpendicular, np.cross(axis, perpendicular), axis))


def _align_registered_scene(achieved, reference, axes):

    from scipy.spatial.transform import Rotation
    result = achieved.copy()
    fixed = [i for i, axis in enumerate(axes) if not np.any(axis)]
    for world in range(achieved.shape[1]):
        if fixed or achieved.shape[2] == 1:
            anchor = fixed[0] if fixed else 0
            qa = achieved[0, world, anchor, 3:]
            qr = reference[0, 0, anchor, 3:]
            quaternion = _quaternion_multiply(qr, qa * [-1, -1, -1, 1])
            rotation = _rotation_matrix(quaternion)
        else:
            ar = _rotation_matrix(reference[0, 0, 0, 3:]) @ axes[0]
            aa = _rotation_matrix(achieved[0, world, 0, 3:]) @ axes[0]
            dr = reference[0, 0, 1, :3] - reference[0, 0, 0, :3]
            da = achieved[0, world, 1, :3] - achieved[0, world, 0, :3]
            rotation = _axis_basis(ar, dr) @ _axis_basis(aa, da).T
            quaternion = Rotation.from_matrix(rotation).as_quat()
        offset = reference[0, 0, 0, :3] - rotation @ achieved[0, world, 0, :3]
        result[:, world, :, :3] = achieved[:, world, :, :3] @ rotation.T + offset
        result[:, world, :, 3:] = _quaternion_multiply(quaternion, achieved[:, world, :, 3:])
    return result


def _axial_rotations(axis, order=_SYMMETRY_ANGLES):
    angles = np.arange(order) * (2 * np.pi / order)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (np.cos(angles)[:, None, None] * np.eye(3)
            + (1 - np.cos(angles))[:, None, None] * np.outer(axis, axis)
            + np.sin(angles)[:, None, None] * skew)


def _symmetry_point_errors(achieved, reference, points, axis, order=360):

    ra, rr = _rotation_matrix(achieved[..., 3:]), _rotation_matrix(reference[..., 3:])
    rotations = _axial_rotations(axis, order) if np.any(axis) else np.eye(3)[None]
    best = np.full(achieved.shape[:-1], np.inf)
    for start in range(0, len(achieved), 32):
        end = start + 32
        moved = np.einsum("...ij,vj->...vi", ra[start:end], points) + achieved[start:end, ..., None, :3]
        block = best[start:end]
        for symmetry in rotations:
            equivalent = points @ symmetry.T
            target = np.einsum("...ij,vj->...vi", rr[start:end], equivalent) + reference[start:end, ..., None, :3]
            block[:] = np.minimum(block, np.linalg.norm(moved - target, axis=-1).mean(axis=-1))
    return best


def _symmetry_orientation_errors(achieved, reference, axes, orders):
    errors = _quaternion_geodesic_angle(achieved[..., 3:], reference[..., 3:])
    for body, axis in enumerate(axes):
        if orders[body] == 2:
            from scipy.spatial.transform import Rotation
            symmetry = Rotation.from_rotvec(np.pi * axis).as_quat()
            alternate = _quaternion_multiply(reference[:, :, body, 3:], symmetry)
            errors[:, :, body] = np.minimum(errors[:, :, body],
                _quaternion_geodesic_angle(achieved[:, :, body, 3:], alternate))
        elif np.any(axis):
            aa = _rotation_matrix(achieved[:, :, body, 3:]) @ axis
            ar = _rotation_matrix(reference[:, :, body, 3:]) @ axis
            errors[:, :, body] = np.arctan2(np.linalg.norm(np.cross(aa, ar), axis=-1),
                                           np.clip(np.sum(aa * ar, axis=-1), -1, 1))
    return errors


def _symmetry_relative_position_cm(achieved, reference, axes, orders):
    errors = []
    for first in range(achieved.shape[2]):
        ra = _rotation_matrix(achieved[:, :, first, 3:])
        rr = _rotation_matrix(reference[:, :, first, 3:])
        for second in range(first + 1, achieved.shape[2]):
            da = achieved[:, :, second, :3] - achieved[:, :, first, :3]
            dr = reference[:, :, second, :3] - reference[:, :, first, :3]
            la = np.einsum("...ji,...j->...i", ra, da)
            lr = np.einsum("...ji,...j->...i", rr, dr)
            axis = axes[first]
            if orders[first] == 2:
                alternate = lr @ _axial_rotations(axis, 2)[1]
                error = np.minimum(np.linalg.norm(la - lr, axis=-1),
                                   np.linalg.norm(la - alternate, axis=-1))
            elif np.any(axis):
                ha, hr = la @ axis, lr @ axis
                pa = np.linalg.norm(la - ha[..., None] * axis, axis=-1)
                pr = np.linalg.norm(lr - hr[..., None] * axis, axis=-1)
                error = np.hypot(ha - hr, pa - pr)
            else:
                error = np.linalg.norm(la - lr, axis=-1)
            errors.append(error)
    return float(np.mean(errors) * 100) if errors else 0.0


def _registered_pose_metric(metric, achieved, reference, vertices, axes, orders):
    if metric == "add_auc":
        values = [_symmetry_point_errors(achieved[:, :, i], reference[:, :, i], v, axes[i], orders[i])
                  for i, v in enumerate(vertices)]
        return _add_auc(np.stack(values, axis=-1))
    if metric in ("spider_sr", "maniptrans_sr"):
        position = np.linalg.norm(achieved[..., :3] - reference[..., :3], axis=-1).mean(axis=0)
        orientation = _symmetry_orientation_errors(achieved, reference, axes, orders).mean(axis=0)
        if metric == "spider_sr":
            return float(np.mean((position.mean(axis=-1) <= 0.1) & (orientation.mean(axis=-1) <= 0.5)))
        return float(np.mean(np.all((position < 0.03) & (orientation < np.deg2rad(30)), axis=-1)))
    if metric == "mppe_cm":
        points = _MPPE_KEYPOINT_DIRECTIONS * _MPPE_KEYPOINT_RADIUS_M
        errors = [_symmetry_point_errors(achieved[:, :, i], reference[:, :, i], points, axis, orders[i])
                  for i, axis in enumerate(axes)]
        return float(np.mean(errors) * 100)
    if metric == "rpe_cm":
        return _symmetry_relative_position_cm(achieved, reference, axes, orders)
    raise ValueError(f"unknown pose metric {metric!r}")


                                                                                        
_TRACK = "track_3"
_ALIGN_INITIAL_POSE = True
_REGISTER_OBJECT_MESH = True
_METRIC = "add_auc"
_METRIC_DISPLAY = "AUC"
_ACCEPTED = {'eval_script_sha256': [], 'eval_code_sha256': []}
                                                                                        

_PROVENANCE_COLUMNS = ['eval_script_sha256', 'eval_code_sha256', 'checkpoint_sha256', 'packer_script_sha256', 'uploader_script_sha256', 'code_commit_url']
_UNIFORM_PROVENANCE_COLUMNS = ['eval_script_sha256', 'eval_code_sha256', 'packer_script_sha256', 'uploader_script_sha256', 'code_commit_url']
_PER_EPISODE_PROVENANCE_COLUMNS = ['checkpoint_sha256']
_CODE_COMMIT_PATTERN = '^https://[\\w.-]*github[\\w.-]*/[\\w.-]+/[\\w.-]+/commit/[0-9a-fA-F]{7,40}/?$'
_REQUIRED_SUBMISSION_COLUMNS = _POSE_COLUMNS + _PROVENANCE_COLUMNS
_REQUIRED_SOLUTION_COLUMNS = (
    ["episode_index", "world_index", "frame_index", "object_slot", "object_name"] + _REF_COLUMNS
)
_USAGE_COLUMN = "Usage"

                                                                                         
                                                                                
_VERTICES_B64 = ""


def _load_vertices(solution: pd.DataFrame | None = None) -> dict:

    blob = _VERTICES_B64.strip()
    if not blob and solution is not None and _METRIC == "add_auc":
        column = "auc_vertex_samples_b64"
        if column not in solution.columns:
            raise ValueError(f"solution is missing '{column}'")
        nonempty = solution[column].dropna().astype(str).str.strip()
        nonempty = nonempty[nonempty != ""]
        if len(nonempty) != 1:
            raise ValueError(f"solution must carry exactly one '{column}' payload")
        blob = nonempty.iloc[0]
    if not blob:
        return {}
    try:
        with np.load(io.BytesIO(base64.b64decode("".join(blob.split()), validate=True))) as archive:
            return {name: archive[name].astype(np.float64) for name in archive.files}
    except Exception as error:
        raise ValueError("solution or metric has invalid AUC vertex payload") from error


_VERTICES = _load_vertices()


                                                                                         

def _check_solution(solution: pd.DataFrame, row_id_column_name: str) -> None:
    if row_id_column_name not in solution.columns:
        raise ValueError(f"solution is missing the '{row_id_column_name}' row-id column")
    missing = [c for c in _REQUIRED_SOLUTION_COLUMNS if c not in solution.columns]
    if missing:
        raise ValueError(f"solution is missing required column(s): {missing}")
    if len(solution) == 0:
        raise ValueError("solution has no rows")
    if solution[row_id_column_name].duplicated().any():
        raise ValueError(f"solution has duplicate '{row_id_column_name}' values")
    if solution[row_id_column_name].isnull().any():
        raise ValueError(f"solution has null '{row_id_column_name}' values")
    for column in ["episode_index", "world_index", "frame_index", "object_slot"]:
        if not pandas.api.types.is_numeric_dtype(solution[column]) or solution[column].isnull().any():
            raise ValueError(f"solution column '{column}' must be numeric and non-null")
    reference = solution[_REF_COLUMNS].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(reference)):
        raise ValueError("solution contains a non-finite reference pose value")
    if solution["object_name"].isnull().any():
        raise ValueError("solution contains a null object_name")
    unknown = sorted(set(solution["object_name"].astype(str)) - set(_VERTICES)) if _VERTICES else []
    if unknown:
        raise ValueError(
            f"solution references object(s) with no embedded vertices: {unknown}. "
            "Regenerate the metric with metrics/build_kaggle_metrics.py."
        )


def _check_submission(submission: pd.DataFrame, row_id_column_name: str) -> None:
    if row_id_column_name not in submission.columns:
        raise ParticipantVisibleError(f"submission is missing the '{row_id_column_name}' column")
    for column in _REQUIRED_SUBMISSION_COLUMNS:
        if column not in submission.columns:
            raise ParticipantVisibleError(f"submission is missing required column '{column}'")
    for column in _POSE_COLUMNS:
        if not pandas.api.types.is_numeric_dtype(submission[column]):
            raise ParticipantVisibleError(f"submission column '{column}' must be numeric")
        values = submission[column].to_numpy(dtype=np.float64, na_value=np.nan)
        if not np.all(np.isfinite(values)):
            raise ParticipantVisibleError(
                f"submission column '{column}' contains a null, NaN or infinite value"
            )
    if submission[row_id_column_name].duplicated().any():
        first = submission.loc[submission[row_id_column_name].duplicated(), row_id_column_name].iloc[0]
        raise ParticipantVisibleError(
            f"submission has duplicate '{row_id_column_name}' values, first is '{first}'"
        )


def _check_provenance(merged: pd.DataFrame) -> None:

    for column in _PROVENANCE_COLUMNS:
                                                                                               
                                                                                           
                                                                                                
                                                                                  
        if merged[column].isna().any():
            raise ParticipantVisibleError(f"every row must carry a non-empty '{column}'")
        values = merged[column].astype(str).str.strip()
        if (values == "").any() or values.isin({"nan", "None", "NaN"}).any():
            raise ParticipantVisibleError(f"every row must carry a non-empty '{column}'")

    for column in _UNIFORM_PROVENANCE_COLUMNS:
        unique = set(merged[column].astype(str).str.strip())
        if len(unique) != 1:
            raise ParticipantVisibleError(
                f"'{column}' must be identical on every row; found {len(unique)} distinct values. "
                "One submission is one evaluator build, run from one commit."
            )

    for column in _PER_EPISODE_PROVENANCE_COLUMNS:
        per_episode = merged.groupby("episode_index")[column].apply(
            lambda block: block.astype(str).str.strip().nunique()
        )
        offenders = sorted(per_episode[per_episode > 1].index.tolist())
        if offenders:
            raise ParticipantVisibleError(
                f"'{column}' must be identical within each episode; episode(s) "
                f"{offenders[:5]} carry more than one value. One episode is one set of weights, "
                "but different episodes may use different ones."
            )

                                                                                               
                                                                                                 
                                                                                                 
                                
                                                                                            
    commit = merged["code_commit_url"].astype(str).str.strip().iloc[0]
    if not re.match(_CODE_COMMIT_PATTERN, commit):
        raise ParticipantVisibleError(
            f"'code_commit_url' must be a GitHub commit URL, for example "
            f"https://github.com/<owner>/<repo>/commit/<sha>. Got '{commit}'. A branch or tag URL "
            "is not accepted, because it does not pin the code that produced this submission."
        )

    for column, accepted in _ACCEPTED.items():
        if not accepted:
            continue
        reported = merged[column].astype(str).str.strip().iloc[0].lower()
        if reported not in accepted:
            raise ParticipantVisibleError(
                f"{column} '{reported}' is not a released version of the evaluator. "
                "Re-run with the official evaluate_policy.py and resubmit."
            )


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:

    if _USAGE_COLUMN in solution.columns:
        solution = solution.drop(columns=[_USAGE_COLUMN])
    if _REGISTER_OBJECT_MESH:
        return _score_registered_track3(solution, submission, row_id_column_name)

    _check_solution(solution, row_id_column_name)
    vertices_by_name = _VERTICES if _VERTICES else _load_vertices(solution)
    if _METRIC == "add_auc":
        missing_names = sorted(set(solution["object_name"].astype(str)) - set(vertices_by_name))
        if missing_names:
            raise ValueError(f"solution has no AUC vertices for object(s): {missing_names}")
    _check_submission(submission, row_id_column_name)

                                                                                              
                                                                                                 
                                                           
    solution = solution.copy()
    submission = submission.copy()
    solution[row_id_column_name] = solution[row_id_column_name].astype(str)
    submission[row_id_column_name] = submission[row_id_column_name].astype(str)

    keep = [row_id_column_name, *_REQUIRED_SUBMISSION_COLUMNS]
    merged = solution.merge(
        submission[keep], on=row_id_column_name, how="left", validate="one_to_one"
    )
    missing = merged["pos_x"].isnull()
    if missing.any():
        first = merged.loc[missing, row_id_column_name].iloc[0]
        raise ParticipantVisibleError(
            f"submission is missing {int(missing.sum())} required row(s); first is '{first}'. "
            "Every scored frame of every scored episode must be present."
        )

    _check_provenance(merged)

    values = []
    for episode, group in merged.groupby("episode_index", sort=True):
        try:
            achieved, reference, object_ids, names = _episode_tensors(group)
        except ValueError as error:
                                                                                             
                                                            
            raise ParticipantVisibleError(f"episode {int(episode)}: {error}") from error

        achieved = _normalise_quaternions(achieved, "submission", ParticipantVisibleError)
        reference = _normalise_quaternions(reference, "solution", ValueError)

                                                                                         
                                                                                            
                                                                                     
        if _ALIGN_INITIAL_POSE:
            achieved = _warp_to_reference_start(achieved, reference)

        if _METRIC == "add_auc":
            vertices = np.stack([vertices_by_name[str(name)] for name in names])
            values.append(_add_auc(_object_add(achieved, reference, vertices)))
        elif _METRIC == "spider_sr":
            values.append(float(np.mean(_spider_success(achieved, reference))))
        elif _METRIC == "maniptrans_sr":
            values.append(float(np.mean(_maniptrans_success(achieved, reference, object_ids))))
        elif _METRIC == "rpe_cm":
                                                                                                  
            values.append(_relative_position_error_cm(achieved, reference))
        elif _METRIC == "mppe_cm":
            values.append(_mean_per_point_error_cm(achieved, reference))
        else:
            raise ValueError(f"unknown metric {_METRIC!r}")

    if not values:
        raise ValueError(f"no episode produced a {_METRIC_DISPLAY} value")
    result = float(np.mean(values))
    if not np.isfinite(result):
        raise ValueError(f"computed {_METRIC_DISPLAY} is not finite: {result}")
    return result


def _score_registered_track3(solution, submission, row_id_column_name):

    _check_solution(solution, row_id_column_name)
    _check_submission(submission, row_id_column_name)
    if "object_mesh_b64" not in submission.columns:
        raise ParticipantVisibleError("Track 3 requires reconstructed meshes. Repack with --meshes.")
    if "reference_geometry_b64" not in solution.columns:
        raise ValueError("host solution requires reference_geometry_b64")
    payload = solution["reference_geometry_b64"].dropna().astype(str)
    payload = payload[payload.str.strip() != ""]
    if len(payload) != 1:
        raise ValueError("host solution must contain one reference geometry payload")
    geometry = _decode_archive(payload.iloc[0], 16_000_000)
    lookup = {str(name): i for i, name in enumerate(geometry['names'])}
    solution = solution.copy(); submission = submission.copy()
    solution[row_id_column_name] = solution[row_id_column_name].astype(str)
    submission[row_id_column_name] = submission[row_id_column_name].astype(str)
    if set(submission[row_id_column_name]) != set(solution[row_id_column_name]):
        raise ParticipantVisibleError("submission row IDs differ from the Track 3 sample")
    merged = solution.merge(submission[[row_id_column_name, *_REQUIRED_SUBMISSION_COLUMNS,
                                       'object_mesh_b64']], on=row_id_column_name,
                            how='left', validate='one_to_one')
    _check_provenance(merged)
    scores = []
    for episode, group in merged.groupby('episode_index', sort=True):
        achieved, reference, object_ids, names = _episode_tensors(group)
        achieved = _normalise_quaternions(achieved, 'submission', ParticipantVisibleError)
        reference = _normalise_quaternions(reference, 'solution', ValueError)
        achieved = achieved.copy(); reference = reference.copy()
        vertices, axes, orders = [], [], []
        for slot, name in enumerate(names):
            rows = group[group.object_slot == slot].sort_values(['world_index','frame_index'])
            encoded = rows.object_mesh_b64.fillna('').astype(str)
            present = encoded.str.strip() != ''
            if present.sum() != 1 or not present.iloc[0]:
                raise ParticipantVisibleError(f'episode {episode}, {name}: provide one mesh on its first pose row')
            try:
                candidate = _decode_object_mesh(encoded.iloc[0])
            except Exception as error:
                raise ParticipantVisibleError(f'episode {episode}, {name}: {error}') from error
            i = lookup[str(name)]
            mesh = (geometry[f'v{i}'], geometry[f'f{i}'].astype(np.int64))
            anchor = geometry[f'anchor{i}']
            candidate_anchor, rotation = _register_object_body(candidate, mesh, anchor,
                                                              1_000_003 * int(episode) + 101 * slot)
            achieved[:, :, slot] = _convert_object_body(achieved[:, :, slot], candidate_anchor, rotation)
            reference[:, :, slot, :3] += np.einsum('...ij,j->...i',
                _rotation_matrix(reference[:, :, slot, 3:]), anchor)
            vertices.append(geometry[f'points{i}'] - anchor)
            axes.append(geometry[f'axis{i}'])
            orders.append(int(geometry[f'order{i}']))
        achieved = _align_registered_scene(achieved, reference, axes)
        scores.append(_registered_pose_metric(_METRIC, achieved, reference, vertices, axes, orders))
    return float(np.mean(scores))
__KIT_EOF__
cat > $W/d2.py << '__D2_EOF__'
#!/usr/bin/env python3
"""refine1104 ep6 round 2 (D2): data-driven de-smoothing of bob1's in-pot spoon (rule + gates pre-registered in
../gates.json, written 12:22 EST before this script existed).  bob1's spoon mesh and its levelling are kept: only a
world-frame rigid correction is left-multiplied onto bob1's reference poses on f36-171.
Steps: Ken's raw-vs-smoothed FoundationPose residual -> flip rejection (> 30 deg) -> Gaussian sigma s -> per-frame choice
between the corrected pose and bob1's pose by the round-1 mask/depth cost -> choice smoothed (sigma s) -> 6-frame ramps.
Gates computed here: G1 (primary parity CV, mesh-relative), G1s (strict parity CV), G2 (depth), G3 (steps),
G4 (kit-registration roll consistency + SR margins), G5 (raw right hand).  Read-only on every input; writes only
into WD/out."""
import importlib.util
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

WD = "/data/user_data/haisenk/v2d/refine1104/ep6_d2/work"
OUT = os.path.join(WD, "out")
os.makedirs(OUT, exist_ok=True)
sys.argv = ["fit6.py", OUT, "none"]
sys.path.insert(0, WD)
import fit6 as F6  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter1d  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

T0 = time.time()
sp, pot = F6.objs[1], F6.objs[0]
K, w2c, S, N = F6.K, F6.w2c, F6.S, F6.N
c2w = np.linalg.inv(w2c)
W = np.arange(42, 166)
REST_FR = list(range(0, 25)) + list(range(220, 243))
RAMP_FR = list(range(36, 42)) + list(range(166, 172))
SIGMAS = (1.0, 1.5, 2.0)
FLIP_DEG = 30.0
IDX = np.arange(N)


def log(m):
    print(f"[d2 +{time.time() - T0:6.1f}s] {m}", flush=True)


# spoon long axis (body frame), bowl end
Vs = sp["V"]
MU = Vs.mean(0)
_, EVEC = np.linalg.eigh(np.cov((Vs - MU).T))
AX1, AX2 = EVEC[:, 2], EVEC[:, 1]
_s = (Vs - MU) @ AX1
BOWL = 1.0 if np.ptp(((Vs - MU) @ AX2)[_s > 0.5 * _s.max()]) > np.ptp(((Vs - MU) @ AX2)[_s < 0.5 * _s.min()]) else -1.0


def metrics(f, Tc):
    """round-1 cost_at (cost, IoU_occ, median FS-depth residual cm) + D3 depth-gradient slope + visible centroid."""
    Ms, Hm, D = F6.frame_data(f)
    M = Ms[1]
    O = Hm | Ms[0]
    sil, zb = F6.render(sp, Tc, K[f], need_depth=True)
    inter = (sil & M).sum()
    uo = (M | (sil & ~O)).sum()
    iou = inter / uo if uo else 0.0
    valid = (M & sil) & np.isfinite(D) & np.isfinite(zb)
    c = 1.0 - iou
    dm = float("nan")
    if valid.sum() > 30:
        dm = float(np.median(D[valid] - zb[valid]) * 100)
        c += 0.04 * min(abs(dm), 6.0)
    slope = float("nan")
    z_obj = F6.obj_depth(sp, Tc)
    pxcm = S * z_obj / K[f][0] * 100.0
    dh = distance_transform_edt(~Hm) * pxcm if Hm.any() else np.full(M.shape, 1e9)
    vv = binary_erosion(M & sil, iterations=1) & np.isfinite(D) & np.isfinite(zb) & (dh >= 1.5)
    ys, xs = np.nonzero(vv)
    if len(ys) >= 60:
        u = (xs + 0.5) * S - 0.5
        v = (ys + 0.5) * S - 0.5
        zz = zb[ys, xs]
        Kf = K[f]
        Pc = np.stack([(u - Kf[2]) / Kf[0] * zz, (v - Kf[3]) / Kf[1] * zz, zz], 1)
        body = (Pc - Tc[:3, 3]) @ Tc[:3, :3]
        s = ((body - MU) @ AX1) * BOWL * 100.0
        r = (D[ys, xs] - zz) * 100.0
        bins = np.arange(np.floor(s.min() / 2) * 2, s.max() + 2, 2.0)
        bc, bmed = [], []
        for a, b in zip(bins[:-1], bins[1:]):
            sel = (s >= a) & (s < b)
            if sel.sum() >= 15:
                bc.append(0.5 * (a + b))
                bmed.append(float(np.median(r[sel])))
        if len(bc) >= 3:
            slope = float(np.polyfit(bc, bmed, 1)[0])
    vis = sil & ~O
    yv, xv = np.nonzero(vis)
    vc = [float(xv.mean() * S), float(yv.mean() * S)] if len(xv) else [float("nan")] * 2
    return [float(c), float(iou), dm, slope, vc[0], vc[1]]


def frame_task(args):
    f, poses = args
    Ms, Hm, D = F6.frame_data(f)
    M = Ms[1] & ~Hm
    ys, xs = np.nonzero(M)
    out = {"_mask": [float(xs.mean() * S), float(ys.mean() * S), int(len(xs))] if len(xs) else None}
    for key, Tw in poses.items():
        out[key] = metrics(f, w2c[f] @ np.asarray(Tw))
    return f, out


def run_requests(req):
    jobs = [(f, {k: np.asarray(v).tolist() for k, v in d.items()}) for f, d in sorted(req.items()) if d]
    with Pool(8) as pool:
        return dict(pool.map(frame_task, jobs, chunksize=2))


# ---------------------------------------------------------------- smoothing residual from Ken's FP run
rows = json.load(open(os.path.join(WD, "d4_rawfp.json")))["rows"]
Tr = np.array([r["Tr"] for r in rows])
Ts = np.array([r["Ts"] for r in rows])
kV, kF = F6.rr.load_obj(os.path.join(F6.E, "wooden_spoon", "result", "mesh.obj"))
MUK = kV.mean(0)
RV = np.zeros((N, 3))
DL = np.zeros((N, 3))
PV = np.zeros((N, 3))
ANG = np.zeros(N)
for f in range(N):
    Rd = Tr[f][:3, :3] @ Ts[f][:3, :3].T
    cs = Ts[f][:3, :3] @ MUK + Ts[f][:3, 3]
    cr = Tr[f][:3, :3] @ MUK + Tr[f][:3, 3]
    Rc = c2w[f][:3, :3]
    RV[f] = Rot.from_matrix(Rc @ Rd @ Rc.T).as_rotvec()
    DL[f] = Rc @ (cr - cs)
    PV[f] = Rc @ cs + c2w[f][:3, 3]
    ANG[f] = np.degrees(np.linalg.norm(RV[f]))
GOOD = ANG <= FLIP_DEG
log(f"loaded; flip frames (> {FLIP_DEG} deg) in W: {IDX[W][~GOOD[W]].tolist()}; all: {int((~GOOD).sum())}")


def build_corr(sig, use=None):
    good = GOOD.copy() if use is None else (GOOD & use)
    r, d = RV.copy(), DL.copy()
    for arr in (r, d):
        for k in range(3):
            arr[~good, k] = np.interp(IDX[~good], IDX[good], arr[good, k])
    return gaussian_filter1d(r, sig, axis=0, mode="nearest"), gaussian_filter1d(d, sig, axis=0, mode="nearest")


def apply_corr(Tw, w, r, d, p):
    Rd = Rot.from_rotvec(w * r).as_matrix()
    T = Tw.copy()
    T[:3, :3] = Rd @ Tw[:3, :3]
    T[:3, 3] = Rd @ (Tw[:3, 3] - p) + p + w * d
    return T


def full_weight(wW):
    w = np.zeros(N)
    w[42:166] = wW
    for j in range(1, 7):
        w[42 - j] = w[42] * (1.0 - j / 7.0)
        w[165 + j] = w[165] * (1.0 - j / 7.0)
    return w


def track(w, r, d):
    Tw = sp["T"].copy()
    for f in range(N):
        if w[f] > 0:
            Tw[f] = apply_corr(sp["T"][f], w[f], r[f], d[f], PV[f])
    return Tw


def steps(Tw):
    rs = np.array([np.degrees(Rot.from_matrix(Tw[i][:3, :3].T @ Tw[i + 1][:3, :3]).magnitude()) for i in range(N - 1)])
    cen = np.einsum("nij,j->ni", Tw[:, :3, :3], MU) + Tw[:, :3, 3]
    ts = np.linalg.norm(np.diff(cen, axis=0), axis=1) * 100
    return rs, ts


def step_gate(Tw):
    rs, ts = steps(Tw)
    a, b = rs[35:172], ts[35:172]
    o = dict(rot_max=round(float(a.max()), 2), rot_argmax=int(35 + a.argmax()), rot_p90=round(float(np.percentile(a, 90)), 2),
             trans_max_cm=round(float(b.max()), 3), trans_argmax=int(35 + b.argmax()), trans_p90_cm=round(float(np.percentile(b, 90)), 3))
    o["pass"] = bool(a.max() <= 15.0 and np.percentile(a, 90) <= 10.6 and b.max() <= 3.0 and np.percentile(b, 90) <= 1.63)
    return o


def twist_deg(Rrel, axis):
    q = Rot.from_matrix(Rrel).as_quat()  # x y z w
    q = q * np.sign(q[..., 3:4] + 1e-15)
    proj = q[..., :3] @ axis
    return np.degrees(2.0 * np.arctan2(proj, q[..., 3]))


if __name__ == "__main__":
    # ------------------------------------------------ stage 1: reference + candidates
    CORR = {}
    req = {f: {} for f in range(N)}
    for f in list(W) + REST_FR + RAMP_FR:
        req[f]["ref"] = sp["T"][f]
    for s in SIGMAS:
        CORR[("full", s)] = build_corr(s)
        for P in (0, 1):
            CORR[("strict", s, P)] = build_corr(s, use=(IDX % 2 == P))
        r, d = CORR[("full", s)]
        for f in W:
            req[f][f"cand_full_{s}"] = apply_corr(sp["T"][f], 1.0, r[f], d[f], PV[f])
            rs_, ds_ = CORR[("strict", s, f % 2)]
            req[f][f"cand_strict_{s}"] = apply_corr(sp["T"][f], 1.0, rs_[f], ds_[f], PV[f])
    res = run_requests(req)
    log(f"stage 1 done ({sum(len(v) for v in req.values())} renders)")
    REST = float(np.mean([res[f]["ref"][1] for f in REST_FR]))
    rest_parts = [round(float(np.mean([res[f]["ref"][1] for f in range(0, 25)])), 4),
                  round(float(np.mean([res[f]["ref"][1] for f in range(220, 243)])), 4)]
    log(f"REST (bob1 mesh own rest-phase IoU_occ) = {REST:.4f} {rest_parts}; bar 0.85*REST = {0.85 * REST:.4f}")

    # ------------------------------------------------ stage 2: tracks (full, primary CV, strict CV)
    req2 = {f: {} for f in range(N)}
    TR = {}
    for s in SIGMAS:
        r, d = CORR[("full", s)]
        cf = np.array([1.0 if res[f][f"cand_full_{s}"][0] < res[f]["ref"][0] else 0.0 for f in W])
        wf = full_weight(gaussian_filter1d(cf, s, mode="nearest"))
        Tf = track(wf, r, d)
        TR[("full", s)] = (Tf, wf, cf)
        for f in range(36, 172):
            req2[f][f"full_{s}"] = Tf[f]
        for P in (0, 1):
            trm = (W % 2) == P
            cI = np.interp(W, W[trm], cf[trm])
            wP = full_weight(gaussian_filter1d(cI, s, mode="nearest"))
            TP = track(wP, r, d)
            for f in W[~trm]:
                req2[f][f"cv_{s}"] = TP[f]
            rs_, ds_ = CORR[("strict", s, P)]
            ct = np.array([1.0 if res[f][f"cand_strict_{s}"][0] < res[f]["ref"][0] else 0.0 for f in W[trm]])
            cS = np.interp(W, W[trm], ct)
            wS = full_weight(gaussian_filter1d(cS, s, mode="nearest"))
            TS = track(wS, rs_, ds_)
            for f in W[~trm]:
                req2[f][f"strict_{s}"] = TS[f]
    res2 = run_requests(req2)
    log(f"stage 2 done ({sum(len(v) for v in req2.values())} renders)")

    def g1(key, src=res2):
        io_n = np.array([src[f][key][1] for f in W])
        io_o = np.array([res[f]["ref"][1] for f in W])
        de_n = np.array([src[f][key][2] for f in W])
        de_o = np.array([res[f]["ref"][2] for f in W])
        sl_n = np.array([src[f][key][3] for f in W])
        o = dict(n=len(W), iou_occ_mean_bob1=round(float(io_o.mean()), 4), iou_occ_mean_new=round(float(io_n.mean()), 4),
                 gain=round(float(io_n.mean() - io_o.mean()), 4), frac_worse_gt_0p05=round(float(np.mean(io_n < io_o - 0.05)), 4),
                 absdep_median_bob1_cm=round(float(np.nanmedian(np.abs(de_o))), 3), absdep_median_new_cm=round(float(np.nanmedian(np.abs(de_n))), 3),
                 slope_median_bob1=round(float(np.nanmedian([res[f]["ref"][3] for f in W])), 4), slope_median_new=round(float(np.nanmedian(sl_n)), 4),
                 mean_cost=round(float(np.mean([src[f][key][0] for f in W])), 5),
                 mean_cost_bob1=round(float(np.mean([res[f]["ref"][0] for f in W])), 5),
                 ratio_to_rest=round(float(io_n.mean() / REST), 4),
                 per_block_iou={f"f{a}-{b - 1}": [round(float(io_o[a - 42:b - 42].mean()), 3), round(float(io_n[a - 42:b - 42].mean()), 3)]
                                for a, b in ((42, 70), (70, 100), (100, 130), (130, 166))})
        o["absdep_drop_cm"] = round(o["absdep_median_bob1_cm"] - o["absdep_median_new_cm"], 3)
        return o

    per_sigma = {}
    for s in SIGMAS:
        Tf, wf, cf = TR[("full", s)]
        cv = g1(f"cv_{s}")
        st = g1(f"strict_{s}")
        ins = g1(f"full_{s}")
        per_sigma[s] = dict(steps=step_gate(Tf), cv_primary=cv, cv_strict=st, insample_full=ins,
                            choice_frac=round(float(cf.mean()), 3), weight_mean_W=round(float(wf[42:166].mean()), 3),
                            n_changed=int((wf > 0).sum()))
        log(f"sigma {s}: steps {per_sigma[s]['steps']} | CV {cv['iou_occ_mean_new']} (bob1 {cv['iou_occ_mean_bob1']}, gain {cv['gain']}, worse {cv['frac_worse_gt_0p05']}, "
            f"dep {cv['absdep_median_bob1_cm']}->{cv['absdep_median_new_cm']}, cost {cv['mean_cost']}) | strict {st['iou_occ_mean_new']} gain {st['gain']} worse {st['frac_worse_gt_0p05']} | "
            f"in-sample {ins['iou_occ_mean_new']} | choice {cf.mean():.2f}")
    passing = [s for s in SIGMAS if per_sigma[s]["steps"]["pass"]]
    if passing:
        s_star = sorted(passing, key=lambda s: (per_sigma[s]["cv_primary"]["mean_cost"], -s))[0]
    else:
        s_star = sorted(SIGMAS, key=lambda s: (per_sigma[s]["cv_primary"]["mean_cost"], -s))[0]
    log(f"step-passing sigmas {passing}; chosen sigma {s_star}")
    P_ = per_sigma[s_star]
    gates = {}
    cv = P_["cv_primary"]
    gates["G1"] = dict(cv, REST=round(REST, 4), REST_parts=rest_parts, bar=round(0.85 * REST, 4),
                       **{"pass": bool(cv["iou_occ_mean_new"] >= 0.85 * REST and cv["gain"] >= 0.05 and cv["frac_worse_gt_0p05"] < 0.20)})
    st = P_["cv_strict"]
    gates["G1s"] = dict(st, **{"pass": bool(st["gain"] >= 0.03 and st["frac_worse_gt_0p05"] < 0.25)})
    gates["G2"] = dict(absdep_median_bob1_cm=cv["absdep_median_bob1_cm"], absdep_median_new_cm=cv["absdep_median_new_cm"],
                       drop_cm=cv["absdep_drop_cm"], **{"pass": bool(cv["absdep_drop_cm"] >= 0.5)})
    gates["G3"] = dict(P_["steps"], sigma_passing=passing, bob1=step_gate(sp["T"]))
    gates["G3"]["pass"] = bool(s_star in passing)

    # ------------------------------------------------ G4 kit registration + roll / SR margins
    Tf, wf, cf = TR[("full", s_star)]
    g4 = {}
    try:
        spec = importlib.util.spec_from_file_location("kitauc", os.path.join(WD, "kit_AUC.py"))
        kit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kit)
        mp = os.path.join(F6.SCENE, "score_input", "meshes", "episode_000006", "wooden_spoon.obj")
        sv, sf = F6.rr.load_obj(mp)
        seed = 1_000_003 * 6 + 101
        cc, cr = kit._surface_moments(sv, sf)
        anchor, Rreg = kit._register_object_body((sv, sf), (sv, sf), cc, seed)
        g4["a_kit_self_registration_deg"] = round(float(np.degrees(Rot.from_matrix(Rreg).magnitude())), 4)
        g4["a_anchor_shift_mm"] = round(float(np.linalg.norm(anchor - cc) * 1000), 4)
        # info only: two different surface samplings of the same mesh (is the registration roll-stable?)
        cp = (kit._sample_mesh_surface(sv, sf, 3000, seed) - cc)
        rp = kit._sample_mesh_surface(sv, sf, 3000, seed + 1) - cc
        R2, _ = kit._align_rigid(rp, cp)
        g4["info_resampled_registration_deg"] = round(float(np.degrees(Rot.from_matrix(R2).magnitude())), 3)
        g4["score_mesh"] = mp
        g4["score_mesh_nV"] = int(len(sv))
    except Exception as e:  # noqa: BLE001
        g4["a_error"] = repr(e)
    Rrel = np.einsum("nji,njk->nik", sp["T"][:, :3, :3], Tf[:, :3, :3])
    angs = np.degrees(Rot.from_matrix(Rrel).magnitude())
    roll = np.abs(twist_deg(Rrel, AX1))
    cen_o = np.einsum("nij,j->ni", sp["T"][:, :3, :3], MU) + sp["T"][:, :3, 3]
    cen_n = np.einsum("nij,j->ni", Tf[:, :3, :3], MU) + Tf[:, :3, 3]
    dc = np.linalg.norm(cen_n - cen_o, axis=1) * 100
    g4.update(b_rot_max_deg=round(float(angs.max()), 2), b_rot_argmax=int(angs.argmax()), b_roll_max_deg=round(float(roll.max()), 2),
              b_roll_argmax=int(roll.argmax()), c_rot_mean_deg_all243=round(float(angs.mean()), 3),
              c_centroid_mean_cm_all243=round(float(dc.mean()), 3), info_rot_median_in_W=round(float(np.median(angs[42:166])), 2),
              info_centroid_median_in_W_cm=round(float(np.median(dc[42:166])), 3), info_centroid_max_cm=round(float(dc.max()), 3))
    g4["pass"] = bool(g4.get("a_kit_self_registration_deg", 99) <= 1.0 and angs.max() <= 30.0 and roll.max() <= 30.0
                      and angs.mean() <= 10.0 and dc.mean() <= 2.0)
    gates["G4"] = g4

    # ------------------------------------------------ G5 raw right hand
    z = np.load(os.path.join(WD, "f1d_joints_cam.npz"))
    bc = np.load(os.path.join(F6.SCENE, "bundle_cleaned", "result.npz"))
    Cw = F6.A[None] @ bc["camera_to_world_transform"].astype(np.float64)
    Jm = np.einsum("nij,nkj->nki", Cw[:, :3, :3], z["bundle_merged_joints_cam"]) + Cw[:, None, :3, 3]
    vm = z["bundle_merged_valid"]
    smp = sp["samples"][::4]
    dist = {}
    for f in range(N):
        if not (vm[f] and wf[f] > 0):
            continue
        r_old = cKDTree(smp @ sp["T"][f][:3, :3].T + sp["T"][f][:3, 3]).query(Jm[f])[0] * 100
        r_new = cKDTree(smp @ Tf[f][:3, :3].T + Tf[f][:3, 3]).query(Jm[f])[0] * 100
        dist[f] = (float(r_old.min()), float(r_new.min()))

    def bst(fs):
        fs = [f for f in fs if f in dist]
        if not fs:
            return None
        a = np.array([dist[f] for f in fs])
        return dict(n=len(fs), min_old=round(float(np.median(a[:, 0])), 3), min_new=round(float(np.median(a[:, 1])), 3),
                    frac_closer=round(float(np.mean(a[:, 1] < a[:, 0])), 3))
    g5 = dict(changed=bst(sorted(dist)), windows={f"f{a}-{b - 1}": bst(range(a, b)) for a, b in ((36, 70), (70, 100), (100, 130), (130, 172))})
    g5["pass"] = bool(g5["changed"] is not None and g5["changed"]["min_new"] <= g5["changed"]["min_old"] + 0.05)
    gates["G5"] = g5
    all_pass = all(gates[k]["pass"] for k in ("G1", "G1s", "G2", "G3", "G4", "G5"))

    # ------------------------------------------------ info: video motion agreement (mask centroid vs visible silhouette centroid)
    def motion_corr(key_of):
        a, b = [], []
        for f in range(42, 165):
            m0, m1 = res[f]["_mask"], res[f + 1]["_mask"]
            p0, p1 = key_of(f), key_of(f + 1)
            if m0 is None or m1 is None or p0 is None or p1 is None:
                continue
            if not all(np.isfinite(p0[4:6])) or not all(np.isfinite(p1[4:6])):
                continue
            a.append([m1[0] - m0[0], m1[1] - m0[1]])
            b.append([p1[4] - p0[4], p1[5] - p0[5]])
        a, b = np.array(a), np.array(b)
        cx = float(np.corrcoef(a[:, 0], b[:, 0])[0, 1])
        cy = float(np.corrcoef(a[:, 1], b[:, 1])[0, 1])
        return dict(n=len(a), corr_dx=round(cx, 3), corr_dy=round(cy, 3), mask_step_px_median=round(float(np.median(np.linalg.norm(a, axis=1))), 2),
                    pose_step_px_median=round(float(np.median(np.linalg.norm(b, axis=1))), 2))
    info_motion = dict(bob1=motion_corr(lambda f: res[f]["ref"]),
                       d2_insample=motion_corr(lambda f: res2[f].get(f"full_{s_star}")))

    # ------------------------------------------------ targets in the bundle_cleaned frame (for the H5_PATCH build)
    A = F6.A
    Ai = np.linalg.inv(A)
    Tb_ref = bc["object_1_to_world_transform"].astype(np.float64)
    map_res = float(np.abs(A[None] @ Tb_ref - sp["T"]).max())
    Tb_new = Tb_ref.copy()
    mask = wf > 0
    for f in np.flatnonzero(mask):
        Csc = Tf[f] @ np.linalg.inv(sp["T"][f])
        Tb_new[f] = (Ai @ Csc @ A) @ Tb_ref[f]
    chk = float(np.abs(A[None] @ Tb_new - Tf)[mask].max()) if mask.any() else 0.0
    np.savez(os.path.join(OUT, "d2_targets.npz"), object_1_to_world_transform=Tb_new, object_1_mask=mask,
             sigma=np.float64(s_star), weight=wf, choice_W=cf, scene_world_track=Tf, bob1_scene_world_track=sp["T"], A=A)
    # judge pseudo-truth (round-1 V1 construction: flip rejection + sigma 1, no choice, full weight on W + ramps)
    r1, d1 = CORR[("full", 1.0)]
    np.save(os.path.join(OUT, "pseudo_truth_V1_scene_world.npy"), track(full_weight(np.ones(len(W))), r1, d1))
    result = dict(rule="see ../gates.json fix_under_test", sigma_chosen=s_star, gates=gates, all_pass_G1_G5=all_pass,
                  per_sigma={str(k): v for k, v in per_sigma.items()}, info_video_motion=info_motion,
                  flip_frames_W=IDX[W][~GOOD[W]].tolist(), flip_frames_all=IDX[~GOOD].tolist(),
                  per_frame=dict(weight=np.round(wf, 4).tolist(), choice_W=cf.astype(int).tolist(),
                                 ref=[res[f]["ref"][:4] for f in W], cand=[res[f][f"cand_full_{s_star}"][:4] for f in W],
                                 new_full=[res2[f][f"full_{s_star}"][:4] for f in range(36, 172)],
                                 cv=[res2[f][f"cv_{s_star}"][:4] for f in W],
                                 rot_vs_bob1_deg=np.round(angs, 2).tolist(), roll_vs_bob1_deg=np.round(roll, 2).tolist(),
                                 centroid_vs_bob1_cm=np.round(dc, 3).tolist()),
                  bundle_map=dict(scene_eq_A_bundle_residual_m=map_res, targets_roundtrip_max=chk, n_target_frames=int(mask.sum()),
                                  target_frames=[int(np.flatnonzero(mask).min()), int(np.flatnonzero(mask).max())] if mask.any() else None),
                  sec=round(time.time() - T0, 1))
    json.dump(result, open(os.path.join(OUT, "d2_result.json"), "w"), indent=1)
    short = {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict)) or kk in ("REST_parts", "sigma_passing")} for k, v in gates.items()}
    print(json.dumps(dict(sigma=s_star, all_pass_G1_G5=all_pass, gates=short, info_video_motion=info_motion,
                          bundle_map=result["bundle_map"]), indent=1), flush=True)
    log("D2 DONE")
__D2_EOF__
md5sum $W/kit_AUC.py $W/d2.py | cut -c1-8,33-
cd $W && (timeout 520 /home/haisenk/anaconda3/bin/python $W/d2.py > $W/out/d2.log 2>&1; echo EXIT $? >> $W/out/d2.log) &
P=$!; while kill -0 $P 2>/dev/null; do sleep 15; echo -n .; done; echo
tail -c 9000 $W/out/d2.log; date
