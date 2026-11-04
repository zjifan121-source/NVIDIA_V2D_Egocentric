"""Optional observed-point/table constraint in metric, original-mesh coordinates.

The point is an input-reviewed body/table contact, not the lowest mesh vertex.
No depth is modified. A corrected public pose must reset the estimator state.
"""
import numpy as np


def project_contact(pose, camera_to_world, point_local, plane_world,
                    frame, rest_frames, maximum_shift_m):
    pose = np.array(pose, dtype=float, copy=True)
    camera = np.asarray(camera_to_world, dtype=float)
    point = np.asarray(point_local, dtype=float)
    plane = np.asarray(plane_world, dtype=float)
    if point.shape != (3,) or plane.shape != (4,) or not np.isfinite(point).all() or not np.isfinite(plane).all():
        raise ValueError('Expected a finite local contact point and world plane')
    if not np.isclose(np.linalg.norm(plane[:3]), 1, atol=1e-6):
        raise ValueError('Contact plane normal must be unit length')
    world = camera @ pose
    gap = float((world[:3, :3] @ point + world[:3, 3]) @ plane[:3] + plane[3])
    active = frame in rest_frames
    correction = 0.0
    if active:
        if abs(gap) > maximum_shift_m:
            raise ValueError(f'CONTACT_BOUND frame {frame}: {gap} m')
        correction = -gap
        pose[:3, 3] += camera[:3, :3].T @ (correction * plane[:3])
    return pose, dict(frame=int(frame), contact_active=active, pre_gap_m=gap,
                      normal_correction_m=correction, post_gap_m=gap + correction)


def apply_to_tracker(tracker, pose, constraint, frame, from_matrix):
    raw = np.array(pose.to_matrix(), dtype=float, copy=True)
    corrected, info = project_contact(raw, constraint['camera_to_world_file'][frame],
        constraint['point_file'], constraint['plane_file'], frame,
        constraint['rest_frames'], constraint['maximum_shift_m'])
    result = from_matrix(corrected)
    if info['contact_active']:
        tracker.reset_to_pose(result)
    return result, raw, info


def load_contact(manifest, config, payload, checked_path, validate_transform):
    """All additional inputs must be included in the normal frozen file inventory."""
    cfg = config.get('contact_constraint')
    if cfg is None:
        return None
    names = ['initial_pose_file', 'camera_to_world_file', 'plane_file', 'point_file']
    if set(cfg) != set(names + ['rest_frames', 'maximum_shift_m', 'input_review_evidence']):
        raise ValueError('Unknown/missing contact-constraint setting')
    if not cfg['input_review_evidence']:
        raise ValueError('Observed contact and rest intervals require input-side review')
    if not 0 < cfg['maximum_shift_m'] <= 0.06:
        raise ValueError('Contact correction limit must be within (0, 0.06] metres')
    rest = cfg['rest_frames']
    if not rest or len(rest) != len(set(rest)) or any(type(t) is not int or not 0 <= t < manifest['frame_count'] for t in rest):
        raise ValueError('Invalid observed rest frames')
    values = {}
    for name in names:
        if cfg[name] not in manifest['files']:
            raise ValueError('Unpinned contact input: ' + name)
        values[name] = np.load(checked_path(payload, cfg[name]), allow_pickle=False)
    values['initial_pose_file'] = validate_transform(values['initial_pose_file'])
    cameras = values['camera_to_world_file']
    if cameras.shape != (manifest['frame_count'], 4, 4):
        raise ValueError('Incomplete contact camera timeline')
    for camera in cameras:
        validate_transform(camera)
    # Validate point/plane without changing a pose; depth, scale and rotations stay fixed.
    project_contact(values['initial_pose_file'], cameras[0], values['point_file'],
                    values['plane_file'], -1, rest, cfg['maximum_shift_m'])
    return dict(**values, rest_frames=rest, maximum_shift_m=cfg['maximum_shift_m'])
