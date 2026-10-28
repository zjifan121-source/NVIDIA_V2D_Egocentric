"""Align a scene to an observed stereo tabletop, undoing per-object stage-2 tilts.

Consumes plane normals already measured from RGB-selected table pixels and metric
depth. Keeps the cleaned original reconstruction tracks before mesh-based leveling;
applies one common rigid world rotation. No scans, GT, trajectory freezing or scale
changes. Retarget the output with --no-level and inspect its support/contact checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

import run_load_retarget as L


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('scene', type=Path)
    p.add_argument('plane_diagnostic', type=Path)
    p.add_argument('output_bundle', type=Path)
    p.add_argument('--episode', type=int, required=True)
    args = p.parse_args()
    if args.output_bundle.exists():
        raise FileExistsError(args.output_bundle)
    rows = [r for r in json.loads(args.plane_diagnostic.read_text()) if r['episode'] == args.episode and r['inliers'] >= 1000]
    if len(rows) < 5:
        raise ValueError('At least five well-supported table plane fits are required')
    normals = np.asarray([r['normal'] for r in rows])
    normal = np.median(normals, axis=0)
    normal /= np.linalg.norm(normal)
    spread = np.degrees(np.arccos(np.clip(normals @ normal, -1, 1))).max()
    angle = np.degrees(np.arccos(normal[2]))
    if spread > 5 or angle > 20:
        raise ValueError(f'Table fits require inspection: spread={spread:.2f}, tilt={angle:.2f}')
    base = args.scene / 'bundle_cleaned'
    data = dict(np.load(base / 'result.npz'))
    # Reproduce cleaning without the mesh-based leveling stage. This recovers the
    # source camera-relative poses, including the same rest and jump repairs.
    with tempfile.TemporaryDirectory(prefix='v2d_prelevel_') as temp:
        path, cleaning = L.cleaned_bundle(args.scene / 'bundle_merged', Path(temp) / 'bundle', L.TRACK3_FPS,
                                         rest_hold=True, hand_clean=True, object_clean=True,
                                         level=False, attach=False, hand_smooth=2.0)
        pre = dict(np.load(path / 'result.npz'))
    previous_world = data['camera_to_world_transform'][0] @ np.linalg.inv(pre['camera_to_world_transform'][0])
    if not np.allclose(previous_world @ pre['camera_to_world_transform'], data['camera_to_world_transform'], atol=2e-5):
        raise ValueError('Stage-2 cameras are not related by one rigid world transform')
    if not np.allclose(previous_world @ pre['object_to_world_transform'], data['object_to_world_transform'], atol=2e-5):
        raise ValueError('Object-0 preprocessing differs; cannot safely undo leveling')
    H = np.eye(4)
    H[:3, :3] = Rotation.align_vectors([[0., 0., 1.]], [normal])[0].as_matrix()
    restored = []
    for slot, (key, _, _) in enumerate(L.object_slots(data)):
        before = previous_world @ pre[key]
        correction = np.linalg.inv(before[0]) @ data[key][0]
        if not np.allclose(before @ correction, data[key], atol=2e-5):
            raise ValueError(f'Object {slot} does not have a constant stage-2 correction')
        restored.append(dict(slot=slot, removed_local_tilt_deg=float(np.degrees(Rotation.from_matrix(correction[:3, :3]).magnitude()))))
        data[key] = (H @ before).astype(np.float32)
    data['camera_to_world_transform'] = (H @ data['camera_to_world_transform']).astype(np.float32)
    camera = data['camera_to_world_transform']
    data['object_to_camera_transform'] = (np.linalg.inv(camera) @ data['object_to_world_transform']).astype(np.float32)
    for side in ('left', 'right'):
        data[f'hand_{side}_wrist_to_world_transform'] = (camera @ data[f'hand_{side}_wrist_to_camera_transform']).astype(np.float32)
    if 'gravity_alignment_transform' in data:
        data['gravity_alignment_transform'] = (H @ previous_world @ data['gravity_alignment_transform']).astype(np.float32)
    shutil.copytree(base, args.output_bundle)
    np.savez(args.output_bundle / 'result.npz', **data)
    report = dict(source_scene=str(args.scene), episode=args.episode, table_normal_before=normal.tolist(),
                  rotation_deg=float(angle), normal_spread_deg=float(spread), measurement_frames=[r['frame'] for r in rows],
                  removed_object_tilts=restored, world_transform=H.tolist(), scale_changed=False,
                  preprocessing=cleaning, method='Observed stereo table normal; restore source poses before mesh-based local tilts')
    (args.output_bundle / 'table_alignment.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
