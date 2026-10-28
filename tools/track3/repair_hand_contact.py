"""Repair a displaced grasp in a copied bundle, using its retargeted pinch geometry.

Object/camera trajectories and meshes are preserved. During a specified, visually
confirmed hold, translate the selected hand by one object-local offset so the
thumb/index pinch midpoint lies inside the nearest edge of the object's midplane.
The caller must inspect contact and retarget again before training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh


def pinch_target(vertices, fingertips, margin=0.015):
    centre = vertices.mean(axis=0)
    _, _, axes = np.linalg.svd(vertices - centre, full_matrices=False)
    coords = (vertices - centre) @ axes.T
    lo, hi = coords.min(0), coords.max(0)
    if hi[2] - lo[2] >= 0.4 * (hi[1] - lo[1]):
        raise ValueError('This repair requires a thin object with an identifiable midplane')
    axis = fingertips[0] - fingertips[1]
    alignment = abs(axis @ axes[2]) / np.linalg.norm(axis)
    if alignment < 0.5:
        raise ValueError('Pinch axis does not straddle the object thickness; translation is insufficient')
    midpoint = fingertips.mean(axis=0)
    target = (midpoint - centre) @ axes.T
    target[:2] = np.clip(target[:2], lo[:2] + margin, hi[:2] - margin)
    target[2] = (lo[2] + hi[2]) / 2
    return centre + target @ axes - midpoint, float(alignment)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('scene', type=Path)
    p.add_argument('output_bundle', type=Path)
    p.add_argument('--object', required=True)
    p.add_argument('--side', choices=['left', 'right'], required=True)
    p.add_argument('--hold', type=int, nargs=2, required=True, metavar=('START', 'END'))
    p.add_argument('--anchor', type=int, required=True)
    p.add_argument('--blend', type=int, default=8)
    p.add_argument('--rigid-grip', action='store_true', help='Use the anchor wrist pose relative to the object throughout the hold')
    p.add_argument('--target-local', type=float, nargs=3, help='Explicit mesh-local pinch midpoint for a non-planar object; inspect the chosen feature first')
    p.add_argument('--base-bundle', type=Path, help='Apply another hold repair to this already-corrected bundle')
    args = p.parse_args()
    if args.output_bundle.exists():
        raise FileExistsError(args.output_bundle)
    parquet = next(args.scene.glob('assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave/*.parquet'))
    table = pq.read_table(parquet)
    cell = lambda key: np.asarray(table[key][0].as_py())
    names = table['object_body_names'][0].as_py()
    slot = next(i for i, n in enumerate(names) if n.endswith('_' + args.object))
    positions, quats = cell('object_body_position'), cell('object_body_wxyz')
    rotations = Rotation.from_quat(quats[:, slot][:, [1, 2, 3, 0]]).as_matrix()
    frame_names = table[f'{args.side}_robot_frame_names'][0].as_py()
    tips = [frame_names.index(f'{args.side}_{digit}_tip_site') for digit in ('thumb', 'index')]
    frames = cell(f'robot_{args.side}_frames')
    anchor = args.anchor
    start, end = args.hold
    if not 0 <= start <= anchor < end <= len(positions):
        raise ValueError('Invalid hold/anchor range')
    mesh = trimesh.load(args.scene / f'score_input/meshes/episode_{int(json.loads((args.scene / "report.json").read_text())["episode"]):06d}' / (args.object + '.obj'), force='mesh', process=False)
    vertices = np.asarray(mesh.vertices)
    local_tips = (frames[anchor, tips, :3] - positions[anchor, slot]) @ rotations[anchor]
    if args.target_local is None:
        delta, alignment = pinch_target(vertices, local_tips)
    else:
        target = np.asarray(args.target_local)
        if not np.isfinite(target).all() or np.any(target < vertices.min(0)) or np.any(target > vertices.max(0)):
            raise ValueError('Explicit target must lie within the mesh bounds')
        delta, alignment = target - local_tips.mean(axis=0), None
    if np.linalg.norm(delta) > 0.4:
        raise ValueError('Repair exceeds 40 cm; inspect the world alignment first')
    source = args.base_bundle or args.scene / 'bundle_cleaned'
    data = dict(np.load(source / 'result.npz'))
    # The loader may rigidly move the world. Recover that fixed transform from object 0.
    first = np.eye(4)
    first[:3, :3] = Rotation.from_quat(quats[0, 0, [1, 2, 3, 0]]).as_matrix()
    first[:3, 3] = positions[0, 0]
    world = first @ np.linalg.inv(data['object_to_world_transform'][0])
    predicted = np.einsum('ij,njk->nik', world, data['object_to_world_transform'])
    actual_rot = Rotation.from_quat(quats[:, 0][:, [1, 2, 3, 0]]).as_matrix()
    if not np.allclose(predicted[:, :3, 3], positions[:, 0], atol=1e-5) or not np.allclose(predicted[:, :3, :3], actual_rot, atol=1e-5):
        raise ValueError('Bundle and processed reference do not share one rigid world transform')
    weight = np.zeros(len(positions))
    weight[start:end] = np.minimum(1, np.minimum(np.arange(1, end-start+1), np.arange(end-start, 0, -1)) / (args.blend + 1))
    offset = np.einsum('nij,j->ni', rotations, delta) * weight[:, None]
    raw_offset = offset @ np.linalg.inv(world[:3, :3]).T
    camera = data['camera_to_world_transform'].astype(float)
    cam_offset = np.einsum('nji,nj->ni', camera[:, :3, :3], raw_offset)
    if args.rigid_grip:
        # Use the exact wrist convention of the loader, including its MANO rest offset.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from run_load_retarget import mano_rest_wrist
        j0 = mano_rest_wrist(args.side, data[f'hand_{args.side}_betas'].reshape(10))
        mirror = np.diag([1., -1., -1.])
        hand_r = Rotation.from_rotvec(data[f'hand_{args.side}_wrist_orient_in_camera']).as_matrix()
        if args.side == 'left':
            hand_r = mirror @ hand_r @ mirror
        old_r = world[:3, :3] @ camera[:, :3, :3] @ hand_r
        old_t = (np.einsum('nij,nj->ni', camera[:, :3, :3], data[f'hand_{args.side}_wrist_trans_in_camera'] + j0) + camera[:, :3, 3]) @ world[:3, :3].T + world[:3, 3]
        grip_r = rotations[anchor].T @ old_r[anchor]
        grip_t = rotations[anchor].T @ (old_t[anchor] - positions[anchor, slot]) + delta
        target_r = rotations @ grip_r
        target_t = np.einsum('nij,j->ni', rotations, grip_t) + positions[:, slot]
        old_rot = Rotation.from_matrix(old_r)
        new_r = (old_rot * Rotation.from_rotvec((old_rot.inv() * Rotation.from_matrix(target_r)).as_rotvec() * weight[:, None])).as_matrix()
        new_t = old_t * (1-weight[:, None]) + target_t * weight[:, None]
        raw_r = np.linalg.inv(world[:3, :3]) @ new_r
        raw_t = (new_t - world[:3, 3]) @ np.linalg.inv(world[:3, :3]).T
        camera_r = np.einsum('nji,njk->nik', camera[:, :3, :3], raw_r)
        if args.side == 'left':
            camera_r = mirror @ camera_r @ mirror
        camera_t = np.einsum('nji,nj->ni', camera[:, :3, :3], raw_t - camera[:, :3, 3]) - j0
        active = weight > 0
        data[f'hand_{args.side}_wrist_orient_in_camera'][active] = Rotation.from_matrix(camera_r[active]).as_rotvec()
        data[f'hand_{args.side}_wrist_trans_in_camera'][active] = camera_t[active]
    else:
        data[f'hand_{args.side}_wrist_trans_in_camera'] = (data[f'hand_{args.side}_wrist_trans_in_camera'] + cam_offset).astype(np.float32)
    hand_cam = data[f'hand_{args.side}_wrist_to_camera_transform'].copy()
    active = weight > 0
    hand_cam[active, :3, 3] = data[f'hand_{args.side}_wrist_trans_in_camera'][active]
    hand_cam[active, :3, :3] = Rotation.from_rotvec(data[f'hand_{args.side}_wrist_orient_in_camera'][active]).as_matrix()
    data[f'hand_{args.side}_wrist_to_camera_transform'] = hand_cam
    data[f'hand_{args.side}_wrist_to_world_transform'] = (camera @ hand_cam).astype(np.float32)
    shutil.copytree(source, args.output_bundle)
    np.savez(args.output_bundle / 'result.npz', **data)
    tree = cKDTree(vertices)
    report = dict(source_scene=str(args.scene), source_parquet=str(parquet), source_npz_sha256=hashlib.sha256((source/'result.npz').read_bytes()).hexdigest(),
                  side=args.side, object=args.object, hold=[start, end], anchor=anchor, blend=args.blend, rigid_grip=args.rigid_grip,
                  object_local_translation_m=delta.tolist(), translation_cm=float(np.linalg.norm(delta)*100), pinch_normal_alignment=alignment,
                  anchor_thumb_index_distance_before_cm=(tree.query(local_tips)[0]*100).tolist(), anchor_thumb_index_distance_after_cm=(tree.query(local_tips+delta)[0]*100).tolist(),
                  object_and_camera_arrays_unchanged=all(np.array_equal(data[k], np.load(source/'result.npz')[k]) for k in data if k.startswith(('object_', 'camera_'))))
    prior = source / 'contact_repairs.json'
    previous = json.loads(prior.read_text()) if prior.exists() else ([json.loads((source/'contact_repair.json').read_text())] if (source/'contact_repair.json').exists() else [])
    report['source_bundle'] = str(source)
    (args.output_bundle/'contact_repair.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output_bundle/'contact_repairs.json').write_text(json.dumps(previous+[report], indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
