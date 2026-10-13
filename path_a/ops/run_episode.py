#!/usr/bin/env python3
"""Run one inspected episode/object with provenance, a GPU lock and QA gates.

Exit zero means numeric checks passed; visual review is still mandatory.
The original NVIDIA stages handle resumption. Do not resume partially written
stage outputs until their coverage has been inspected and invalid files moved
aside with a recorded reason.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def capture(command, **kwargs):
    return subprocess.check_output(command, text=True, **kwargs).strip()


def write_json(path, record):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(record, indent=2) + '\n')
    tmp.replace(path)


def validate(out, expected_frames):
    import numpy as np
    import trimesh
    bundle = out / 'result_slam_gravity_aligned'
    for name in ['result.npz', 'mesh.obj', 'manifest.json', 'threejs_scene/index.html']:
        if not (bundle / name).is_file() or (bundle / name).stat().st_size == 0:
            raise ValueError(f'Missing or empty final artifact: {bundle / name}')
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for name in manifest['mesh'].get('materials', []) + manifest['mesh'].get('textures', []):
        if not (bundle / name).is_file():
            raise ValueError(f'Missing mesh material or texture: {name}')
    if manifest['n_frames'] != expected_frames:
        raise ValueError(f"Frame count mismatch: {manifest['n_frames']} != {expected_frames}")
    coverage = {}
    with np.load(bundle / 'result.npz', allow_pickle=False) as data:
        for role, prefix in [('camera', 'camera_to_world'), ('object', 'object_to_world'),
                             ('hand_left', 'hand_left_wrist_to_world'), ('hand_right', 'hand_right_wrist_to_world')]:
            valid = data[role + '_is_valid'].astype(bool)
            transforms = data[prefix + '_transform']
            if valid.shape != (expected_frames,) or transforms.shape != (expected_frames, 4, 4):
                raise ValueError(f'Invalid dimensions for {role}')
            coverage[role] = {'valid': int(valid.sum()), 'total': expected_frames}
            required = .95 if role in ('camera', 'object') else .5
            if valid.mean() < required:
                raise ValueError(f'Low {role} coverage: {valid.mean():.3f}; gate={required}')
            good = transforms[valid]
            if not np.isfinite(good).all() or not np.allclose(good[:, 3], [0, 0, 0, 1], atol=1e-4):
                raise ValueError(f'Invalid {role} transforms')
            rotation = good[:, :3, :3]
            if not np.allclose(np.linalg.det(rotation), 1, atol=.02) or not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=.02):
                raise ValueError(f'Non-rigid {role} rotation')
        if not bool(data['gravity_alignment_applied']):
            raise ValueError('Gravity alignment not applied')
    mesh = trimesh.load(bundle / 'mesh.obj', force='mesh', process=False)
    if not np.isfinite(mesh.vertices).all() or len(mesh.vertices) < 32 or (mesh.area_faces > 0).sum() < 32:
        raise ValueError('Empty, nonfinite or degenerate mesh')
    from PIL import Image
    masks = sorted((out / 'masks/1').glob('*.png'))
    if len(masks) != expected_frames:
        raise ValueError(f'Incomplete object masks: {len(masks)} != {expected_frames}')
    mask_pixels = [int(np.count_nonzero(np.asarray(Image.open(path)))) for path in masks]
    if np.mean(np.asarray(mask_pixels) > 0) < .95:
        raise ValueError('Object mask missing from more than 5% of frames')
    overlays = list(out.glob('*overlay*.mp4'))
    if len(overlays) < 2:
        raise ValueError('Missing diagnostic overlays')
    for path in overlays:
        if path.stat().st_size == 0:
            raise ValueError(f'Empty overlay: {path}')
        probe = json.loads(capture(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                  '-show_entries', 'stream=width,height,nb_frames', '-of', 'json', str(path)]))
        if not probe.get('streams'):
            raise ValueError(f'Undecodable overlay: {path}')
    return {'bundle': str(bundle), 'coverage': coverage, 'mesh_vertices': len(mesh.vertices),
            'mesh_faces': len(mesh.faces), 'mesh_extent_m': mesh.extents.tolist(),
            'object_mask_nonzero_pixels': mask_pixels,
            'overlays': [str(path) for path in overlays], 'visual_review': 'PENDING'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--object', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--image-overrides', type=Path, help='Explicit receipts for locally extended images')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    cfg = json.loads(args.config.read_text())
    prompt = cfg['objects'][args.object]
    video = repo / 'assets/data/track_3' / cfg['split'] / 'videos/chunk-000' / cfg['camera'] / f"episode_{cfg['episode']:06d}.mp4"
    assert sha(video) == cfg['video_sha256'], 'Input video hash mismatch'
    recon = repo / 'baseline/reconstruction'
    root = recon / 'data/outputs/wentacc_dev'
    root.mkdir(parents=True, exist_ok=True)
    if not Path('/mnt/v2d').is_mount() or not root.resolve().is_relative_to('/mnt/v2d'):
        raise RuntimeError('Output directory must be on the mounted data disk')
    lock = (root / '.single_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = root / f"ep{cfg['episode']:02d}" / args.object
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise RuntimeError('Output exists; inspect cached stages before --resume')
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free < 25 * 2**30:
        raise RuntimeError('Less than 25 GiB free on data disk')
    command = [sys.executable, '-u', 'modules/v2d_pipelines/run_ego_reconstruction.py',
               '--video', str(video), '--object_prompt', prompt, '--output_dir', str(out),
               '--reference_frame', str(cfg['reference_frame']), '--undistort', '--hand_tracking', 'hamer',
               '--run_droid_slam', '--run_gravity_alignment', '--run_gsplat_refinement',
               '--export_threejs_result', '--dev']
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = out / 'runs' / run_id
    run.mkdir(parents=True)
    images = json.loads((Path(__file__).parent / 'reconstruction_images_afe7dad.json').read_text())
    overrides = json.loads(args.image_overrides.read_text()) if args.image_overrides else {}
    provenance = {'started': now(), 'config': cfg, 'object': args.object, 'command': command,
                  'cwd': str(recon), 'commit': capture(['git', 'rev-parse', 'HEAD'], cwd=repo),
                  'branch': capture(['git', 'branch', '--show-current'], cwd=repo),
                  'git_status': capture(['git', 'status', '--porcelain'], cwd=repo),
                  'runner_sha256': sha(__file__), 'config_sha256': sha(args.config),
                  'image_lock_sha256': sha(Path(__file__).parent / 'reconstruction_images_afe7dad.json'),
                  'gpu': capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader']),
                  'reproducibility_note': 'Pinned code and images; upstream HaMeR route does not forward --seed to SAM3D; GPU determinism is not guaranteed.'}
    provenance['images'] = []
    patch = capture(['git', 'diff', 'HEAD', '--binary'], cwd=repo)
    (run / 'tracked_changes.diff').write_text(patch + '\n')
    provenance['tracked_changes_sha256'] = sha(run / 'tracked_changes.diff')
    for image in images['images']:
        actual = json.loads(capture(['docker', 'image', 'inspect', image['local_tag']]))[0]
        ref = image['image'] + '@' + image['platform_digest']
        override = overrides.get(image['local_tag'])
        if override:
            if actual['Id'] != override['image_id'] or override['base_reference'] != ref:
                raise RuntimeError('Derived image identity mismatch: ' + image['local_tag'])
            if sha(Path(__file__).parent / override['dockerfile']) != override['dockerfile_sha256']:
                raise RuntimeError('Derived Dockerfile changed since build')
        elif ref not in actual['RepoDigests']:
            raise RuntimeError('Image digest mismatch: ' + image['local_tag'])
        provenance['images'].append({'tag': image['local_tag'], 'id': actual['Id'], 'base_digest': ref, 'override': override})
    write_json(run / 'provenance.json', provenance)
    (run / 'pip_freeze.txt').write_text(capture([sys.executable, '-m', 'pip', 'freeze']) + '\n')
    status = {'episode': cfg['episode'], 'object': args.object, 'started': now(),
              'state': 'RUNNING', 'run': str(run), 'output': str(out), 'pid': os.getpid()}
    write_json(out / 'status.json', status)
    try:
        with (run / 'pipeline.log').open('w') as log:
            subprocess.run(command, cwd=recon, stdout=log, stderr=subprocess.STDOUT, check=True)
        status['state'] = 'VALIDATING'
        write_json(out / 'status.json', status)
        qa = validate(out, cfg['frames'])
        write_json(run / 'numeric_qa.json', qa)
        status['state'] = 'NEEDS_VISUAL_REVIEW'
    except Exception as exc:
        status['state'] = 'FAILED_EXECUTION' if isinstance(exc, subprocess.CalledProcessError) else 'FAILED_VALIDATION'
        status['reason'] = str(exc)
        raise
    finally:
        status['updated'] = now()
        write_json(out / 'status.json', status)
        with (root / 'episode_status.jsonl').open('a') as f:
            f.write(json.dumps(status) + '\n')


if __name__ == '__main__':
    main()
