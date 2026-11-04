"""Explicit native-cache adapters and paired scene export. CPU only."""
from pathlib import Path
import importlib.util
import json
import re
import shlex
import shutil

import cv2
import numpy as np
import trimesh

SCRIPTS = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('refinement_contract', SCRIPTS / 'metric_scene_refinement/common.py')
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)
sha, put, fields, mesh_name = contract.sha, contract.put, contract.fields, contract.mesh_name
CONVENTION = 'T_camera_from_original_mesh_metres'


def read(path):
    return json.loads(Path(path).read_text())


def resolve(base, value):
    return (Path(base) / value).resolve()


def object_slots(bundle):
    with np.load(Path(bundle) / 'result.npz', allow_pickle=False) as z:
        slots = sorted(0 if k == 'object_to_camera_transform' else int(k.split('_')[1])
                       for k in z.files if re.fullmatch(r'object(?:_\d+)?_to_camera_transform', k))
    if not slots or slots != list(range(len(slots))):
        raise ValueError('Expected all contiguous native object slots, starting at0')
    return slots


def arrays(bundle):
    with np.load(Path(bundle) / 'result.npz', allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def validate_scene(bundle):
    contract.bundle_files(bundle)
    data = arrays(bundle)
    camera = data['camera_to_world_transform']
    contract.validate_poses(camera, 'camera')
    n = len(camera)
    if n < 3 or not np.array_equal(data['camera_is_valid'], np.ones(n, bool)):
        raise ValueError('Complete common camera required; no implicit interpolation')
    for slot in object_slots(bundle):
        ck, wk = fields(slot)
        prefix = ck.removesuffix('_to_camera_transform')
        contract.validate_poses(data[ck], ck)
        contract.validate_poses(data[wk], wk)
        if len(data[ck]) != n or len(data[wk]) != n or not np.allclose(camera @ data[ck], data[wk], atol=1e-5):
            raise ValueError('Object/camera frame count or world closure mismatch')
        if float(data[prefix + '_scale']) != 1.:
            raise ValueError('Input mesh must already have metric scale baked exactly once')
        if not np.array_equal(data[prefix + '_is_valid'], np.ones(n, bool)):
            raise ValueError('Incomplete object poses are not a compatible input bundle')
        mesh = trimesh.load(Path(bundle) / mesh_name(slot), force='mesh', process=False)
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError('Invalid mesh')
    return n


def dependencies(mesh):
    """OBJ/MTL closure, including nested textures; paths cannot escape root."""
    mesh = Path(mesh).resolve()
    root, pending, found = mesh.parent, [mesh], set()
    while pending:
        p = pending.pop().resolve()
        if not p.is_relative_to(root) or not p.is_file():
            raise ValueError('Missing or escaping mesh dependency: ' + str(p))
        if p in found:
            continue
        found.add(p)
        if p.suffix.lower() in ('.obj', '.mtl'):
            for line in p.read_text().splitlines():
                words = shlex.split(line, comments=True)
                if words and words[0] == 'mtllib':
                    pending.extend(p.parent / w for w in words[1:])
                elif words and (words[0].startswith('map_') or words[0] in ('bump', 'disp', 'decal', 'norm')):
                    if len(words) != 2:
                        raise ValueError('Convert option-bearing material maps before running')
                    pending.append(p.parent / words[1])
    return sorted(found)


def scene_files(bundle, extra_files=()):
    root = Path(bundle).resolve()
    files = set(contract.bundle_files(root))
    # Tables are scene assets, not regenerated independently for each candidate.
    for pattern in ('table*.npy', 'table*.npz', 'table*.json', 'table*.ply', 'table*.stl', 'table*.obj', 'metadata.json'):
        for p in root.glob(pattern):
            files.update(dependencies(p) if p.suffix == '.obj' else [p])
    for name in extra_files:
        p = resolve(root, name)
        if not p.is_relative_to(root) or not p.is_file():
            raise ValueError('Missing/escaping extra scene asset')
        files.add(p)
    return sorted(files)


def copy_scene(source, output, extra_files=()):
    source, output = Path(source).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for p in scene_files(source, extra_files):
        q = output / p.relative_to(source)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)


def install_mesh(mesh, bundle, slot):
    """Namespace dependencies so replacing one object cannot recolor a partner."""
    mesh, bundle = Path(mesh).resolve(), Path(bundle)
    namespace = Path('optimized_assets') / f'slot_{slot}'
    for p in dependencies(mesh):
        if p == mesh:
            continue
        q = bundle / namespace / p.relative_to(mesh.parent)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
    lines = []
    for line in mesh.read_text().splitlines(keepends=True):
        words = shlex.split(line, comments=True)
        if words and words[0] == 'mtllib':
            line = 'mtllib ' + ' '.join(shlex.quote(str(namespace / w)) for w in words[1:]) + '\n'
        lines.append(line)
    (bundle / mesh_name(slot)).write_text(''.join(lines))


def same(a, b):
    return np.array_equal(a, b, equal_nan=True) if np.issubdtype(a.dtype, np.number) else np.array_equal(a, b)


def export_track(source, mesh, pose_file, pair_file, output, slot, extra_files=()):
    pair = read(pair_file)
    if pair.get('mesh_sha256') != sha(mesh) or pair.get('poses_sha256') != sha(pose_file):
        raise ValueError('Tracking mesh/pose digest mismatch; never attach an old pose cache')
    if pair.get('additional_scale') != 1.0 or pair.get('mesh_scale_already_baked') is not True:
        raise ValueError('Expected metric tracking without another scale application')
    source_data = arrays(source)
    n = len(source_data['camera_to_world_transform'])
    with np.load(pose_file, allow_pickle=False) as z:
        ids, poses = z['frame_ids'], z['public_poses']
    if not np.array_equal(ids, np.arange(n)) or poses.shape != (n, 4, 4) or pair.get('frame_count') != n:
        raise ValueError('Full chronological tracking required; partial cache is not exportable')
    contract.validate_poses(poses, 'FP output')
    copy_scene(source, output, extra_files)
    install_mesh(mesh, output, slot)
    ck, wk = fields(slot)
    result = {k: v.copy() for k, v in source_data.items()}
    result[ck] = poses.copy()
    result[wk] = result['camera_to_world_transform'] @ poses
    np.savez_compressed(Path(output) / 'result.npz', **result)
    for k, v in source_data.items():
        if k not in (ck, wk) and not same(v, result[k]):
            raise ValueError('Unrequested scene field changed: ' + k)
    validate_scene(output)
    put(Path(output) / 'MESH_POSE_PAIR.json', dict(
        slot=slot, mesh_sha256=sha(Path(output) / mesh_name(slot)),
        result_sha256=sha(Path(output) / 'result.npz'),
        tracking_pair_sha256=sha(pair_file), frame_count=n, convention=CONVENTION,
        unchanged_fields=[k for k in source_data if k not in (ck, wk)]))


def adapt_observations(spec, base, n, output):
    """Explicit encoding/grid adapter; never estimates calibration or depth."""
    allowed = {'root', 'K', 'K_image_size', 'rgb', 'depth', 'mask', 'depth_encoding', 'resize_to_depth', 'validity'}
    if set(spec) - allowed:
        raise ValueError('Unknown observation adapter setting')
    root = resolve(base, spec['root'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('rgb', 'depth', 'masks', 'validity'):
        (output / name).mkdir()
    patterns = {k: spec.get(k, default) for k, default in
                [('rgb', 'rgb/{frame:06d}.png'), ('depth', 'depth/{frame:06d}.png'), ('mask', 'masks/{frame:06d}.png')]}
    enc = spec['depth_encoding']
    if enc not in ('native_inverse_u16', 'metres_npy', 'millimetres_u16'):
        raise ValueError('Explicit supported depth_encoding is required')
    kp = resolve(root, spec.get('K', 'K_fp.json'))
    k = np.load(kp, allow_pickle=False).tolist() if kp.suffix == '.npy' else read(kp)
    if isinstance(k, list):
        matrix = np.asarray(k, float)
        if matrix.shape != (3, 3) or not np.allclose(matrix[2], [0, 0, 1]) or matrix[0, 1] != 0 or matrix[1, 0] != 0:
            raise ValueError('Expected zero-skew pinhole K')
        k = dict(fx=matrix[0, 0], fy=matrix[1, 1], cx=matrix[0, 2], cy=matrix[1, 2])
    size = spec.get('K_image_size', [k.get('width'), k.get('height')])
    if len(size) != 2 or any(type(x) is not int or x <= 0 for x in size):
        raise ValueError('K_image_size=[width,height] required for matrix/size-less intrinsics')
    hashes = {str(kp): sha(kp)}
    target = None
    aliases = {key: set() for key in patterns}
    for frame in range(n):
        paths = {key: resolve(root, pattern.format(frame=frame)) for key, pattern in patterns.items()}
        for key, p in paths.items():
            if p in aliases[key]:
                raise ValueError('Repeated input frame path: explicit complete timeline required')
            aliases[key].add(p)
            hashes[str(p)] = sha(p)
        rgb = cv2.imread(str(paths['rgb']), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(paths['mask']), cv2.IMREAD_UNCHANGED)
        depth = np.load(paths['depth'], allow_pickle=False) if enc == 'metres_npy' else cv2.imread(str(paths['depth']), -1)
        if rgb is None or mask is None or depth is None or depth.ndim != 2:
            raise ValueError(f'Unreadable RGB/mask/depth at frame{frame}')
        if mask.ndim != 2 or not np.isin(mask, [0, 1, 255]).all() or not mask.any():
            raise ValueError('Expected nonempty binary target mask')
        if target is None:
            target = depth.shape
        if depth.shape != target:
            raise ValueError('Depth grid varies over time')
        for key, image in [('rgb', rgb), ('mask', mask)]:
            if image.shape[:2] != target and spec.get('resize_to_depth') is not True:
                raise ValueError('Grid mismatch: explicit resize_to_depth is required for same-camera resize')
            if image.shape[:2] != target and list(image.shape[:2][::-1]) != size:
                raise ValueError('Resized RGB/mask must start on the declared K grid')
        rgb = cv2.resize(rgb, target[::-1], interpolation=cv2.INTER_LINEAR) if rgb.shape[:2] != target else rgb
        mask = cv2.resize(mask, target[::-1], interpolation=cv2.INTER_NEAREST) if mask.shape != target else mask
        if enc == 'native_inverse_u16':
            if depth.dtype != np.uint16:
                raise ValueError('Native inverse depth must be uint16')
            code = depth
        else:
            if enc == 'millimetres_u16' and depth.dtype != np.uint16:
                raise ValueError('Millimetre PNG must be uint16')
            metric = depth.astype(float) / (1000. if enc == 'millimetres_u16' else 1.)
            good = np.isfinite(metric) & (metric > 0)
            code = np.zeros(target, np.uint16)
            values = np.rint(65535. / (metric[good] + 1.))
            if np.any((values <= 0) | (values >= 65535)):
                raise ValueError('Metric depth outside representable native encoding range')
            code[good] = values.astype(np.uint16)
        valid = (code > 0) & (code < 65535)
        if spec.get('validity'):
            vp = resolve(root, spec['validity'].format(frame=frame))
            hashes[str(vp)] = sha(vp)
            supplied = cv2.imread(str(vp), -1)
            if supplied is None or supplied.shape != target or not np.isin(supplied, [0, 1, 255]).all():
                raise ValueError('Validity must already match the depth grid')
            valid &= supplied > 0
        for name, image in [('rgb', rgb), ('masks', mask), ('depth', code), ('validity', valid.astype(np.uint8))]:
            if not cv2.imwrite(str(output / name / f'{frame:06d}.png'), image):
                raise IOError('Could not write adapted observation')
    sx, sy = target[1] / size[0], target[0] / size[1]
    if (sx != 1 or sy != 1) and spec.get('resize_to_depth') is not True:
        raise ValueError('K grid differs from depth grid')
    intrinsics = dict(fx=k['fx'] * sx, fy=k['fy'] * sy,
                      cx=(k['cx'] + .5) * sx - .5, cy=(k['cy'] + .5) * sy - .5,
                      width=target[1], height=target[0])
    if not np.isfinite(list(intrinsics.values())).all() or min(intrinsics['fx'], intrinsics['fy']) <= 0:
        raise ValueError('Invalid adapted K')
    put(output / 'K_fp.json', intrinsics)
    put(output / 'ADAPTER.json', dict(source_sha256=hashes, depth_encoding=enc,
        validity='supplied AND native finite encoding' if spec.get('validity') else 'native finite encoding only; not stereo confidence',
        resize='same-camera half-pixel pinhole resize only; no undistortion or cross-camera reprojection',
        native_inverse_depth_quantized=enc != 'native_inverse_u16', frame_count=n))
    return hashes
