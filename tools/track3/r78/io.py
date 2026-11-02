"""Explicit units, hash-bound inputs, and portable mesh/pose artifacts."""
from pathlib import Path
import hashlib
import json
import shutil
import os

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .shared_scale import validate_pose


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read(path):
    return json.loads(Path(path).read_text())


def fresh(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def array(path):
    if Path(path).suffix == '.npy':
        return np.load(path, allow_pickle=False)
    return np.asarray(Image.open(path))


def intrinsics(path):
    if Path(path).suffix == '.npy':
        K = array(path)
    else:
        d = read(path)
        K = d.get('K', d.get('matrix'))
        if K is None:
            K = [[d['fx'], 0, d['cx']], [0, d['fy'], d['cy']], [0, 0, 1]]
    K = np.asarray(K, float)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or
            K[0, 0] <= 0 or K[1, 1] <= 0 or
            not np.allclose(K[2], [0, 0, 1]) or
            not np.allclose([K[0, 1], K[1, 0]], 0) or abs(np.linalg.det(K)) < 1e-12):
        raise ValueError('Invalid OpenCV pinhole K')
    return K


def pose(path):
    if Path(path).suffix == '.npy':
        return validate_pose(array(path))
    d = read(path)
    if 'matrix' in d:
        return validate_pose(d['matrix'])
    if not np.allclose(d.get('scale', [1, 1, 1]), 1):
        raise ValueError('Bake mesh scale before supplying rigid poses')
    q = np.asarray(d['rotation'])  # Native Transform3d quaternion is wxyz.
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = d['translation']
    return validate_pose(T)


def decode_depth(raw, encoding):
    if raw.ndim != 2:
        raise ValueError('Depth must be a single-channel image')
    if encoding == 'float_m_npy':
        if not np.issubdtype(raw.dtype, np.floating):
            raise ValueError('Expected floating point metres')
        return raw.astype(float)
    if encoding == 'v2d_inverse_uint16':
        if raw.dtype != np.uint16:
            raise ValueError('Expected uint16 inverse depth')
        depth = np.full(raw.shape, np.nan)
        valid = (raw > 0) & (raw < 65535)
        depth[valid] = 65535. / raw[valid] - 1.
        return depth
    raise ValueError('Unknown depth encoding; units are never guessed')


def check_review(review, mesh_hash):
    if (review.get('state') != 'PASS' or review.get('GT_used') is not False or
            review.get('mesh_sha256') != mesh_hash or
            not review.get('critical_features') or
            review.get('critical_missing_or_extraneous_count') != 0):
        raise ValueError('A hash-matched input-only structural PASS with no critical defects is required')


class Inputs:
    """All file reads must match the caller's frozen manifest, including poses."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.data = read(self.path)
        self.base = self.path.parent
        self.used = {self.path: sha(self.path)}
        d = self.data
        if d.get('schema_version') != 1:
            raise ValueError('Expected schema_version=1')
        for key in ('source_revision', 'source_code_commit', 'frame_grid',
                    'depth_encoding', 'depth_source', 'object'):
            if not d.get(key):
                raise ValueError('Missing input provenance: ' + key)
        if 'episode' not in d or not isinstance(d.get('sha256'), dict):
            raise ValueError('episode and sha256 are required')
        self.context = {k: d[k] for k in ('episode', 'object', 'source_revision',
                                        'source_code_commit', 'frame_grid',
                                        'depth_encoding', 'depth_source')}
        for key in ('source_revision', 'source_code_commit'):
            if str(d[key]).startswith('REPLACE_'):
                raise ValueError('Replace example provenance before running: ' + key)

    def file(self, name):
        p = (self.base / name).resolve()
        expected = self.data['sha256'].get(name)
        if expected is None or not p.is_file() or sha(p) != expected:
            raise ValueError('Missing input or SHA mismatch: ' + str(name))
        self.used[p] = expected
        return p

    def unchanged(self):
        for p, digest in self.used.items():
            if sha(p) != digest:
                raise ValueError('Input changed during execution: ' + str(p))

    def rows(self):
        rows = self.data.get('frames', [])
        ids = [r['id'] for r in rows]
        if (not ids or any(type(i) is not int or i < 0 for i in ids)
                or len(ids) != len(set(ids))):
            raise ValueError('Nonempty unique nonnegative integer frame IDs required')
        return rows

    def frame(self, row):
        paths = {k: self.file(row[k]) for k in ('rgb', 'depth', 'K', 'mask', 'validity')}
        rgb = np.asarray(Image.open(paths['rgb']).convert('RGB'))
        raw = array(paths['depth'])
        depth = decode_depth(raw, self.data['depth_encoding'])
        # Preserve the original FP adapter's float32 inverse-code arithmetic;
        # fitting and pointmap preparation retain their float64 decode.
        fp_depth = depth.astype(np.float32)
        if self.data['depth_encoding'] == 'v2d_inverse_uint16':
            code_valid = (raw > 0) & (raw < 65535)
            fp_depth = np.zeros(raw.shape, np.float32)
            fp_depth[code_valid] = 65535. / raw[code_valid].astype(np.float32) - 1.
        mask, validity = array(paths['mask']) > 0, array(paths['validity']) > 0
        if rgb.shape != (*depth.shape, 3) or mask.shape != depth.shape or validity.shape != depth.shape:
            raise ValueError('RGB/depth/mask/validity pixel grids differ')
        K = intrinsics(paths['K'])
        valid = mask & validity & np.isfinite(depth) & (depth > 0)
        if valid.sum() < 500:
            raise ValueError('Insufficient valid object depth')
        return dict(id=row['id'], rgb=rgb, depth=depth, fp_depth=fp_depth, mask=mask, validity=validity,
                    valid=valid, K=K, shape=depth.shape,
                    input_sha256={k: sha(p) for k, p in paths.items()})

    def mesh(self):
        d = self.data
        mesh = self.file(d['mesh'])
        if mesh.suffix.lower() != '.obj' or d.get('mesh_scale_baked') is not True:
            raise ValueError('A scale-baked OBJ is required')
        if d.get('pose_convention') != 'original_mesh_to_camera':
            raise ValueError('Expected original_mesh_to_camera convention')
        check_review(d.get('structure_review', {}), sha(mesh))
        return mesh


def mesh_dependencies(mesh):
    """Collect only referenced appearance files, not unrelated cached images."""
    root = Path(mesh).parent.resolve()
    files = []
    def local(name, base):
        p = (base / name).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            raise ValueError('Mesh appearance reference missing or outside mesh directory: ' + name)
        return p
    for line in Path(mesh).read_text().splitlines():
        if line.startswith('mtllib '):
            mtl = local(line.split(maxsplit=1)[1], root)
            files.append(mtl)
            for material in mtl.read_text().splitlines():
                parts = material.split()
                if parts and (parts[0].startswith('map_') or parts[0] in ('bump', 'disp', 'decal', 'norm')):
                    files.append(local(parts[-1], mtl.parent))
    return list(dict.fromkeys(files))


def copy_appearance(source, dest):
    for p in mesh_dependencies(source):
        q = Path(dest).parent / p.relative_to(Path(source).parent)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)


def seal(root, metadata, name='artifact.json'):
    """Bind exact exported mesh, poses, metrics and code to one artifact."""
    root = Path(root)
    metadata = dict(metadata)
    metadata['sha256'] = {str(p.relative_to(root)): sha(p) for p in root.rglob('*')
                          if p.is_file() and p.name != name}
    metadata['implementation_sha256'] = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}
    write(root / name, metadata)
    return metadata


def artifact(path, states):
    path = Path(path).resolve()
    d = read(path)
    if d.get('state') not in states or not d.get('sha256'):
        raise ValueError('Artifact is incomplete or not accepted: ' + str(path))
    for rel, digest in d['sha256'].items():
        p = (path.parent / rel).resolve()
        if not p.is_relative_to(path.parent) or not p.is_file() or sha(p) != digest:
            raise ValueError('Artifact SHA mismatch: ' + rel)
    return d


def freeze(config, output):
    """Resolve a user config and snapshot its inputs without rebinding old hashes."""
    config, output = Path(config).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    d = read(config)
    expected, frozen = d.get('sha256', {}), {}
    def bind(name):
        p = (config.parent / name).resolve()
        digest = sha(p)
        if name in expected and expected[name] != digest:
            raise ValueError('Refusing to refresh a changed frozen input: ' + name)
        rel = os.path.relpath(p, output.parent)
        frozen[rel] = digest
        return rel
    if isinstance(d.get('structure_review'), str):
        review_path = (config.parent / d['structure_review']).resolve()
        bind(d['structure_review'])
        d['structure_review'] = read(review_path)
    for key in ('mesh', 'pose_artifact'):
        if key in d:
            d[key] = bind(d[key])
    for row in d['frames']:
        for key in ('rgb', 'K', 'depth', 'validity', 'mask', 'pose'):
            if key in row:
                row[key] = bind(row[key])
    d['sha256'] = frozen
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: concurrent invocations cannot replace a frozen manifest.
    with output.open('x') as stream:
        json.dump(d, stream, indent=2, allow_nan=False)
        stream.write('\n')
