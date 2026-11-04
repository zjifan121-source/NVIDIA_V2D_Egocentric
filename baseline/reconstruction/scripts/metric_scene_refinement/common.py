"""Input contracts and immutable provenance for opt-in CPU scene refinement."""
from pathlib import Path
import hashlib
import json
import shlex

import cv2
import numpy as np
import trimesh

KINDS = ('part_shape', 'contact_shape', 'temporal_pose')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def fields(slot):
    prefix = 'object' if slot == 0 else f'object_{slot}'
    return prefix + '_to_camera_transform', prefix + '_to_world_transform'


def mesh_name(slot):
    return 'mesh.obj' if slot == 0 else f'mesh_{slot}.obj'


def validate_poses(value, name):
    value = np.asarray(value)
    if value.ndim != 3 or value.shape[1:] != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f'{name}: expected finite [frames,4,4]')
    if not np.allclose(value[:, 3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError(f'{name}: invalid homogeneous row')
    r = value[:, :3, :3]
    if not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=1e-5) or not np.allclose(np.linalg.det(r), 1, atol=1e-5):
        raise ValueError(f'{name}: expected SE(3), with metric scale baked into mesh')


def bundle_files(root):
    """Only the native bundle and OBJ-referenced material/texture dependencies."""
    root = Path(root).resolve()
    selected = {root / 'result.npz'}
    pending = list(root.glob('mesh*.obj'))
    if not pending:
        raise ValueError('No native mesh OBJ in bundle')
    while pending:
        p = pending.pop()
        p = p.resolve()
        if not p.is_relative_to(root) or not p.is_file():
            raise ValueError(f'Missing/escaping mesh dependency: {p.name}')
        if p in selected:
            continue
        selected.add(p)
        if p.suffix.lower() not in ('.obj', '.mtl'):
            continue
        for line in p.read_text().splitlines():
            tokens = shlex.split(line, comments=True)
            if tokens and tokens[0] == 'mtllib':
                pending.extend(p.parent / x for x in tokens[1:])
            elif tokens and (tokens[0].startswith('map_') or tokens[0] in ('bump', 'disp', 'decal', 'norm')):
                # Repository exporters use a plain texture filename. Refuse
                # option-bearing MTLs rather than silently dropping assets.
                if len(tokens) != 2:
                    raise ValueError('Texture map options require explicit conversion before refinement')
                pending.append(p.parent / tokens[1])
    if any(not p.is_file() for p in selected):
        raise ValueError('Incomplete native bundle')
    return sorted(selected)


def validate_spec(spec):
    kind = spec['kind']
    if kind not in KINDS:
        raise ValueError('Unknown refinement kind')
    review = spec.get('structure_review', {})
    if review.get('accepted') is not True or not review.get('evidence'):
        raise ValueError('Explicit input-side structure/identity review is required')
    if spec.get('coordinate_convention') != 'T_camera_from_original_mesh_metres':
        raise ValueError('Expected original-mesh metric pose convention')
    inputs = spec['inputs']
    root = Path(inputs['bundle'])
    files = bundle_files(root)
    with np.load(root / 'result.npz', allow_pickle=False) as z:
        bundle = {k: z[k] for k in z.files}
    slot = spec['object_slot']
    if type(slot) is not int or slot < 0:
        raise ValueError('object_slot must be a nonnegative integer')
    ck, wk = fields(slot)
    prefix = ck.removesuffix('_to_camera_transform')
    if slot < 0 or ck not in bundle or wk not in bundle:
        raise ValueError('Requested object slot is absent')
    n = len(bundle[ck])
    if n < 3:
        raise ValueError('At least three frames required')
    validate_poses(bundle[ck], ck)
    validate_poses(bundle[wk], wk)
    validate_poses(bundle['camera_to_world_transform'], 'camera_to_world_transform')
    if len(bundle['camera_to_world_transform']) != n or len(bundle[wk]) != n:
        raise ValueError('Camera/object frame mismatch')
    if not np.allclose(bundle['camera_to_world_transform'] @ bundle[ck], bundle[wk], atol=1e-5):
        raise ValueError('Camera/world pose closure mismatch')
    for k in [prefix + '_is_valid', 'camera_is_valid']:
        if np.shape(bundle[k]) != (n,) or not np.asarray(bundle[k], bool).all():
            raise ValueError(f'{k}: incomplete input; no silent interpolation/cache reuse')
    if float(bundle[prefix + '_scale']) != 1:
        raise ValueError('Bake object scale once before refinement')
    mesh = trimesh.load(root / mesh_name(slot), force='mesh', process=False)
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError('Invalid target mesh')
    if sha(root / mesh_name(slot)) != spec['mesh_pose_pair']['mesh_sha256'] or sha(root / 'result.npz') != spec['mesh_pose_pair']['result_sha256']:
        raise ValueError('Mesh and pose pair does not match the reviewed inputs')
    p = spec['parameters']
    expected = {
        'part_shape': {'support', 'heldout', 'maximum_iterations', 'maximum_seconds'},
        'contact_shape': {'support', 'heldout', 'graph_nodes', 'maximum_seconds'},
        'temporal_pose': {'calibration_frames', 'fit_frames', 'validation_frames',
                          'contact_frames', 'landmark_groups', 'maximum_seconds'},
    }[kind]
    if set(p) != expected:
        raise ValueError('Unknown or missing parameter; numerical weights are fixed by implementation')
    if type(p['maximum_seconds']) is not int or not 0 < p['maximum_seconds'] <= 2700:
        raise ValueError('CPU budget must be an integer in 1..2700 seconds')
    if kind == 'part_shape' and p['maximum_iterations'] != 15:
        raise ValueError('Expected the fixed 15-update shape budget')
    if kind == 'contact_shape' and p['graph_nodes'] != 16:
        raise ValueError('Expected the fixed 16-node graph')
    supports = p.get('support', [])
    heldout = p.get('heldout', [])
    if kind != 'temporal_pose' and (not supports or not heldout or set(supports) & set(heldout)):
        raise ValueError('Disjoint support and heldout frame lists required')
    frame_ids = list(range(n)) if kind == 'temporal_pose' else supports + heldout
    if len(set(frame_ids)) != len(frame_ids) or any(type(i) is not int or not 0 <= i < n for i in frame_ids):
        raise ValueError('Invalid or duplicate frame IDs')
    observation_root = Path(inputs['observations'])
    kfile = observation_root / 'K_fp.json'
    files.append(kfile)
    k = json.loads(kfile.read_text())
    if not all(np.isfinite(k[x]) for x in ('fx', 'fy', 'cx', 'cy')) or min(k['fx'], k['fy']) <= 0:
        raise ValueError('Invalid intrinsics')
    shape = None
    for frame in frame_ids:
        names = ['masks'] if kind == 'temporal_pose' else ['rgb', 'depth', 'masks']
        for name in names:
            f = observation_root / name / f'{frame:06d}.png'
            image = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f'Missing frame input: {name}/{frame:06d}.png')
            if shape is None:
                shape = image.shape[:2]
            if image.shape[:2] != shape:
                raise ValueError('RGB/depth/mask pixel-grid mismatch')
            if name == 'depth' and (image.ndim != 2 or image.dtype != np.uint16):
                raise ValueError('Expected native inverse-depth uint16 PNG')
            if name == 'masks' and (image.ndim != 2 or not np.isin(image, [0, 1, 255]).all() or not image.any()):
                raise ValueError('Expected nonempty binary object mask')
            files.append(f)
        if kind == 'contact_shape':
            f = Path(inputs['validity']) / f'{frame:06d}.png'
            image = cv2.imread(str(f), 0)
            if image is None or image.shape != shape or not np.isin(image, [0, 1, 255]).all():
                raise ValueError('Missing/incompatible frozen validity support')
            files.append(f)
    if 'width' in k and (k['height'], k['width']) != shape:
        raise ValueError('Intrinsic image size differs from observations')
    if kind == 'part_shape':
        files.append(Path(inputs['semantic_vertices']))
        marks = json.loads(Path(inputs['semantic_vertices']).read_text())
        if set(marks) != {'parts', 'center'} or len(marks['parts']) != 2 or any(type(i) is not int or not 0 <= i < len(mesh.vertices) for i in marks['parts'] + [marks['center']]):
            raise ValueError('Expected two input-reviewed part vertices and one center vertex')
    else:
        plane_file = Path(inputs['table_plane'])
        plane = np.load(plane_file, allow_pickle=False)
        if plane.shape != (4,) or not np.isfinite(plane).all() or not np.isclose(np.linalg.norm(plane[:3]), 1, atol=1e-6):
            raise ValueError('Expected normalized world-space plane')
        files.append(plane_file)
    if kind == 'contact_shape':
        f = Path(inputs['contact_annotation'])
        ann = json.loads(f.read_text())
        if ann.get('frame') not in supports or not ann.get('source'):
            raise ValueError('Observed contact annotation must be a support frame')
        u, v = ann['mask_boundary_pixel']
        if not (0 <= u < shape[1] and 0 <= v < shape[0]):
            raise ValueError('Contact pixel outside image')
        files.append(f)
    if kind == 'temporal_pose':
        for key in ['landmarks', 'measurement_receipt', 'calibration']:
            files.append(Path(inputs[key]))
        receipt = json.loads(Path(inputs['measurement_receipt']).read_text())
        if receipt.get('state') != 'OBSERVABILITY_PASS':
            raise ValueError('Stereo-landmark observability has not passed')
        for key in ['calibration_frames', 'fit_frames', 'validation_frames', 'contact_frames']:
            values = p[key]
            if not values or len(values) != len(set(values)) or any(type(t) is not int or not 0 <= t < n for t in values):
                raise ValueError('Invalid ' + key)
        if set(p['fit_frames']) & set(p['validation_frames']) or not set(p['calibration_frames']) <= set(p['fit_frames']):
            raise ValueError('Landmark calibration/fit/validation frame leakage')
        with np.load(inputs['landmarks'], allow_pickle=False) as z:
            ids, xyz, qualified, sigma = (z[k] for k in ['frame_ids', 'points_camera', 'qualified', 'sigma_z'])
        if ids.ndim != 1 or len(set(ids.tolist())) != len(ids) or np.any(ids < 0) or np.any(ids >= n) or ids.dtype.kind not in 'iu':
            raise ValueError('Invalid landmark timeline')
        if xyz.ndim != 3 or xyz.shape[0] != len(ids) or xyz.shape[2] != 3 or qualified.shape != xyz.shape[:2] or sigma.shape != qualified.shape:
            raise ValueError('Landmark shape mismatch')
        if qualified.dtype != bool:
            raise ValueError('Landmark qualification must be a boolean array')
        if not np.isfinite(xyz[qualified]).all() or not np.isfinite(sigma[qualified]).all() or np.any(sigma[qualified] < 0):
            raise ValueError('Invalid qualified landmark measurements')
        if len(p['landmark_groups']) != xyz.shape[1]:
            raise ValueError('One weighting group per landmark required')
        if any(type(g) is not int for g in p['landmark_groups']):
            raise ValueError('Landmark groups must be integer labels')
        if any((qualified[np.isin(ids, p['calibration_frames']), j]).sum() < 2
               for j in range(xyz.shape[1])):
            raise ValueError('At least two qualified calibration observations per landmark required')
        for key in ['fit_frames', 'validation_frames']:
            if not qualified[np.isin(ids, p[key])].any():
                raise ValueError('No qualified observations for ' + key)
        contact = np.asarray(json.loads(Path(inputs['calibration']).read_text())['contact_point_local_m'])
        if contact.shape != (3,) or not np.isfinite(contact).all():
            raise ValueError('Expected finite input-derived contact point in matching mesh coordinates')
    return sorted(set(f.resolve() for f in files)), n


def freeze(spec_file, output):
    spec_file, output = Path(spec_file).resolve(), Path(output)
    if output.exists():
        raise ValueError('Frozen manifest already exists')
    spec = json.loads(spec_file.read_text())
    if 'file_sha256' in spec:
        raise ValueError('Input spec is already frozen')
    spec['inputs'] = {k: str((spec_file.parent / v).resolve()) for k, v in spec['inputs'].items()}
    files, n = validate_spec(spec)
    code = sorted(Path(__file__).parent.glob('*.py'))
    spec.update(format_version=1, frame_count=n,
                file_sha256={str(p): sha(p) for p in files},
                code_sha256={p.name: sha(p) for p in code})
    put(output, spec)
    return spec


def load_frozen(path):
    spec = json.loads(Path(path).read_text())
    if spec.get('format_version') != 1 or not spec.get('file_sha256') or not spec.get('code_sha256'):
        raise ValueError('Use freeze before run')
    for name, digest in spec['code_sha256'].items():
        if Path(name).name != name or sha(Path(__file__).parent / name) != digest:
            raise ValueError('Refinement implementation changed after freeze')
    for filename, digest in spec['file_sha256'].items():
        if sha(filename) != digest:
            raise ValueError('Input changed after freeze: ' + Path(filename).name)
    files, n = validate_spec(spec)
    if set(map(str, files)) != set(spec['file_sha256']) or n != spec['frame_count']:
        raise ValueError('Frozen file inventory changed')
    return spec
