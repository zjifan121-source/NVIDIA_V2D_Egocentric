"""Materialize human-reviewed JSON annotations; never invent measurements."""
from pathlib import Path
import numpy as np

from inputs import put, resolve


def materialize(values, kind, base, stage):
    allowed = {
        'part_shape': {'semantic_vertices'},
        'contact_shape': {'table_plane', 'contact_annotation', 'validity'},
        'temporal_pose': {'table_plane', 'landmarks', 'measurement_receipt'},
    }[kind]
    if set(values) - allowed:
        raise ValueError('Unknown optimizer input key')
    result = {}
    directory = Path(stage) / ('annotations_' + kind)
    directory.mkdir(exist_ok=False)
    for key, value in values.items():
        if isinstance(value, str):
            result[key] = str(resolve(base, value))
            continue
        if key == 'table_plane':
            plane = np.asarray(value, float)
            if plane.shape != (4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3]) < 1e-12:
                raise ValueError('table_plane requires four finite world-plane coefficients')
            path = directory / 'plane.npy'
            np.save(path, plane / np.linalg.norm(plane[:3]))
        elif key == 'landmarks':
            if not isinstance(value, dict) or set(value) != {'frame_ids', 'points_camera', 'qualified', 'sigma_z'}:
                raise ValueError('Inline landmarks need frame_ids, points_camera, qualified, sigma_z')
            # Preserve qualification; do not cast integer flags to boolean or
            # turn arbitrary finite measurements into observability passes.
            ids = np.asarray(value['frame_ids'])
            qualified = np.asarray(value['qualified'])
            if ids.dtype.kind not in 'iu' or qualified.dtype != bool:
                raise ValueError('Landmark IDs must be integers and qualified flags must be boolean')
            path = directory / 'landmarks.npz'
            np.savez_compressed(path, frame_ids=ids, points_camera=np.asarray(value['points_camera'], float),
                                qualified=qualified, sigma_z=np.asarray(value['sigma_z'], float))
        elif key in ('semantic_vertices', 'contact_annotation', 'measurement_receipt'):
            if not isinstance(value, dict):
                raise ValueError('Expected annotation object or file path')
            if key == 'measurement_receipt' and not value.get('evidence'):
                raise ValueError('A measurement receipt needs input observability evidence')
            path = directory / (key + '.json')
            put(path, value)
        else:
            raise ValueError(key + ' must be a path')
        result[key] = str(path.resolve())
    return result


def scaffold(input_root, config_file, output_root):
    """Create non-executable review templates for all actual native object slots."""
    from pipeline import SCORED20
    from inputs import object_slots, validate_scene
    root, config_file = Path(input_root).resolve(), Path(config_file).resolve()
    recipes = []
    for episode in SCORED20:
        base = root / f'episode_{episode:06d}'
        bundle = base / 'bundle'
        n = validate_scene(bundle)
        path = base / 'recipe.json'
        if path.exists():
            raise FileExistsError('Do not overwrite reviewed recipe: ' + str(path))
        objects = []
        for slot in object_slots(bundle):
            objects.append(dict(slot=slot, name=f'object_{slot}',
                review=dict(accepted=False, evidence='Review identity, structure, coordinate pairing, and observed frame roles'),
                observations=dict(root=f'objects/slot_{slot}/observations', depth_encoding='native_inverse_u16')))
        recipes.append((path, dict(episode=episode, bundle='bundle', extra_files=[], objects=objects)))
    if config_file.exists():
        raise FileExistsError(config_file)
    # Validate the complete inventory before creating any template.
    config_file.parent.mkdir(parents=True, exist_ok=True)
    for path, recipe in recipes:
        put(path, recipe)
    import os
    put(config_file, dict(format_version=1, input_root=os.path.relpath(root, config_file.parent),
        output_root=str(output_root), recipe_pattern='episode_{episode:06d}/recipe.json',
        runtime=dict(gpu_python='envs/fp/bin/python', source_root='baseline/reconstruction/modules',
                     weights_dir='weights', runtime_info='runtime_info.json')))
    return [str(p) for p, _ in recipes]
