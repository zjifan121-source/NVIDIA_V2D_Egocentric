"""One orchestrator for 5454d57 optimizers, FP, and complete native scenes.

No default production workflow is changed. No score, GT, or scan is read.
"""
from pathlib import Path
import argparse
import datetime
import importlib.metadata
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import trimesh

from inputs import (SCRIPTS, CONVENTION, adapt_observations, arrays, contract,
                    copy_scene, dependencies, export_track, fields, mesh_name,
                    object_slots, put, read, resolve, same, scene_files, sha, validate_scene)
from annotations import materialize, scaffold

SCORED20 = [1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 35, 36, 37, 38]
REFINE = SCRIPTS / 'metric_scene_refinement'
TRACK = SCRIPTS / 'metric_mesh_tracking'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def inventory(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(Path(root).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def code_hashes():
    files = list(Path(__file__).parent.glob('*.py'))
    files += list(REFINE.glob('*.py')) + list(TRACK.glob('*.py'))
    return {str(p.relative_to(SCRIPTS)): sha(p) for p in files}


def verify_hashes(hashes):
    for name, digest in hashes.items():
        if sha(name) != digest:
            raise ValueError('Frozen input changed: ' + str(name))


def review_inputs(recipe):
    review = recipe['review']
    if review.get('accepted') is not True or not review.get('evidence'):
        raise ValueError('Input-side identity, structure, units and mesh/pose review required')


def read_config(path):
    path = Path(path).resolve()
    config = read(path)
    if set(config) - {'format_version', 'input_root', 'output_root', 'recipe_pattern', 'runtime', 'episode_ids'}:
        raise ValueError('Unknown batch configuration key')
    if config.get('format_version') != 1:
        raise ValueError('Expected format_version1')
    episodes = config.get('episode_ids', SCORED20)
    if episodes != SCORED20:
        raise ValueError('This command requires exactly the fixed Scored20 list, including ep38')
    config['input_root'] = str(resolve(path.parent, config['input_root']))
    config['output_root'] = str(resolve(path.parent, config['output_root']))
    config.setdefault('recipe_pattern', 'episode_{episode:06d}/recipe.json')
    config['config_path'] = str(path)
    config['config_sha256'] = sha(path)
    runtime = config.get('runtime', {})
    if set(runtime) - {'gpu_python', 'source_root', 'weights_dir', 'runtime_info'}:
        raise ValueError('Unknown FP runtime setting')
    config['runtime'] = {k: str(resolve(path.parent, v)) for k, v in runtime.items()}
    return config


def input_files_from_spec(spec):
    paths, _ = contract.validate_spec(spec)
    return {str(p): sha(p) for p in paths}


def contact_point(bundle, obj, base, shape_output=None):
    """Rebind observed contact to the ACTUAL mesh, never an earlier mesh."""
    shape = obj.get('shape')
    if shape and shape['kind'] == 'contact_shape':
        if shape_output is not None:
            return np.load(Path(shape_output) / 'contact_point.npy', allow_pickle=False)
        # Preflight only: the old mesh's matching pose maps the same observed
        # table-ray intersection. Execution uses the fitted contact point above.
        inputs = shape['inputs']
        annotation = inputs['contact_annotation']
        if isinstance(annotation, str):
            annotation = read(resolve(base, annotation))
        plane = inputs['table_plane']
        plane = np.load(resolve(base, plane), allow_pickle=False) if isinstance(plane, str) else np.asarray(plane, float)
        if plane.shape != (4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3]) < 1e-12:
            raise ValueError('Invalid contact table plane')
        plane = plane / np.linalg.norm(plane[:3])
        k = read(Path(obj['_observations']) / 'K_fp.json')
        K = np.array([[k['fx'], 0, k['cx']], [0, k['fy'], k['cy']], [0, 0, 1.]])
        data = arrays(bundle)
        t = annotation['frame']
        C = data['camera_to_world_transform'][t]
        P = data[fields(obj['slot'])[0]][t]
        ray = np.linalg.solve(K, np.r_[annotation['mask_boundary_pixel'], 1.])
        nc = C[:3, :3].T @ plane[:3]
        denominator = float(nc @ ray)
        if abs(denominator) < 1e-8:
            raise ValueError('Observed contact ray is parallel to table')
        point = ray * (-(plane[3] + plane[:3] @ C[:3, 3]) / denominator)
        if point[2] <= 0:
            raise ValueError('Observed table contact is behind camera')
        return P[:3, :3].T @ (point - P[:3, 3])
    source = obj.get('contact_point')
    if source is None:
        raise ValueError('Contact/temporal stage needs contact_point or contact_shape output')
    mesh = Path(bundle) / mesh_name(obj['slot'])
    if set(source) == {'vertex'}:
        vertices = trimesh.load(mesh, force='mesh', process=False).vertices
        index = source['vertex']
        if type(index) is not int or not 0 <= index < len(vertices):
            raise ValueError('Contact vertex is outside matching mesh')
        return np.asarray(vertices[index])
    if set(source) == {'file', 'mesh_sha256'}:
        if shape is not None:
            raise ValueError('A pre-shape contact point cannot be attached to a new shape; use a reviewed vertex or contact_shape')
        if sha(mesh) != source['mesh_sha256']:
            raise ValueError('Local contact point belongs to another mesh')
        point = np.load(resolve(base, source['file']), allow_pickle=False)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('Invalid contact point')
        return point
    raise ValueError('contact_point must name a vertex or a mesh-hash-bound NPY file')


def refinement_spec(kind, step, bundle, obj, base, stage, shape_output=None):
    inputs = materialize(step['inputs'], kind, base, stage)
    inputs.update(bundle=str(Path(bundle).resolve()), observations=obj['_observations'])
    if kind == 'contact_shape':
        # Preserve explicit validity; otherwise use adapter's finite-depth support.
        inputs.setdefault('validity', str(Path(obj['_observations']) / 'validity'))
    if kind == 'temporal_pose':
        calibration = Path(stage) / 'calibration.json'
        put(calibration, dict(contact_point_local_m=contact_point(bundle, obj, base, shape_output).tolist(),
                              mesh_sha256=sha(Path(bundle) / mesh_name(obj['slot']))))
        inputs['calibration'] = str(calibration.resolve())
    return dict(kind=kind, object_slot=obj['slot'], coordinate_convention=CONVENTION,
                structure_review=dict(accepted=True, evidence=obj['review']['evidence']),
                mesh_pose_pair=dict(mesh_sha256=sha(Path(bundle) / mesh_name(obj['slot'])),
                                    result_sha256=sha(Path(bundle) / 'result.npz')),
                historical_Public_GT_exposed=False, inputs=inputs, parameters=step['parameters'])


def prepare_episode(config, episode, prepared):
    base = Path(config['input_root'])
    rp = resolve(base, config['recipe_pattern'].format(episode=episode))
    recipe = read(rp)
    if set(recipe) - {'episode', 'bundle', 'extra_files', 'objects'} or recipe['episode'] != episode:
        raise ValueError('Invalid episode recipe')
    base = rp.parent
    bundle = resolve(base, recipe['bundle'])
    if prepared.resolve().is_relative_to(bundle):
        raise ValueError('Prepared output must not be inside an input bundle')
    n = validate_scene(bundle)
    prepared.mkdir(parents=True, exist_ok=False)
    slots = object_slots(bundle)
    if sorted(o['slot'] for o in recipe['objects']) != slots:
        raise ValueError('Every object slot needs an explicit refine or preserve recipe')
    hashes = {str(rp): sha(rp)}
    hashes.update({str(p): sha(p) for p in scene_files(bundle, recipe.get('extra_files', []))})
    plan = []
    new_modules = 0
    for obj in sorted(recipe['objects'], key=lambda x: x['slot']):
        allowed = {'slot', 'name', 'preserve', 'review', 'observations', 'shape', 'tracking', 'temporal', 'contact_point'}
        if set(obj) - allowed:
            raise ValueError('Unknown object recipe key')
        if obj.get('preserve'):
            if set(obj) - {'slot', 'name', 'preserve'}:
                raise ValueError('Preserve cannot also declare active stages')
            if not isinstance(obj['preserve'], str):
                raise ValueError('Preserve requires an explicit reason')
            plan.append(dict(slot=obj['slot'], stages=['preserve'], reason=obj['preserve']))
            continue
        review_inputs(obj)
        shape, track, temporal = obj.get('shape'), obj.get('tracking'), obj.get('temporal')
        if not shape and not temporal and not (track and track.get('contact')):
            raise ValueError('No 5454d57 optimizer enabled: tracking-only is not the new pipeline')
        if shape and (shape.get('kind') not in ('part_shape', 'contact_shape') or track is None):
            raise ValueError('Shape refinement requires part_shape/contact_shape followed by fresh FP')
        if track is not None:
            if set(track) - {'anchor_frame', 'contact'} or type(track['anchor_frame']) is not int or not 0 <= track['anchor_frame'] < n:
                raise ValueError('Invalid fixed input-reviewed FP anchor')
            if track.get('contact') and shape and track['anchor_frame'] not in shape['parameters']['support'] + shape['parameters']['heldout']:
                raise ValueError('Provided contact anchor must have a matching new-shape auxiliary pose')
            if track.get('contact') and shape and shape['kind'] == 'part_shape' and track['anchor_frame'] not in shape['parameters']['support']:
                raise ValueError('Part-shape exports matching support poses only')
        directory = prepared / f'slot_{obj["slot"]}'
        directory.mkdir()
        if directory.resolve().is_relative_to(resolve(base, obj['observations']['root'])):
            raise ValueError('Prepared output must not be inside input observations')
        hashes.update(adapt_observations(obj['observations'], base, n, directory / 'observations'))
        obj['_observations'] = str((directory / 'observations').resolve())
        hashes.update({str(p): sha(p) for p in (directory / 'observations').rglob('*') if p.is_file()})
        if obj.get('contact_point', {}).get('file'):
            point_file = resolve(base, obj['contact_point']['file'])
            hashes[str(point_file)] = sha(point_file)
        stages, planes = [], []
        for kind, step in ([(shape['kind'], shape)] if shape else []) + ([('temporal_pose', temporal)] if temporal else []):
            if set(step) != ({'kind', 'inputs', 'parameters'} if kind != 'temporal_pose' else {'inputs', 'parameters'}):
                raise ValueError('Unexpected optimizer configuration keys')
            sd = directory / ('check_' + kind)
            sd.mkdir()
            spec = refinement_spec(kind, step, bundle, obj, base, sd)
            hashes.update(input_files_from_spec(spec))
            if 'table_plane' in spec['inputs']:
                planes.append(np.load(spec['inputs']['table_plane'], allow_pickle=False))
            put(sd / 'spec.json', spec)
            new_modules += 1
        if shape:
            stages.append(shape['kind'])
        if track:
            stages += ['contact_anchor' if track.get('contact') else 'register_anchor', 'bidirectional_FP', 'paired_scene_export']
            if track.get('contact'):
                c = track['contact']
                if set(c) != {'table_plane', 'rest_frames', 'maximum_shift_m', 'evidence'} or not c['evidence'] or not 0 < c['maximum_shift_m'] <= .06:
                    raise ValueError('Invalid observed-contact tracking configuration')
                rest = c['rest_frames']
                if not rest or len(set(rest)) != len(rest) or any(type(f) is not int or not 0 <= f < n for f in rest):
                    raise ValueError('Invalid observed rest frames')
                plane = resolve(base, c['table_plane'])
                v = np.load(plane, allow_pickle=False)
                if v.shape != (4,) or not np.isfinite(v).all() or not np.isclose(np.linalg.norm(v[:3]), 1, atol=1e-6):
                    raise ValueError('Expected normalized world-space table plane')
                hashes[str(plane)] = sha(plane)
                planes.append(v)
                contact_point(bundle, obj, base)
                new_modules += 1
        if planes and any(not (np.allclose(p, planes[0], atol=1e-8, rtol=0) or
                               np.allclose(-p, planes[0], atol=1e-8, rtol=0)) for p in planes[1:]):
            raise ValueError('Shape, contact FP and temporal stages must share the same world table plane')
        if temporal:
            stages.append('temporal_pose')
        plan.append(dict(slot=obj['slot'], stages=stages))
    if not new_modules:
        raise ValueError('Episode has no enabled 5454d57 optimization; refusing silent legacy-only completion')
    prepared_recipe = dict(recipe=recipe, base=str(base), bundle=str(bundle), frame_count=n,
                           episode=episode, plan=plan, file_sha256=hashes)
    put(prepared / 'PREPARED.json', prepared_recipe)
    return prepared_recipe


class NativeBackend:
    """Subprocess boundaries isolate the legacy scripts' common/run imports."""
    def __init__(self, runtime, interactive=False):
        self.runtime = runtime
        self.interactive = interactive

    def command(self, args, log, gpu=False, timeout=None):
        env = os.environ.copy()
        if gpu:
            env['PYTHONPATH'] = self.runtime['source_root'] + os.pathsep + env.get('PYTHONPATH', '')
        with Path(log).open('xb') as stream:
            subprocess.run(list(map(str, args)), stdout=stream, stderr=subprocess.STDOUT,
                           check=True, env=env, timeout=timeout)

    def refine(self, spec, stage):
        stage.mkdir(parents=True, exist_ok=False)
        put(stage / 'spec.json', spec)
        self.command([sys.executable, REFINE / 'run.py', 'freeze', '--spec', stage / 'spec.json', '--output', stage / 'frozen.json'], stage / 'freeze.log')
        self.command([sys.executable, REFINE / 'run.py', 'run', '--manifest', stage / 'frozen.json', '--output', stage / 'candidate'], stage / 'run.log', timeout=spec['parameters']['maximum_seconds'] + 30)
        rec = read(stage / 'candidate/EXECUTION.json')
        if rec.get('exit_code') != 0:
            raise ValueError('Optimizer failed; artifacts retained')
        return stage / 'candidate', rec['result']

    def review(self, stage, artifacts):
        evidence = {str(p): sha(p) for p in artifacts}
        put(stage / 'REVIEW_REQUEST.json', dict(state='REVIEW_REQUIRED', artifacts=evidence))
        if not self.interactive:
            raise ValueError('Output review required; use the same run command with --interactive-reviews for actual execution')
        print('\nReview target identity/visible structure/alignment in:', flush=True)
        for p in artifacts:
            print(' ', p, flush=True)
        answer = input('Type ACCEPT only after checking these artifacts, otherwise REJECT: ').strip()
        put(stage / 'VISUAL_REVIEW.json', dict(accepted=answer == 'ACCEPT', artifacts=evidence, utc=utc()))
        if answer != 'ACCEPT':
            raise ValueError('Visual review rejected; candidate preserved without downstream execution')

    def track(self, spec, payload, stage):
        stage.mkdir(parents=True, exist_ok=False)
        put(stage / 'spec.json', spec)
        rt = self.runtime
        self.command([sys.executable, TRACK / 'prepare.py', 'freeze', '--spec', stage / 'spec.json', '--payload', payload,
                      '--runtime-info', rt['runtime_info'], '--source-root', rt['source_root'], '--output', stage / 'package'], stage / 'freeze.log')
        args = [rt['gpu_python'], TRACK / 'run.py', '--manifest', stage / 'package/manifest.json', '--protocol', stage / 'package/protocol.json',
                '--payload', payload, '--weights-dir', rt['weights_dir'], '--output', stage / 'fp']
        self.command(args + ['--stage', 'anchor'], stage / 'anchor.log', gpu=True, timeout=900)
        self.review(stage, [stage / 'fp/anchor/anchor_overlay.png', stage / 'fp/anchor/RECEIPT.json'])
        self.command([sys.executable, TRACK / 'prepare.py', 'review', '--package', stage / 'package', '--output', stage / 'fp',
                      '--reviewer', 'pipeline-operator', '--evidence', 'anchor/anchor_overlay.png', '--confirm-identity-structure-alignment'], stage / 'review.log')
        self.command(args + ['--stage', 'track'], stage / 'track.log', gpu=True, timeout=900)
        rec = read(stage / 'fp/track/RECEIPT.json')
        if rec.get('state') != 'COMPLETE_FULL_TRACK' or rec.get('exit_code') != 0:
            raise ValueError('Full FP track not complete')
        return stage / 'fp/track'


def shape_pair(shape, output, frame):
    pair = read(output / 'MESH_POSE_PAIR.json')
    filename = 'candidate_parameters.npz' if shape['kind'] == 'part_shape' else 'parameters.npz'
    if pair['mesh_sha256'] != sha(output / 'mesh.obj') or pair['poses_sha256'] != sha(output / filename):
        raise ValueError('Shape auxiliary pose/mesh pair mismatch')
    if pair.get('convention') != CONVENTION or pair.get('scale_baked') is not True:
        raise ValueError('Invalid shape coordinate convention')
    with np.load(output / filename, allow_pickle=False) as z:
        ids = z['support_frames'] if shape['kind'] == 'part_shape' else z['frame_ids']
        poses = z['auxiliary_poses'] if shape['kind'] == 'part_shape' else z['poses']
        if list(ids).count(frame) != 1:
            raise ValueError('No matching new-shape pose for anchor')
        pose = poses[list(ids).index(frame)]
    contract.validate_poses(pose[None], 'shape anchor')
    return pose


def tracking_payload(bundle, obj, base, episode, output, shape_output=None):
    output.mkdir(parents=True, exist_ok=False)
    shape = obj.get('shape')
    mesh = (shape_output / 'mesh.obj' if shape_output is not None else Path(bundle) / mesh_name(obj['slot'])).resolve()
    # Always verify the new shape/auxiliary pairing, including the normal
    # registration path which intentionally does NOT seed from those poses.
    if shape_output is not None:
        shape_pair(shape, shape_output, shape['parameters']['support'][0])
    for p in dependencies(mesh):
        target = output / 'asset' / p.relative_to(mesh.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    obs = Path(obj['_observations'])
    # Read-only copies make the frozen package independent of later stages.
    for name in ('rgb', 'depth', 'masks'):
        shutil.copytree(obs / name, output / name)
    shutil.copy2(obs / 'K_fp.json', output / 'K_fp.json')
    n = validate_scene(bundle)
    config = obj['tracking']
    spec = dict(episode=episode, object=obj.get('name', f'object_{obj["slot"]}'), frame_count=n,
                anchor_frame=config['anchor_frame'], metric_mesh_reviewed=True, anchor_selection_reviewed=True,
                mesh_file=str(Path('asset') / mesh.name), K_file='K_fp.json',
                mesh_dependencies=[str(Path('asset') / p.relative_to(mesh.parent)) for p in dependencies(mesh) if p != mesh],
                frames=[dict(frame_id=i, rgb=f'rgb/{i:06d}.png', depth=f'depth/{i:06d}.png', mask=f'masks/{i:06d}.png') for i in range(n)])
    if config.get('contact'):
        c = config['contact']
        data = arrays(bundle)
        pose = shape_pair(shape, shape_output, config['anchor_frame']) if shape_output else data[fields(obj['slot'])[0]][config['anchor_frame']]
        point = contact_point(bundle, obj, base, shape_output)
        np.save(output / 'initial_pose.npy', pose)
        np.save(output / 'camera_to_world.npy', data['camera_to_world_transform'])
        np.save(output / 'contact_point.npy', point)
        shutil.copy2(resolve(base, c['table_plane']), output / 'plane.npy')
        spec['contact_constraint'] = dict(initial_pose_file='initial_pose.npy', camera_to_world_file='camera_to_world.npy',
            point_file='contact_point.npy', plane_file='plane.npy', rest_frames=c['rest_frames'],
            maximum_shift_m=c['maximum_shift_m'], input_review_evidence=c['evidence'])
    return spec, output / spec['mesh_file']


def run_episode(prepared, output, backend):
    """Actual stage graph: never infer success from the existence of a file."""
    verify_hashes(prepared['file_sha256'])
    recipe, base = prepared['recipe'], Path(prepared['base'])
    current = Path(prepared['bundle'])
    output.mkdir(parents=True, exist_ok=False)
    stages = []
    started = time.monotonic()
    receipt = dict(state='RUNNING', episode=prepared['episode'], started_utc=utc(), stages=stages)
    put(output / 'EXECUTION.json', receipt)
    try:
        for obj in sorted(recipe['objects'], key=lambda x: x['slot']):
            if obj.get('preserve'):
                stages.append(dict(slot=obj['slot'], stage='preserve', reason=obj['preserve']))
                continue
            directory = output / f'slot_{obj["slot"]}'
            directory.mkdir()
            shape, track, temporal = obj.get('shape'), obj.get('tracking'), obj.get('temporal')
            shape_output = None
            if shape:
                spec = refinement_spec(shape['kind'], shape, current, obj, base, directory)
                candidate, result = backend.refine(spec, directory / 'shape')
                stages.append(dict(slot=obj['slot'], stage=shape['kind'], result=result))
                if result.get('input_gate_pass') is not True:
                    raise ValueError('Shape input gate failed; no old-mesh fallback or FP launch')
                shape_output = candidate / 'output'
                backend.review(directory / 'shape', [shape_output / 'comparison.jpg', shape_output / 'RESULTS.json'])
            if track:
                spec, mesh = tracking_payload(current, obj, base, prepared['episode'], directory / 'tracking_input', shape_output)
                tracked = backend.track(spec, directory / 'tracking_input', directory / 'tracking')
                next_bundle = directory / 'tracked_bundle'
                export_track(current, mesh, tracked / 'matched_mesh_poses.npz', tracked / 'MESH_POSE_PAIR.json',
                             next_bundle, obj['slot'], recipe.get('extra_files', []))
                current = next_bundle
                stages.append(dict(slot=obj['slot'], stage='contact_FP' if track.get('contact') else 'FP',
                                   mesh_pose_pair=read(current / 'MESH_POSE_PAIR.json')))
            if temporal:
                spec = refinement_spec('temporal_pose', temporal, current, obj, base, directory, shape_output)
                candidate, result = backend.refine(spec, directory / 'temporal')
                stages.append(dict(slot=obj['slot'], stage='temporal_pose', result=result))
                if result.get('bounds_ok') is not True or not (candidate / 'bundle/result.npz').is_file():
                    raise ValueError('Temporal fit did not export a bounded full candidate')
                # Verify target-only changes independently of the optimizer.
                before, after = arrays(current), arrays(candidate / 'bundle')
                ck, wk = fields(obj['slot'])
                if before.keys() != after.keys() or any(not same(v, after[k]) for k, v in before.items() if k not in (ck, wk)):
                    raise ValueError('Temporal solver changed a protected scene field')
                next_bundle = directory / 'temporal_bundle'
                copy_scene(current, next_bundle, recipe.get('extra_files', []))
                shutil.copy2(candidate / 'bundle/result.npz', next_bundle / 'result.npz')
                validate_scene(next_bundle)
                current = next_bundle
        final = output / 'bundle'
        copy_scene(current, final, recipe.get('extra_files', []))
        # Do not label finite/complete trajectories as visual or policy acceptance.
        temporal_pending = any(s['stage'] == 'temporal_pose' for s in stages)
        put(final / 'run.json', dict(state='COMPLETE_EXPERIMENTAL_PIPELINE', method='5454d57 integrated optimizers',
            episode=prepared['episode'], frame_count=prepared['frame_count'], stages=stages,
            full_scene=True, input_gate_pending=temporal_pending, production_adopted=False,
            GT_used=False, policy_evaluated=False, protected='common camera, hands, table and untargeted objects preserved',
            pairs={str(slot): dict(mesh_sha256=sha(final / mesh_name(slot)), result_sha256=sha(final / 'result.npz'))
                   for slot in object_slots(final)}))
        validate_scene(final)
        verify_hashes(prepared['file_sha256'])
        receipt.update(state='COMPLETE_EXPERIMENTAL_PIPELINE', exit_code=0, bundle_files_sha256=inventory(final))
    except BaseException as error:
        receipt.update(state='FAILED_PRESERVED', exit_code=1, error=repr(error))
        raise
    finally:
        receipt.update(seconds=time.monotonic() - started, ended_utc=utc())
        put(output / 'EXECUTION.json', receipt)
    return receipt


def prepare_batch(config, output):
    output = Path(output).resolve()
    root = Path(config['input_root'])
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Output must be separate from input_root')
    output.mkdir(parents=True, exist_ok=False)
    rows, ready = [], []
    for ep in SCORED20:
        try:
            data = prepare_episode(config, ep, output / 'prepared' / f'episode_{ep:06d}')
            ready.append(data)
            rows.append(dict(episode=ep, state='READY', frames=data['frame_count'], plan=data['plan']))
        except Exception as e:
            rows.append(dict(episode=ep, state='MISSING_OR_INVALID_INPUT', error=repr(e)))
    needs_gpu = any(o.get('tracking') for d in ready for o in d['recipe']['objects'])
    rt = config['runtime']
    runtime_errors = []
    if needs_gpu:
        for key in ('gpu_python', 'source_root', 'weights_dir', 'runtime_info'):
            if key not in rt or not Path(rt[key]).exists():
                runtime_errors.append('Missing FP runtime: ' + key)
        if not runtime_errors:
            try:
                info = read(rt['runtime_info'])
                required = ('backend', 'model_registry', 'model_version', 'tensorrt_version', 'gpu_compute_capability', 'models')
                if any(info.get(k) is None for k in required):
                    runtime_errors.append('runtime_info must be native FoundationPoseTracker.runtime_info')
            except (ValueError, AttributeError) as error:
                runtime_errors.append('Invalid runtime_info: ' + str(error))
            for name in ('v2d_foundation_pose/lib/foundation_pose_tracker.py', 'v2d_foundation_pose/lib/FoundationPose/estimater.py'):
                if not (Path(rt['source_root']) / name).is_file():
                    runtime_errors.append('Missing native module: ' + name)
    state = 'READY_FOR_SCORED20' if len(ready) == 20 and not runtime_errors else 'INPUTS_INCOMPLETE_NO_EXECUTION'
    runtime_hashes = {}
    if needs_gpu and not runtime_errors:
        runtime_files = [Path(rt['runtime_info'])] + [Path(rt['source_root']) / name for name in
            ('v2d_foundation_pose/lib/foundation_pose_tracker.py', 'v2d_foundation_pose/lib/FoundationPose/estimater.py')]
        runtime_hashes = {str(p): sha(p) for p in runtime_files}
    report = dict(state=state, episodes=rows, runtime_errors=runtime_errors, no_GPU_calls=True,
                  no_fit_started=True, code_sha256=code_hashes(), config_sha256=config['config_sha256'],
                  runtime_file_sha256=runtime_hashes, cpu_python=sys.version,
                  cpu_packages={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'trimesh', 'numba')})
    put(output / 'PREFLIGHT.json', report)
    return ready, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init', help='Generate review-required recipes from20 existing native bundles; no inference')
    init.add_argument('--input-root', required=True, type=Path)
    init.add_argument('--config-out', required=True, type=Path)
    init.add_argument('--output-root', required=True, help='Output path, relative to generated config')
    for name in ('validate', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True, type=Path)
        p.add_argument('--output', type=Path, help='Override output_root; must be new')
        if name == 'run':
            p.add_argument('--interactive-reviews', action='store_true', help='Review actual shape/FP overlays within this single command')
    args = parser.parse_args()
    if args.command == 'init':
        paths = scaffold(args.input_root, args.config_out, args.output_root)
        print(f'Created {len(paths)} review-required recipes; no optimizer or GPU started')
        return 0
    config = read_config(args.config)
    output = (args.output or Path(config['output_root'])).resolve()
    ready, report = prepare_batch(config, output)
    print('Preflight:', report['state'], output / 'PREFLIGHT.json', flush=True)
    if report['state'] != 'READY_FOR_SCORED20':
        return 2
    if args.command == 'validate':
        return 0
    if any(o.get('tracking') or o.get('shape') for d in ready for o in d['recipe']['objects']) and not args.interactive_reviews:
        raise ValueError('Use --interactive-reviews; the unified entry handles actual output review without manually chaining scripts')
    backend = NativeBackend(config['runtime'], args.interactive_reviews)
    outcomes = []
    for data in ready:
        ep = data['episode']
        print('Running episode', ep, data['plan'], flush=True)
        try:
            if code_hashes() != report['code_sha256'] or sha(config['config_path']) != config['config_sha256']:
                raise ValueError('Code or configuration changed during the frozen batch')
            verify_hashes(report['runtime_file_sha256'])
            rec = run_episode(data, output / f'episode_{ep:06d}', backend)
            outcomes.append(dict(episode=ep, state=rec['state'], bundle=f'episode_{ep:06d}/bundle'))
        except Exception as error:
            outcomes.append(dict(episode=ep, state='FAILED_PRESERVED', error=repr(error)))
        # Continue independent episodes, never retry or substitute a legacy bundle.
        put(output / 'BATCH.json', dict(state='RUNNING', episodes=outcomes, expected=SCORED20))
    passed = all(x['state'] == 'COMPLETE_EXPERIMENTAL_PIPELINE' for x in outcomes)
    put(output / 'BATCH.json', dict(state='COMPLETE_SCORED20_EXPERIMENTAL' if passed else 'INCOMPLETE_SCORED20',
        episodes=outcomes, expected=SCORED20, production_adopted=False, GT_used=False,
        code_sha256=code_hashes(), config_sha256=config['config_sha256']))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
