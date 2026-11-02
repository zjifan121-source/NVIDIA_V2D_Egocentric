"""CPU integration tests. Simulated FP is explicitly confined to tests."""
from pathlib import Path
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline as pipe
from inputs import (adapt_observations, arrays, copy_scene, export_track, fields,
                    install_mesh, put, read, sha, validate_scene)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'input/episode_000001'
        b = self.source / 'bundle'
        b.mkdir(parents=True)
        self.b = b
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=.05)
        mesh.export(b / 'mesh.obj')
        mesh.export(b / 'mesh_1.obj')
        self.n = 6
        poses = np.tile(np.eye(4), (self.n, 1, 1)); poses[:, 2, 3] = .5
        self.poses = poses
        partner = poses.copy(); partner[:, 0, 3] = .2
        camera = np.tile(np.eye(4), (self.n, 1, 1))
        self.original = dict(object_to_camera_transform=poses, object_to_world_transform=poses.copy(),
            object_1_to_camera_transform=partner, object_1_to_world_transform=partner.copy(),
            camera_to_world_transform=camera, object_scale=1., object_1_scale=1.,
            camera_is_valid=np.ones(self.n, bool), object_is_valid=np.ones(self.n, bool),
            object_1_is_valid=np.ones(self.n, bool), hand_left_joints=np.arange(self.n*63).reshape(self.n, 21, 3),
            hand_right_is_valid=np.array([1, 1, 0, 1, 0, 1], bool), timestamps=np.arange(self.n)/20,
            table_pose=np.eye(4))
        np.savez_compressed(b / 'result.npz', **self.original)
        np.save(b / 'table_plane.npy', [0., 0., 1., -.5])
        (b / 'custom_table.json').write_text('{"existing":true}')
        obs = self.source / 'observations'; obs.mkdir(); self.obs = obs
        put(obs / 'K_fp.json', dict(fx=100, fy=100, cx=16, cy=16, width=32, height=32))
        for sub in ['rgb', 'depth', 'masks']:
            (obs / sub).mkdir()
        for t in range(self.n):
            cv2.imwrite(str(obs / 'rgb' / f'{t:06d}.png'), np.zeros((32, 32, 3), np.uint8))
            cv2.imwrite(str(obs / 'depth' / f'{t:06d}.png'), np.full((32, 32), 43690, np.uint16))
            mask = np.zeros((32, 32), np.uint8); cv2.circle(mask, (16, 16), 10, 1, -1)
            cv2.imwrite(str(obs / 'masks' / f'{t:06d}.png'), mask)
        put(self.source / 'marks.json', dict(parts=[0, 1], center=2))
        put(self.source / 'contact.json', dict(frame=0, mask_boundary_pixel=[16, 16], source='observed fixture'))
        put(self.source / 'measurement.json', dict(state='OBSERVABILITY_PASS'))
        points = np.tile(np.array([[0., 0., .5], [.01, 0., .5]]), (self.n, 1, 1))
        np.savez(self.source / 'landmarks.npz', frame_ids=np.arange(self.n), points_camera=points,
                 qualified=np.ones((self.n, 2), bool), sigma_z=np.full((self.n, 2), .005))
        self.temporal = dict(inputs=dict(landmarks='landmarks.npz', measurement_receipt='measurement.json', table_plane='bundle/table_plane.npy'),
            parameters=dict(calibration_frames=[0, 2], fit_frames=[0, 2, 4], validation_frames=[1, 3, 5],
                            contact_frames=[0], landmark_groups=[0, 1], maximum_seconds=60))
        self.obj = dict(slot=0, name='target', review=dict(accepted=True, evidence='Input-side fixture'),
            observations=dict(root='observations', depth_encoding='native_inverse_u16'),
            contact_point=dict(vertex=2), temporal=self.temporal)
        self.recipe = dict(episode=1, bundle='bundle', extra_files=['custom_table.json'],
                           objects=[self.obj, dict(slot=1, preserve='Partner fixed for paired comparison')])
        self.config = dict(input_root=str(self.root / 'input'), recipe_pattern='episode_{episode:06d}/recipe.json')

    def prepare(self, recipe=None):
        put(self.source / 'recipe.json', recipe or self.recipe)
        return pipe.prepare_episode(self.config, 1, self.root / 'prepared')


class AdapterTests(Fixture):
    def test_native_adapter_preserves_depth_values_and_tracks_hashes(self):
        p = self.root / 'adapted'
        hashes = adapt_observations(self.obj['observations'], self.source, self.n, p)
        self.assertEqual(len(hashes), 19)
        np.testing.assert_array_equal(cv2.imread(str(p / 'depth/000000.png'), -1),
                                      cv2.imread(str(self.obs / 'depth/000000.png'), -1))
        self.assertIn('not stereo confidence', read(p / 'ADAPTER.json')['validity'])

    def test_explicit_mm_and_half_pixel_grid_adapter(self):
        for t in range(self.n):
            cv2.imwrite(str(self.obs / 'depth' / f'{t:06d}.png'), np.full((16, 16), 500, np.uint16))
        cfg = dict(root='observations', depth_encoding='millimetres_u16', resize_to_depth=True)
        output = self.root / 'adapted'
        adapt_observations(cfg, self.source, self.n, output)
        k = read(output / 'K_fp.json')
        self.assertEqual((k['fx'], k['cx'], k['width']), (50, 7.75, 16))
        raw = cv2.imread(str(output / 'depth/000000.png'), -1)
        np.testing.assert_allclose(65535./raw - 1., .5)

    def test_grid_mismatch_and_aliased_timeline_fail(self):
        cv2.imwrite(str(self.obs / 'depth/000000.png'), np.zeros((8, 8), np.uint16))
        with self.assertRaisesRegex(ValueError, 'Grid mismatch'):
            adapt_observations(self.obj['observations'], self.source, self.n, self.root / 'bad')
        cfg = dict(self.obj['observations'], depth='depth/000001.png')
        with self.assertRaisesRegex(ValueError, 'Repeated input frame'):
            adapt_observations(cfg, self.source, self.n, self.root / 'bad2')


class ContractTests(Fixture):
    def test_init_creates_review_required_actual_object_recipes(self):
        for episode in pipe.SCORED20:
            target = self.root / 'input' / f'episode_{episode:06d}'
            if episode != 1:
                shutil.copytree(self.source, target)
        config_file = self.root / 'config/scored20.json'
        paths = pipe.scaffold(self.root / 'input', config_file, 'runs/new')
        self.assertEqual(len(paths), 20)
        recipe = read(self.root / 'input/episode_000038/recipe.json')
        self.assertEqual([o['slot'] for o in recipe['objects']], [0, 1])
        self.assertFalse(recipe['objects'][0]['review']['accepted'])
        self.assertEqual(pipe.read_config(config_file)['input_root'], str((self.root / 'input').resolve()))
        with self.assertRaises(FileExistsError):
            pipe.scaffold(self.root / 'input', config_file, 'runs/new')

    def test_inline_annotations_materialize_without_inventing_evidence(self):
        self.temporal['inputs']['table_plane'] = [0., 0., 2., -1.]
        self.temporal['inputs']['measurement_receipt'] = dict(state='OBSERVABILITY_PASS', evidence='Measured stereo correspondences')
        with np.load(self.source / 'landmarks.npz') as z:
            self.temporal['inputs']['landmarks'] = {k: z[k].tolist() for k in z.files}
        prepared = self.prepare()
        spec = read(self.root / 'prepared/slot_0/check_temporal_pose/spec.json')
        np.testing.assert_allclose(np.load(spec['inputs']['table_plane']), [0, 0, 1, -.5])
        self.assertEqual(prepared['plan'][0]['stages'], ['temporal_pose'])

    def test_all_twenty_preflight_without_fitting_and_missing_episode_is_explicit(self):
        for episode in pipe.SCORED20:
            target = self.root / 'input' / f'episode_{episode:06d}'
            if episode != 1:
                shutil.copytree(self.source, target)
            recipe = copy.deepcopy(self.recipe); recipe['episode'] = episode
            put(target / 'recipe.json', recipe)
        cfg = dict(self.config, runtime={}, config_sha256='fixture')
        ready, report = pipe.prepare_batch(cfg, self.root / 'preflight')
        self.assertEqual(len(ready), 20)
        self.assertEqual(report['state'], 'READY_FOR_SCORED20')
        self.assertTrue(report['no_fit_started'])
        (self.root / 'input/episode_000038/recipe.json').unlink()
        ready, report = pipe.prepare_batch(cfg, self.root / 'missing')
        self.assertEqual(report['state'], 'INPUTS_INCOMPLETE_NO_EXECUTION')
        self.assertEqual(report['episodes'][-1]['state'], 'MISSING_OR_INVALID_INPUT')

    def test_scored20_has_twenty_and_includes38(self):
        self.assertEqual(len(pipe.SCORED20), 20)
        self.assertEqual(pipe.SCORED20[-1], 38)

    def test_no_optimizer_is_not_a_successful_new_pipeline(self):
        self.obj.pop('temporal'); self.obj['tracking'] = dict(anchor_frame=0)
        with self.assertRaisesRegex(ValueError, 'No 5454d57'):
            self.prepare()

    def test_all_preserved_and_missing_partner_rejected(self):
        self.recipe['objects'] = [dict(slot=0, preserve='historical'), dict(slot=1, preserve='historical')]
        with self.assertRaisesRegex(ValueError, 'no enabled'):
            self.prepare()
        self.recipe['objects'].pop()
        put(self.source / 'recipe.json', self.recipe)
        with self.assertRaisesRegex(ValueError, 'Every object'):
            pipe.prepare_episode(self.config, 1, self.root / 'other')

    def test_shape_cannot_skip_new_matching_tracking(self):
        self.obj['shape'] = dict(kind='part_shape', inputs=dict(semantic_vertices='marks.json'),
                                parameters=dict(support=[0, 2, 4], heldout=[1, 3, 5], maximum_iterations=15, maximum_seconds=60))
        with self.assertRaisesRegex(ValueError, 'fresh FP'):
            self.prepare()

    def test_conflicting_world_planes_fail_preflight(self):
        self.obj['tracking'] = dict(anchor_frame=0, contact=dict(table_plane='bundle/table_plane.npy',
            rest_frames=[0], maximum_shift_m=.06, evidence='Input fixture'))
        self.temporal['inputs']['table_plane'] = [0., 0., 1., -.4]
        with self.assertRaisesRegex(ValueError, 'same world table plane'):
            self.prepare()

    def test_mismatched_mesh_pair_and_partial_tracks_fail_export(self):
        z = self.root / 'poses.npz'
        np.savez(z, frame_ids=np.arange(5), public_poses=self.poses[:5])
        pair = dict(mesh_sha256=sha(self.b / 'mesh.obj'), poses_sha256=sha(z), frame_count=6,
                    mesh_scale_already_baked=True, additional_scale=1.)
        put(self.root / 'pair.json', pair)
        with self.assertRaisesRegex(ValueError, 'Full chronological'):
            export_track(self.b, self.b / 'mesh.obj', z, self.root / 'pair.json', self.root / 'delivery', 0)
        pair['mesh_sha256'] = 'different'; put(self.root / 'pair.json', pair)
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            export_track(self.b, self.b / 'mesh.obj', z, self.root / 'pair.json', self.root / 'delivery', 0)

    def test_texture_dependency_replacement_is_namespaced(self):
        mesh = self.root / 'newmesh'; mesh.mkdir()
        (mesh / 'texture.png').write_bytes(b'new')
        (mesh / 'same.mtl').write_text('newmtl color\nmap_Kd texture.png\n')
        (mesh / 'mesh.obj').write_text('mtllib same.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
        (self.b / 'texture.png').write_bytes(b'partner')
        install_mesh(mesh / 'mesh.obj', self.b, 0)
        self.assertEqual((self.b / 'texture.png').read_bytes(), b'partner')
        self.assertIn('optimized_assets/slot_0/same.mtl', (self.b / 'mesh.obj').read_text())


class FakeNativeBackend:
    """No native tracker/CUDA. Test stage wiring using explicit fixture outputs."""
    def __init__(self, reject_shape=False):
        self.events = []
        self.reject_shape = reject_shape

    def refine(self, spec, stage):
        stage.mkdir()
        self.events.append(spec['kind'])
        candidate = stage / 'candidate'; candidate.mkdir()
        source = Path(spec['inputs']['bundle']); slot = spec['object_slot']
        if spec['kind'] != 'temporal_pose':
            out = candidate / 'output'; out.mkdir()
            m = trimesh.load(source / pipe.mesh_name(slot), process=False)
            m.vertices = m.vertices * 1.01
            m.export(out / 'mesh.obj')
            support = spec['parameters']['support']; held = spec['parameters']['heldout']
            p = arrays(source)[fields(slot)[0]]
            if spec['kind'] == 'part_shape':
                filename = 'candidate_parameters.npz'
                np.savez(out / filename, support_frames=support, auxiliary_poses=p[support])
            else:
                filename = 'parameters.npz'
                np.savez(out / filename, frame_ids=support+held, poses=p[support+held])
                np.save(out / 'contact_point.npy', np.array([.01, .02, -.03]))
            put(out / 'MESH_POSE_PAIR.json', dict(mesh_sha256=sha(out / 'mesh.obj'), poses_sha256=sha(out / filename),
                                                convention=pipe.CONVENTION, scale_baked=True))
            (out / 'comparison.jpg').write_bytes(b'test')
            put(out / 'RESULTS.json', {})
            return candidate, dict(state='TEST_ONLY', input_gate_pass=not self.reject_shape)
        out = candidate / 'bundle'
        copy_scene(source, out)
        data = arrays(source); ck, wk = fields(slot)
        data[ck][:, 0, 3] += .001
        data[wk] = data['camera_to_world_transform'] @ data[ck]
        np.savez_compressed(out / 'result.npz', **data)
        self.temporal_input = source
        self.temporal_calibration = read(spec['inputs']['calibration'])
        return candidate, dict(state='TEST_ONLY', bounds_ok=True)

    def review(self, stage, artifacts):
        self.events.append('visual_review')

    def track(self, spec, payload, stage):
        self.events.append('contact_FP' if 'contact_constraint' in spec else 'FP')
        stage.mkdir()
        poses = np.tile(np.eye(4), (spec['frame_count'], 1, 1)); poses[:, 2, 3] = .51
        self.tracking_mesh = Path(payload) / spec['mesh_file']
        self.tracking_spec = spec
        out = stage / 'track'; out.mkdir()
        np.savez(out / 'matched_mesh_poses.npz', frame_ids=np.arange(len(poses)), public_poses=poses)
        put(out / 'MESH_POSE_PAIR.json', dict(mesh_sha256=sha(self.tracking_mesh), poses_sha256=sha(out / 'matched_mesh_poses.npz'),
                                           frame_count=len(poses), mesh_scale_already_baked=True, additional_scale=1.))
        return out


class WiringTests(Fixture):
    def shape_recipe(self, kind):
        self.obj['tracking'] = dict(anchor_frame=0)
        if kind == 'part_shape':
            self.obj['shape'] = dict(kind=kind, inputs=dict(semantic_vertices='marks.json'),
                parameters=dict(support=[0, 2, 4], heldout=[1, 3, 5], maximum_iterations=15, maximum_seconds=60))
        else:
            self.obj['shape'] = dict(kind=kind, inputs=dict(table_plane='bundle/table_plane.npy', contact_annotation='contact.json'),
                parameters=dict(support=[0, 2, 4], heldout=[1, 3, 5], graph_nodes=16, maximum_seconds=60))
            self.obj['tracking']['contact'] = dict(table_plane='bundle/table_plane.npy', rest_frames=[0], maximum_shift_m=.06, evidence='Observed fixture')

    def test_shape_fp_temporal_are_really_called_with_new_artifacts(self):
        self.shape_recipe('part_shape')
        prepared = self.prepare(); backend = FakeNativeBackend()
        rec = pipe.run_episode(prepared, self.root / 'run', backend)
        self.assertEqual(backend.events, ['part_shape', 'visual_review', 'FP', 'temporal_pose'])
        self.assertEqual(rec['state'], 'COMPLETE_EXPERIMENTAL_PIPELINE')
        final = self.root / 'run/bundle'; data = arrays(final)
        np.testing.assert_allclose(data['object_to_camera_transform'][:, 2, 3], .51)
        np.testing.assert_allclose(data['object_to_camera_transform'][:, 0, 3], .001)
        self.assertNotEqual(sha(backend.tracking_mesh), sha(self.b / 'mesh.obj'))
        for k, v in self.original.items():
            if k not in fields(0):
                np.testing.assert_array_equal(v, data[k])
        self.assertEqual(sha(final / 'table_plane.npy'), sha(self.b / 'table_plane.npy'))
        self.assertEqual(sha(final / 'custom_table.json'), sha(self.b / 'custom_table.json'))
        self.assertTrue(read(final / 'run.json')['input_gate_pending'])
        self.assertEqual(backend.temporal_input.name, 'tracked_bundle')

    def test_contact_shape_point_is_rebound_into_fp_and_temporal(self):
        self.shape_recipe('contact_shape')
        self.obj['shape']['inputs']['table_plane'] = [0., 0., 2., -1.]
        self.obj['shape']['inputs']['contact_annotation'] = read(self.source / 'contact.json')
        prepared = self.prepare(); backend = FakeNativeBackend()
        pipe.run_episode(prepared, self.root / 'run', backend)
        self.assertEqual(backend.events, ['contact_shape', 'visual_review', 'contact_FP', 'temporal_pose'])
        np.testing.assert_allclose(backend.temporal_calibration['contact_point_local_m'], [.01, .02, -.03])
        payload = self.root / 'run/slot_0/tracking_input'
        np.testing.assert_allclose(np.load(payload / 'contact_point.npy'), [.01, .02, -.03])
        np.testing.assert_array_equal(np.load(payload / 'camera_to_world.npy'), self.original['camera_to_world_transform'])

    def test_generated_fp_payload_passes_real_freezer_and_cpu_loader(self):
        self.obj['tracking'] = dict(anchor_frame=0, contact=dict(table_plane='bundle/table_plane.npy',
            rest_frames=[0], maximum_shift_m=.06, evidence='Observed table contact fixture'))
        prepared = self.prepare()
        obj = prepared['recipe']['objects'][0]
        payload = self.root / 'payload'
        spec, mesh = pipe.tracking_payload(self.b, obj, self.source, 1, payload)
        put(self.root / 'fp_spec.json', spec)
        put(self.root / 'runtime.json', dict(backend='nvidia_tensorrt', model_registry='TEST_ONLY',
            model_version='fixture', tensorrt_version='10', gpu_compute_capability='8.6', models={}))
        source = pipe.SCRIPTS.parent / 'modules'
        commands = [
            [sys.executable, pipe.TRACK / 'prepare.py', 'freeze', '--spec', self.root / 'fp_spec.json',
             '--payload', payload, '--runtime-info', self.root / 'runtime.json', '--source-root', source,
             '--output', self.root / 'package'],
            [sys.executable, pipe.TRACK / 'run.py', '--manifest', self.root / 'package/manifest.json',
             '--protocol', self.root / 'package/protocol.json', '--payload', payload, '--validate-only']]
        for args in commands:
            result = subprocess.run(list(map(str, args)), text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(sha(mesh), sha(self.b / 'mesh.obj'))

    def test_adapted_input_mutation_is_rejected(self):
        prepared = self.prepare()
        (self.root / 'prepared/slot_0/observations/rgb/000005.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
            pipe.run_episode(prepared, self.root / 'run', FakeNativeBackend())

    def test_failed_shape_gate_never_launches_fp_or_exports_final_bundle(self):
        self.shape_recipe('part_shape')
        prepared = self.prepare(); backend = FakeNativeBackend(reject_shape=True)
        with self.assertRaisesRegex(ValueError, 'Shape input gate failed'):
            pipe.run_episode(prepared, self.root / 'run', backend)
        self.assertEqual(backend.events, ['part_shape'])
        self.assertFalse((self.root / 'run/bundle').exists())
        self.assertEqual(read(self.root / 'run/EXECUTION.json')['state'], 'FAILED_PRESERVED')

    def test_changed_original_and_existing_output_are_not_reused(self):
        prepared = self.prepare()
        (self.obs / 'rgb/000000.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
            pipe.run_episode(prepared, self.root / 'run', FakeNativeBackend())

    def test_real_cpu_temporal_solver_through_native_subprocess_entry(self):
        # Exercise the actual optimizer and scene exporter, not the FP double.
        self.obj['contact_point'] = dict(vertex=0)
        point = trimesh.load(self.b / 'mesh.obj', process=False).vertices[0]
        np.save(self.b / 'table_plane.npy', [0., 0., 1., -(.5 + point[2])])
        prepared = self.prepare()
        try:
            rec = pipe.run_episode(prepared, self.root / 'run', pipe.NativeBackend({}))
        except Exception:
            log = self.root / 'run/slot_0/temporal/run.log'
            if log.exists():
                print(log.read_text())
            raise
        self.assertEqual(rec['state'], 'COMPLETE_EXPERIMENTAL_PIPELINE')
        result = read(self.root / 'run/slot_0/temporal/candidate/EXECUTION.json')
        self.assertEqual(result['kind'], 'temporal_pose')
        self.assertEqual(result['GPU_calls'], 0)
        self.assertEqual(validate_scene(self.root / 'run/bundle'), 6)
        for k, v in self.original.items():
            if k not in fields(0):
                np.testing.assert_array_equal(arrays(self.root / 'run/bundle')[k], v)


if __name__ == '__main__':
    unittest.main()
