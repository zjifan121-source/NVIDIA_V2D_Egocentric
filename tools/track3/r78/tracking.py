"""Normal FoundationPose with a provided metric mesh and no scale search.

Seed: one registration at each of the five fixed fit/heldout frames.
Track: one registration, then consecutive track_one calls. No warm-start poses
from another mesh, cache resume, joint refinement, or diagnostic rendering.
"""
from pathlib import Path
import random
import inspect
import shutil
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .io import Inputs, artifact, copy_appearance, fresh, seal, sha, write
from .selection import selected_asset
from .shared_scale import validate_pose


FP_SETTINGS = dict(backend='nvidia_tensorrt', seed=0, register_iteration=10,
                   track_iteration=5, mask_depth=True, n_particles=1,
                   reregister_iou_thresh=None, skip_object_scale_estimation=True)


class NativeFP:
    def __init__(self, mesh, weights):
        import torch
        from v2d.mesh.lib.mesh import Mesh
        from v2d.foundation_pose.lib import trt_engine
        from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker
        def no_build(*args, **kwargs):
            raise RuntimeError('Prebuilt TensorRT engines required; no implicit engine build')
        trt_engine.build_engine = no_build
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.set_grad_enabled(False)
        self.tracker = FoundationPoseTracker(Mesh.load(str(mesh)), str(weights), backend='nvidia_tensorrt')
        self.runtime = dict(tracker_source_sha256=sha(inspect.getfile(FoundationPoseTracker)),
                            runtime_info=self.tracker.runtime_info)

    def step(self, f, register):
        from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask, Image
        K, (h, w) = f['K'], f['shape']
        intrinsics = CameraIntrinsics(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], width=w, height=h)
        depth = f['fp_depth'].copy()
        depth[~f['valid']] = 0
        args = (Image(data=f['rgb']), DepthImage(depth=depth))
        if register:
            pose = self.tracker.register(*args, Mask(mask=f['mask']), intrinsics, iteration=10)
        else:
            pose = self.tracker.track_one(*args, intrinsics, iteration=5)
        return pose.to_matrix()


def run(manifest, weights, output, selection=None, backend_factory=None):
    inputs = Inputs(manifest)
    selection_hash = sha(selection) if selection else None
    rows = inputs.rows()
    if selection is None:
        mesh = inputs.mesh()
        if (len(rows) != 5 or sum(r.get('role') == 'support' for r in rows) != 3 or
                sum(r.get('role') == 'heldout' for r in rows) != 2):
            raise ValueError('Seed requires three support and two heldout frames')
        mode, fit = 'seed', None
    else:
        path, fit = selected_asset(selection)
        mesh = path.parent / fit['mesh']
        if inputs.context != fit['context']:
            raise ValueError('Tracking source differs from selected fit')
        ids = [r['id'] for r in rows]
        if ids != list(range(ids[0], ids[-1] + 1)):
            raise ValueError('Tracking requires ordered consecutive frame IDs')
        mode = 'track'
    # Preflight all inputs before importing GPU dependencies or creating a tracker.
    fingerprints = {row['id']: inputs.frame(row)['input_sha256'] for row in rows}
    if fit:
        for row in fit['frames']:
            if row['id'] not in fingerprints or fingerprints[row['id']] != row['input_sha256']:
                raise ValueError('Tracking must include the same frozen support/heldout observations')
    out = fresh(output)
    start = time.monotonic()
    try:
        shutil.copy2(mesh, out / 'mesh.obj')
        copy_appearance(mesh, out / 'mesh.obj')
        (out / 'poses').mkdir()
        tracker = (backend_factory or NativeFP)(out / 'mesh.obj', weights)
        frames = []
        for index, row in enumerate(rows):
            f = inputs.frame(row)
            register = mode == 'seed' or index == 0
            T = validate_pose(tracker.step(f, register=register))
            rel = f"poses/{f['id']:06d}.npy"
            np.save(out / rel, T)
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            write(out / 'poses' / f"{f['id']:06d}.json", dict(
                rotation=q[[3, 0, 1, 2]].tolist(), translation=T[:3, 3].tolist(), scale=[1., 1., 1.]))
            frames.append(dict(id=f['id'], pose=rel, pose_sha256=sha(out / rel),
                               input_sha256=f['input_sha256'], operation='register' if register else 'track'))
            write(out / 'progress.json', dict(state='RUNNING', completed=len(frames), total=len(rows)))
        inputs.unchanged()
        if sha(mesh) != sha(out / 'mesh.obj'):
            raise ValueError('Provided mesh changed during FP')
        if selection and sha(selection) != selection_hash:
            raise ValueError('Selection changed during FP')
        write(out / 'progress.json', dict(state='COMPLETE', completed=len(frames), total=len(rows)))
        metadata = seal(out, dict(state='SEED_COMPLETE' if mode == 'seed' else 'TRACK_COMPLETE',
            context=inputs.context, frames=frames, mesh='mesh.obj', mesh_sha256=sha(out / 'mesh.obj'),
            mesh_scale_baked=True, coordinate_units='metres', pose_convention='original_mesh_to_camera',
            FP_entry=dict(object_mesh_path='mesh.obj', skip_object_scale_estimation=True),
            numeric_settings=FP_SETTINGS, input_manifest_sha256=sha(manifest),
            selection_sha256=selection_hash, runtime=getattr(tracker, 'runtime', {'backend': 'test_double'}),
            wall_s=time.monotonic()-start))
        if mode == 'seed':
            # Absolute paths are runtime references, not checked-in site configuration.
            fit_manifest = dict(inputs.data)
            fit_manifest['sha256'] = {}
            for key in ('mesh',):
                fit_manifest[key] = str(out / 'mesh.obj')
                fit_manifest['sha256'][fit_manifest[key]] = sha(out / 'mesh.obj')
            fit_manifest['frames'] = []
            for row, result in zip(rows, frames):
                new = dict(row)
                for key in ('rgb', 'depth', 'validity', 'mask', 'K'):
                    new[key] = str(inputs.file(row[key]))
                    fit_manifest['sha256'][new[key]] = sha(new[key])
                new['pose'] = str(out / result['pose'])
                fit_manifest['sha256'][new['pose']] = sha(new['pose'])
                fit_manifest['frames'].append(new)
            fit_manifest['pose_artifact'] = str(out / 'artifact.json')
            fit_manifest['sha256'][fit_manifest['pose_artifact']] = sha(out / 'artifact.json')
            write(out / 'fit_manifest.json', fit_manifest)
        return metadata
    except Exception as exc:
        write(out / 'failure.json', dict(state='FAILED', error=str(exc), wall_s=time.monotonic()-start))
        raise
