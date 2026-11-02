"""The ep27 bounded shared-scale solve; heldout frames never update scale."""
from pathlib import Path
from dataclasses import asdict
import time
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from .shared_scale import Config, centered_pose, public_pose, sample_surface, unproject, fit, model_surface
from .geometry_cpu import raster
from .io import Inputs, sha, write, pose, fresh, artifact, seal, copy_appearance, mesh_dependencies


def fit_frames(inputs, mesh_path):
    d = inputs.data
    rows = inputs.rows()
    if (len(rows) != 5 or sum(r.get('role') == 'support' for r in rows) != 3 or
            sum(r.get('role') == 'heldout' for r in rows) != 2):
        raise ValueError('Freeze exactly three support and two heldout frames')
    provenance_path = inputs.file(d['pose_artifact'])
    provenance = artifact(provenance_path, {'SEED_COMPLETE'})
    if provenance['mesh_sha256'] != sha(mesh_path) or provenance['context'] != inputs.context:
        raise ValueError('Seed poses and mesh/input context do not match')
    seeds = {row['id']: row for row in provenance['frames']}
    frames = []
    for row in rows:
        f = inputs.frame(row)
        pose_path = inputs.file(row['pose'])
        seed = seeds.get(row['id'], {})
        if (seed.get('pose_sha256') != sha(pose_path) or
                seed.get('input_sha256') != f['input_sha256']):
            raise ValueError('Seed pose was computed with different frame inputs')
        h, w = f['shape']
        y, x = np.indices((h, w))
        check = ((x // 8) * 17 + (y // 8) * 31) % 5 == 0
        f.update(role=row['role'], pose=pose(pose_path),
                 points=unproject(f['depth'], f['K'], f['valid'] & ~check),
                 validation_points=unproject(f['depth'], f['K'], f['valid'] & check))
        if len(f['validation_points']) < Config().minimum_correspondences:
            raise ValueError('Insufficient untouched validation pixel support')
        frames.append(f)
    return frames


def export_obj(source, dest, center, scale):
    lines = []
    for line in source.read_text().splitlines(keepends=True):
        if line.startswith('v '):
            parts = line.split()
            v = center + scale * (np.array(parts[1:4], float) - center)
            line = 'v ' + ' '.join(format(x, '.17g') for x in v)
            line += (' ' + ' '.join(parts[4:]) if len(parts) > 4 else '') + '\n'
        lines.append(line)
    dest.write_text(''.join(lines))
    copy_appearance(source, dest)


def run(manifest, output):
    out = fresh(output)
    start = time.monotonic()
    try:
        inputs = Inputs(manifest)
        mesh_path = inputs.mesh()
        appearance_hashes = {p: sha(p) for p in mesh_dependencies(mesh_path)}
        frames = fit_frames(inputs, mesh_path)
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
            raise ValueError('Empty or nonfinite mesh')
        center = (vertices.min(0) + vertices.max(0)) / 2
        points = sample_surface(vertices - center, faces)
        for f in frames:
            f['pose'] = centered_pose(f['pose'], center)
            f['mesh'] = (vertices - center, faces)
        support = [f for f in frames if f['role'] == 'support']
        heldout = [f for f in frames if f['role'] == 'heldout']
        result = fit(points, support)
        write(out / 'support_trace.json', {k: v for k, v in result.items() if k != 'poses'})
        if result['state'] == 'BOUND_REACHED':
            raise ValueError('SCALE_INCOMPLETE: support pose/scale bound reached')
        scale = result['scale']
        validation = fit(points, heldout, fixed_scale=scale)
        write(out / 'heldout_pose_trace.json', {k: v for k, v in validation.items() if k != 'poses'})
        if validation['state'] == 'BOUND_REACHED':
            raise ValueError('SCALE_INCOMPLETE: heldout pose bound reached')
        values, pairs = [], []
        for group, poses in [(support, result['poses']), (heldout, validation['poses'])]:
            for f, T in zip(group, poses):
                filename = f"pose_{f['id']:06d}.npy"
                np.save(out / filename, public_pose(T, center))
                pairs.append(dict(id=f['id'], pose=filename, role=f['role'], input_sha256=f['input_sha256']))
                values.append(dict(id=f['id'], role=f['role'],
                    initial=dict(metrics(points, f, f['pose'], 1.), **silhouette(vertices-center, faces, f, f['pose'], 1.)),
                    candidate=dict(metrics(points, f, T, scale), **silhouette(vertices-center, faces, f, T, scale))))
        export_obj(mesh_path, out / 'mesh.obj', center, scale)
        write(out / 'metrics.json', values)
        inputs.unchanged()
        if any(sha(p) != h for p, h in appearance_hashes.items()):
            raise ValueError('Mesh appearance changed during fit')
        return seal(out, dict(state='FIT_BOUNDED', context=inputs.context,
            input_manifest_sha256=sha(manifest), input_mesh_sha256=sha(mesh_path),
            structure_review=inputs.data['structure_review'], mesh='mesh.obj',
            mesh_sha256=sha(out / 'mesh.obj'), uniform_scale_multiplier=scale,
            mesh_scale_baked=True, world_scale_unchanged=True,
            pose_convention='original_mesh_to_camera', coordinate_units='metres',
            mesh_center=center.tolist(), scale_rule='v_metric = center + s * (v_input - center)',
            pose_rule='T_public = T_center @ Translation(-center)',
            frames=pairs, config=asdict(Config()), support_state=result['state'],
            heldout_state=validation['state'], GT_used=False, wall_s=time.monotonic()-start,
            FP_entry=dict(object_mesh_path='mesh.obj', skip_object_scale_estimation=True)))
    except Exception as exc:
        write(out / 'failure.json', dict(state='FAILED', error=str(exc), wall_s=time.monotonic()-start))
        raise

def metrics(points,frame,T,s):
    model_points,surface=model_surface(points,T,s,frame['K'],frame['shape'],frame.get('mesh'))
    if len(model_points)<200:raise ValueError('Insufficient evaluation surface')
    obs=frame['validation_points'];dist,_=cKDTree(surface).query(obs,workers=1)
    return {'heldout_depth_points':len(obs),'median_surface_distance_m':float(np.median(dist)),
            'p90_surface_distance_m':float(np.quantile(dist,.9)),
            'meaning':'Input measurement consistency on unseen pixel blocks; not official position/orientation accuracy'}


def silhouette(vertices,faces,frame,T,s):
    h,w=frame['shape'];factor=min(1.,512/w)
    shape=(round(h*factor),round(w*factor));K=frame['K'].copy();K[:2]*=factor
    observed=np.asarray(Image.fromarray(frame['mask']).resize((shape[1],shape[0]),Image.Resampling.NEAREST))>0
    validity=np.asarray(Image.fromarray(frame['validity']).resize((shape[1],shape[0]),Image.Resampling.NEAREST))>0
    transformed=(s*vertices)@T[:3,:3].T+T[:3,3]
    predicted=raster(transformed,faces,K,shape)>0
    return {'mask_iou':float((observed&predicted).sum()/max(1,(observed|predicted).sum())),
            'valid_observed_mask_coverage':float(((predicted)&observed&validity).sum()/max(1,(observed&validity).sum())),
            'projected_to_observed_area_ratio':float(predicted.sum()/max(1,observed.sum())),
            'metric_resolution_hw':list(shape)}
