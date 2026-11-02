"""One fixed-seed SAM3D call per preselected, input-reviewed view."""
from pathlib import Path
import inspect
import time
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from .io import Inputs, artifact, copy_appearance, fresh, read, seal, sha, write


def prepare(manifest, candidate, output):
    inputs = Inputs(manifest)
    views = inputs.data.get('candidate_views', [])
    ids = [v['frame_id'] for v in views]
    if not 1 <= len(views) <= 2 or len(set(ids)) != len(ids):
        raise ValueError('Preselect one or two distinct clear views')
    if not 1 <= candidate <= len(views):
        raise ValueError('Candidate index outside frozen pool')
    view = views[candidate - 1]
    if any(view.get(k) is not True for k in ('clear_visible_structure', 'no_neighbor_contamination', 'mask_review_pass')):
        raise ValueError('Input visibility and mask review required')
    rows = inputs.rows()
    if {r['id'] for r in rows} != set(ids):
        raise ValueError('Generation frames must match frozen candidate views')
    f = inputs.frame(next(r for r in rows if r['id'] == view['frame_id']))
    out = fresh(output)
    h, w = f['shape']
    y, x = np.indices((h, w), dtype=np.float32)
    rays = np.stack([x, y, np.ones_like(x)], axis=-1) @ np.linalg.inv(f['K']).T
    pointmap = (rays * f['depth'][..., None]).astype(np.float32)
    pointmap[~f['valid']] = np.nan
    Image.fromarray(f['rgb']).save(out / 'rgb.png')
    Image.fromarray((f['mask'] * 255).astype(np.uint8)).save(out / 'mask.png')
    np.save(out / 'pointmap.npy', pointmap)
    K = f['K']
    write(out / 'K.json', dict(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]),
                               cy=float(K[1, 2]), width=w, height=h))
    inputs.unchanged()
    return seal(out, dict(state='GENERATION_INPUT_READY', context=inputs.context,
        candidate_index=candidate, reference_frame=f['id'], seed=0, GT_used=False,
        input_sha256=f['input_sha256'], input_review=view, input_manifest_sha256=sha(manifest),
        pointmap_convention='OpenCV camera XYZ metres, invalid and non-object samples NaN'))


def bake_generated(source, transform, dest):
    """Match the experiment: bake SAM3D R and uniform s, not translation."""
    t = read(transform)
    s = np.asarray(t['scale'], float)
    if s.shape != (3,) or not np.isfinite(s).all() or np.any(s <= 0) or not np.allclose(s, s[0]):
        raise ValueError('SAM3D scale must be positive and uniform')
    q = np.asarray(t['rotation'], float)
    R = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    lines = []
    for line in Path(source).read_text().splitlines(keepends=True):
        if line.startswith(('v ', 'vn ')):
            p = line.split()
            v = R @ (np.array(p[1:4], float) * (s if p[0] == 'v' else 1))
            line = p[0] + ' ' + ' '.join(format(x, '.17g') for x in v)
            line += (' ' + ' '.join(p[4:]) if len(p) > 4 else '') + '\n'
        lines.append(line)
    Path(dest).write_text(''.join(lines))
    copy_appearance(source, dest)
    return dict(rotation_baked=R.tolist(), uniform_scale_baked=float(s[0]), translation_baked=False)


def run(prepared, weights, output):
    p = Path(prepared).resolve()
    d = artifact(p / 'artifact.json', {'GENERATION_INPUT_READY'})
    prepared_hash = sha(p / 'artifact.json')
    out = fresh(output)
    start = time.monotonic()
    try:
        # Available only in the existing SAM3D runtime; CPU commands never import it.
        from v2d.sam3d.lib.image_to_mesh import image_to_mesh
        raw = out / 'raw'
        raw.mkdir()
        image_to_mesh(str(p / 'rgb.png'), str(p / 'mask.png'), str(raw / 'mesh.obj'),
                      str(raw / 'transform.json'), str(raw / 'intrinsics.json'), str(weights),
                      seed=0, with_mesh_postprocess=True, with_texture_baking=True,
                      pointmap_path=str(p / 'pointmap.npy'), pointmap_intrinsics_path=str(p / 'K.json'))
        baked = bake_generated(raw / 'mesh.obj', raw / 'transform.json', out / 'mesh.obj')
        artifact(p / 'artifact.json', {'GENERATION_INPUT_READY'})
        if sha(p / 'artifact.json') != prepared_hash:
            raise ValueError('Prepared input manifest changed during generation')
        write(out / 'structure_review.template.json', dict(state='PENDING', GT_used=False,
            mesh_sha256=sha(out / 'mesh.obj'), critical_features={},
            critical_missing_or_extraneous_count=None,
            instructions='Review baked mesh against RGB/masks. Record visible critical structure and contamination.'))
        # Copy the template outside this sealed output before recording the review.
        return seal(out, dict(state='GENERATED_PENDING_STRUCTURE_REVIEW', context=d['context'],
            seed=0, candidate_index=d['candidate_index'], reference_frame=d['reference_frame'],
            mesh_sha256=sha(out / 'mesh.obj'), baked_transform=baked,
            input_artifact_sha256=prepared_hash,
            generator_source_sha256=sha(inspect.getfile(image_to_mesh)), wall_s=time.monotonic()-start))
    except Exception as exc:
        write(out / 'failure.json', dict(state='FAILED', error=str(exc), wall_s=time.monotonic()-start))
        raise
