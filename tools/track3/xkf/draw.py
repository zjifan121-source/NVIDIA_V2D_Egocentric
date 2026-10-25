"""SAM 3D draws for a plan, made the way the pipeline's object branch makes its mesh (run_ego_wilor.py step 8:
image_to_mesh with texture baking, mesh post-process and the stereo depth; the SAM 3D transform; FoundationPose scale
estimation), one fixed seed per candidate. Runs in the reconstruction host environment (the one that runs
run_ego_reconstruction.py, with its container settings); writes <plan dir>/draws/<id>/{mesh/, mesh_pretransformed.obj,
mesh_scaled.obj, scale.json, draw.json}. A candidate whose mesh_scaled.obj exists is skipped; an existing
mesh/textured_mesh.obj is kept and only the transform and the scale step are redone. A failed candidate is recorded in
draws/<id>/error.txt and the others still run."""
import json
import os
import sys
import time
import traceback
from pathlib import Path

RECON = Path(__file__).resolve().parents[3] / 'baseline' / 'reconstruction'


def run(plan, candidate=None, dev=True, recon_dir=None):
    plan = os.path.abspath(plan)
    P = json.load(open(plan))
    root = os.path.join(os.path.dirname(plan), 'draws')
    todo = [c for c in P['candidates'] if not candidate or c['id'] in candidate]
    if candidate and len(todo) != len(set(candidate)):
        raise SystemExit(f'unknown candidate in {candidate}; the plan has {[c["id"] for c in P["candidates"]]}')
    os.environ.setdefault('APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.chdir(recon_dir or RECON)   # the pipeline's relative weight paths
    sys.path.insert(0, os.path.abspath('modules/v2d_pipelines'))
    from run_ego_wilor import _apply_sam3d_transform
    from v2d.foundation_pose.docker.run_estimate_mesh_scale import run_estimate_mesh_scale
    from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh
    run_dir, failed = P['run'], []
    for c in todo:
        n, out = c['frame'], os.path.join(root, c['id'])
        if os.path.isfile(f'{out}/mesh_scaled.obj'):
            print(f'{c["id"]}: done already')
            continue
        rgb, depth = f'{run_dir}/frames/{n:06d}.png', f'{run_dir}/depth/{n:06d}.png'
        mask, K = f'{run_dir}/masks/1/{n:06d}.png', f'{run_dir}/intrinsics_stable.json'
        os.makedirs(f'{out}/mesh', exist_ok=True)
        try:
            for p in (rgb, depth, mask, K):
                if not os.path.isfile(p):
                    raise FileNotFoundError(p)
            t0 = time.time()
            sam3d = not os.path.isfile(f'{out}/mesh/textured_mesh.obj')
            if sam3d:
                run_image_to_mesh(
                    image_path=rgb, mask_path=mask, mesh_path=f'{out}/mesh/textured_mesh.obj',
                    transform_path=f'{out}/mesh/mesh_transform.json',
                    intrinsics_path=f'{out}/mesh/mesh_intrinsics.json', weights_dir='data/weights/sam3d',
                    seed=c['seed'], with_texture_baking=True, with_mesh_postprocess=True, depth_path=depth,
                    depth_intrinsics_path=K, depth_mask_path=mask, dev=dev)
            t1 = time.time()
            _apply_sam3d_transform(f'{out}/mesh/textured_mesh.obj', f'{out}/mesh/mesh_transform.json',
                                   f'{out}/mesh_pretransformed.obj')
            run_estimate_mesh_scale(
                mesh_path=f'{out}/mesh_pretransformed.obj', rgb_path=rgb, depth_path=depth, mask_path=mask,
                intrinsics_path=K, weights_dir='data/weights/foundation_pose', scale_path=f'{out}/scale.json',
                rescaled_mesh_path=f'{out}/mesh_scaled.obj', lo=0.5, hi=2.0, n_samples=9, n_levels=4,
                iou_weight=1.0, depth_weight=1.0, registration_iterations=5, dev=dev)
        except Exception:
            failed.append(c['id'])
            open(f'{out}/error.txt', 'w').write(traceback.format_exc())
            print(f'{c["id"]}: FAILED (see {out}/error.txt)', flush=True)
            continue
        rec = {'run': run_dir, 'frame': n, 'seed': c['seed'], 'sam3d': sam3d, 'sam3d_s': round(t1 - t0),
               'scale_s': round(time.time() - t1), 'scale': json.load(open(f'{out}/scale.json'))}
        json.dump(rec, open(f'{out}/draw.json', 'w'), indent=1)
        print(f'{c["id"]}: drawn in {time.time() - t0:.0f} s', flush=True)
    if failed:
        raise SystemExit(f'{len(failed)} draw(s) failed: {failed}')
