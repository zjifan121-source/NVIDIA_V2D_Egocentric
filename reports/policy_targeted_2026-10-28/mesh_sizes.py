"""Extents of every reconstructed mesh of one object across our scenes and bundles (no scans)."""
import glob, sys, numpy as np, trimesh
name = sys.argv[1]
pats = [f'/data/user_data/edmondwa/v2d/ken_night/variants/*/score_input/meshes/*/{name}.obj',
        f'/data/user_data/edmondwa/v2d/ken_night/night_2026-10-24/scenes/*/*/score_input/meshes/*/{name}.obj',
        f'/data/user_data/edmondwa/v2d/recon_data/outputs/load_retarget/*/*/score_input/meshes/*/{name}.obj',
        f'/data/user_data/edmondwa/v2d/ken_night/night_2026-10-24/bundles/*/*/{name}/result_slam_gravity_aligned/mesh.obj']
seen = set()
for p in pats:
    for f in sorted(glob.glob(p)):
        m = trimesh.load(f, force='mesh', process=False)
        e = np.sort(m.extents)[::-1] * 100
        key = tuple(np.round(e, 1))
        tag = '' if key not in seen else ' (dup)'
        seen.add(key)
        print(f'{e[0]:5.1f} {e[1]:5.1f} {e[2]:5.1f} cm  {f.split("/v2d/")[1]}{tag}')
