import sys, glob, numpy as np, trimesh
S = sys.argv[1]
for n in ('blue_cup', 'beige_cup'):
    v = trimesh.load(glob.glob(f'{S}/assets/human_motion_data/ego_recon/processed/*{n}_visual.stl')[0], force='mesh')
    s = trimesh.load(glob.glob(f'{S}/score_input/meshes/*/{n}.obj')[0], force='mesh', process=False)
    print(n, 'visual: verts', len(v.vertices), 'vol cm3', round(v.volume*1e6), 'hull vol', round(v.convex_hull.volume*1e6), 'extent', np.round(v.extents*100, 1))
    print(n, 'score : verts', len(s.vertices), 'hull vol', round(s.convex_hull.volume*1e6), 'extent', np.round(s.extents*100, 1), 'same verts?', len(v.vertices) == len(s.vertices))
    print('  pieces', len(glob.glob(f'{S}/assets/human_motion_data/ego_recon/processed/{n}_collision/*.stl')))
