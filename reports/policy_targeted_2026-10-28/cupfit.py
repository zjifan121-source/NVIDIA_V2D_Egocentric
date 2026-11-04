import sys, numpy as np, trimesh
d = sys.argv[1]
for name in ('blue_cup', 'beige_cup'):
    m = trimesh.load(f'{d}/ep10_ken_c_scored_attach_{name}_visual.stl', force='mesh')
    v = m.vertices; lo, hi = v.min(0), v.max(0)
    print(name, 'bounds cm', np.round(100*lo, 1), np.round(100*hi, 1), 'watertight', m.is_watertight)
    c = v[:, :2].mean(0); z = v[:, 2]; r = np.linalg.norm(v[:, :2] - c, axis=1)
    for q in np.linspace(0.05, 0.95, 7):
        h = lo[2] + q * (hi[2] - lo[2]); s = np.abs(z - h) < 0.004
        print(f'  h={100*(h-lo[2]):4.1f}cm  r_min {100*r[s].min():4.1f} r_max {100*r[s].max():4.1f}')
