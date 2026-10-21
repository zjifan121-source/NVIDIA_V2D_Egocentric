"""Copy a run's result bundle with its mesh replaced by the fitted one (poses untouched).

The fitted mesh (fit_shape.py) is in metres and in the run's own object frame, so the bundle keeps
its poses and gets object_scale 1. Lets the playback scorer check that a mesh swap alone does not
cost AUC, and gives the meshes to submit.
Usage: python tools/track3/swap_mesh.py <src_ns> <fit_ns> <dst_ns> <ep>:<object> [...]
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

OUT = Path(__file__).resolve().parents[2] / 'baseline/reconstruction/data/outputs'


def swap(src, fit, dst, ep, obj):
    run = OUT / src / f'episode_{ep:06d}' / obj
    fitted = OUT / fit / f'episode_{ep:06d}' / obj / 'mesh_fit.obj'
    out = OUT / dst / f'episode_{ep:06d}' / obj
    ob = out / 'result_slam_gravity_aligned'
    if ob.exists():
        shutil.rmtree(ob)
    shutil.copytree(run / 'result_slam_gravity_aligned', ob, ignore=shutil.ignore_patterns('threejs_scene'))
    shutil.copy(run / 'object_track.json', out / 'object_track.json')
    if not fitted.is_file():  # fit skipped (too few frames): keep the original mesh
        return f'{ep}:{obj} kept original'
    trimesh.load(fitted, force='mesh', process=False).export(ob / 'mesh.obj')
    d = dict(np.load(ob / 'result.npz'))
    d['object_scale'] = np.float32(1.0)
    np.savez(ob / 'result.npz', **d)
    return f'{ep}:{obj} swapped'


if __name__ == '__main__':
    src, fit, dst = sys.argv[1:4]
    for a in sys.argv[4:]:
        ep, obj = a.split(':')
        print(swap(src, fit, dst, int(ep), obj), flush=True)
