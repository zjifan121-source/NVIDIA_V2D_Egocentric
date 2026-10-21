"""Score candidate meshes against the released scans with the official CD-O code (public objects only).

For each <ep>:<object> and each namespace, loads the mesh (fs_sam3d-style bundle mesh scaled by
object_scale, or a mesh_fit.obj when present) and reports CD-O cm (official rigid alignment +
symmetric mean surface distance) and the per-axis size error vs the scan.
Usage (tools venv, repo root): python tools/track3/eval_mesh.py <out.json> <ns,...> <ep>:<object> [...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_local as S  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / 'baseline/reconstruction/data/outputs'


def load(ns, ep, obj):
    run = OUT / ns / f'episode_{ep:06d}' / obj
    for name in ('mesh_fit.obj',):
        if (run / name).is_file():
            return trimesh.load(run / name, force='mesh', process=False)
    b = run / 'result_slam_gravity_aligned'
    m = trimesh.load(b / 'mesh.obj', force='mesh', process=False)
    d = np.load(b / 'result.npz')
    m.apply_scale(float(d['object_scale']) if 'object_scale' in d.files else 1.0)
    return m


def main():
    out, nss = Path(sys.argv[1]), sys.argv[2].split(',')
    cdo = S.metric_module('CD-O')
    rows = []
    for a in sys.argv[3:]:
        ep, obj = a.split(':')
        ep = int(ep)
        ref = S.reference_object(obj)
        scan = trimesh.load(S.scan_path(obj), force='mesh', process=False)
        se = np.sort(scan.extents)
        for ns in nss:
            try:
                m = load(ns, ep, obj)
            except Exception as err:  # noqa: BLE001
                rows.append({'ep': ep, 'obj': obj, 'ns': ns, 'error': repr(err)[:200]})
                continue
            cd = cdo._object_chamfer_cm((ref['vertices'], ref['faces']),
                                        (np.asarray(m.vertices), np.asarray(m.faces, dtype=np.int64)), 1000 * ep + 7)
            me = np.sort(m.extents)
            rows.append({'ep': ep, 'obj': obj, 'ns': ns, 'cdo_cm': round(float(cd), 3),
                         'extent_cm': (me * 100).round(1).tolist(), 'scan_cm': (se * 100).round(1).tolist(),
                         'size_err': round(float(np.mean(np.abs(np.log(me / se)))), 3)})
            print(json.dumps(rows[-1]), flush=True)
    out.write_text(json.dumps(rows, indent=1))
    for ns in nss:
        r = [x for x in rows if x['ns'] == ns and 'cdo_cm' in x]
        if r:
            print(ns, 'mean CD-O cm', round(np.mean([x['cdo_cm'] for x in r]), 3),
                  'mean |log size err|', round(np.mean([x['size_err'] for x in r]), 3), len(r))


if __name__ == '__main__':
    main()
