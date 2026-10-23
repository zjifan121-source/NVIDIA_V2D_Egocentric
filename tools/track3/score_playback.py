"""Score stereo-test reconstructions exactly like the oracle study scored the hub ones.

For each output namespace (hub = today's baseline, stereo_sam3d = stereo depth + our mesh,
stereo_scan = stereo depth + released scan), build the same cleaned/merged bundle the public dry
run builds, then run the oracle's variants on it (playback, official metric code).
Usage (tools venv, repo root): python tools/track3/score_playback.py <out.json> <ns,...> <ep,...>
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import playback_oracle as O  # noqa: E402

L, S = O.L, O.S
BUNDLES = Path(os.environ.get('SCORE_BUNDLES', str(O.V2D_DATA / 'stereo_test/bundles')))
NAMES = ['recon', 'recon_smooth_hold_hand', 'still', 'still_gt0', 'place', 'place_pos_gt', 'place_rot_gt',
         'gt_motion_our_place', 'recon_ourmesh']


def build(ns, e):
    """public_dryrun.arm_data (plain arm) with bundles read from data/outputs/<ns>."""
    def bundle_dir_for(episode, obj):
        run = L.OUTPUTS / ns / f'episode_{episode:06d}' / obj
        for c in (run / 'result_slam_gravity_aligned', run):
            if (c / 'result.npz').is_file():
                return c
        raise SystemExit(f'no result.npz under {run}')
    L.bundle_dir_for = bundle_dir_for
    ep, names = e['episode'], [o['name'] for o in e['objects']]
    D = BUNDLES / ns
    D.mkdir(parents=True, exist_ok=True)
    src = L.merged_bundle(ep, D / f'e{ep}_merged')[0] if len(names) > 1 else bundle_dir_for(ep, names[0])
    b, _ = L.cleaned_bundle(src, D / f'e{ep}_plain', L.TRACK3_FPS)
    d = dict(np.load(b / 'result.npz'))
    for i, (_, sk, mf) in enumerate(L.object_slots(d)):
        m = trimesh.load(b / mf, force='mesh', process=False)
        m.apply_scale(float(d.get(sk, 1.0)))
        m.export(D / f'e{ep}_plain_{i}.obj')
    return D


def job(args):
    ns, e, name = args
    O.D = BUNDLES / ns
    try:
        r = O.job((e, name))
    except Exception as err:  # noqa: BLE001
        r = {'ep': e['episode'], 'v': name, 'error': repr(err)[:300]}
    r['ns'] = ns
    return r


def mesh_extents(ns, e):
    out = []
    for i, o in enumerate(e['objects']):
        ours = trimesh.load(BUNDLES / ns / f"e{e['episode']}_plain_{i}.obj", force='mesh', process=False)
        scan = trimesh.load(S.scan_path(o['name']), force='mesh', process=False)
        out.append({'name': o['name'], 'ours_cm': sorted((100 * ours.extents).round(1).tolist()),
                    'scan_cm': sorted((100 * scan.extents).round(1).tolist())})
    return out


if __name__ == '__main__':
    out, nss, eps = Path(sys.argv[1]), sys.argv[2].split(','), {int(x) for x in sys.argv[3].split(',')}
    E = [e for e in S.public_episodes() if e['episode'] in eps]
    built, sizes = [], {}
    for ns in nss:
        for e in E:
            try:
                build(ns, e)
                built.append((ns, e))
                sizes[f"{ns}/{e['episode']}"] = mesh_extents(ns, e)
            except SystemExit as err:
                print(f'skip {ns} ep{e["episode"]}: {err}', flush=True)
    with ProcessPoolExecutor(min(16, len(os.sched_getaffinity(0)))) as ex:  # CPUs this job may use, not the node's
        res = list(ex.map(job, [(ns, e, n) for ns, e in built for n in NAMES]))
    json.dump({'results': res, 'mesh_sizes': sizes}, open(out, 'w'), indent=1)
    by = {}
    for r in res:
        by.setdefault((r['ns'], r['v']), []).append(r)
    print(f"{'ns':14s} {'variant':24s} " + ' '.join(f'{m:>7s}' for m in ('AUC', 'SP-SR', 'RPE', 'MPPE')) + '  per-episode AUC')
    for (ns, v), rs in sorted(by.items()):
        ok = [r for r in rs if 'AUC' in r]
        if ok:
            print(f'{ns:14s} {v:24s} ' + ' '.join(f"{np.mean([r[m] for r in ok]):7.3f}" for m in ('AUC', 'SP-SR', 'RPE', 'MPPE'))
                  + '  ' + ' '.join(f"{r['ep']}:{r['AUC']:.2f}" for r in sorted(ok, key=lambda r: r['ep'])))
