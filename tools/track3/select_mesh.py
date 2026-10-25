"""Pick each object's mesh without ground truth: current mesh vs the R7 candidates (mc_a, mc_b).

Every option is a tracked bundle of the same object in the same run (same masks, depth, hands,
camera). Each is scored like fit_shape.py scores a shape: render the posed mesh at 1/4 resolution
on 40 frames spread over the clip, cost = (1 - silhouette IoU, ignoring pixels hidden behind
something nearer) + capped depth residual, mean of the best 70% of frames. Lower is better. A
candidate replaces the current mesh only if it beats it by at least 10%.
Usage (reconstruction host venv, or tools venv with opencv): python tools/track3/select_mesh.py <src_ns> <baseline_ns> <out.json> <ep>:<object> [...]
  src_ns: run with the masks and depth (fs_sam3d or fs_hands); baseline_ns: the current bundles
  (fs_noref or fs_hands_noref).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_shape import KEEP, NF, S, decode, frame_cost, render  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / 'baseline/reconstruction/data/outputs'
MARGIN = 0.10


def observations(run):
    """The NF evaluation frames (largest masks, spread over the clip) and their masks and depth."""
    oid = json.loads((run / 'object_track.json').read_text())['object_id']
    files = sorted((run / 'masks' / str(oid)).glob('*.png'))
    area = {int(p.stem): (cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0).sum() for p in files}
    ok = sorted(f for f, a in area.items() if a >= 0.3 * max(area.values()))
    frames = [ok[i] for i in np.linspace(0, len(ok) - 1, min(NF, len(ok))).astype(int)]
    obs = []
    for f in frames:
        m = cv2.imread(str(run / 'masks' / str(oid) / f'{f:06d}.png'), cv2.IMREAD_GRAYSCALE) > 0
        h, w = m.shape[0] // S, m.shape[1] // S
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        d = cv2.resize(decode(run / 'depth' / f'{f:06d}.png'), (w, h), interpolation=cv2.INTER_NEAREST)
        obs.append((f, m, d))
    return obs


def score(bundle, obs):
    d = np.load(bundle / 'result.npz')
    mesh = trimesh.load(bundle / 'mesh.obj', force='mesh', process=False)
    mesh.apply_scale(float(d['object_scale']) if 'object_scale' in d.files else 1.0)
    pts = mesh.sample(20000, seed=0)
    oc, K = d['object_to_camera_transform'].astype(np.float64), d['camera_intrinsics'].astype(np.float64)
    costs = []
    for f, m, dm in obs:
        if f >= len(oc) or np.linalg.norm(oc[f, :3, 3]) < 0.10:  # lost frame
            costs.append(2.0)
            continue
        T = oc[f]
        costs.append(frame_cost(m, dm, *render(pts @ T[:3, :3].T + T[:3, 3], K, m.shape)))
    c = np.sort(costs)
    return float(c[:max(1, int(KEEP * len(c)))].mean()), sorted((np.asarray(mesh.extents) * 100).round(1).tolist())


def main():
    src, base, out = sys.argv[1:4]
    rows = {}
    for t in sys.argv[4:]:
        ep, obj = t.split(':')
        e6 = f'episode_{int(ep):06d}'
        obs = observations(OUT / src / e6 / obj)
        current = OUT / base / e6 / obj / 'result_slam_gravity_aligned'
        if not (current / 'result.npz').is_file():  # episodes that kept the refined scene
            current = OUT / src / e6 / obj / 'result_slam_gravity_aligned'
        opts = {'current': current}
        for ns in ('mc_a', 'mc_b'):
            b = OUT / ns / e6 / obj / 'result_slam_gravity_aligned'
            if (b / 'result.npz').is_file():
                opts[ns] = b
        res = {}
        for name, b in opts.items():
            s, ext = score(b, obs)
            fr = OUT / name / e6 / obj / 'candidate_frame.txt'
            res[name] = dict(cost=round(s, 4), extent_cm=ext, frame=int(fr.read_text()) if fr.is_file() else 0)
        best = min((k for k in res if k != 'current'), key=lambda k: res[k]['cost'], default=None)
        pick = best if best and res[best]['cost'] <= (1 - MARGIN) * res['current']['cost'] else 'current'
        rows[t] = dict(options=res, pick=pick)
        print(t, 'pick', pick, json.dumps(res), flush=True)
    Path(out).write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
