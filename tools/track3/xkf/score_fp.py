"""Cross-view score of every finished draw of a plan, inside the FoundationPose runtime (standalone script: the image's
python, the repo's modules at /workspace as the pipeline mounts them, and the plan's paths bound at the same paths).

SAM 3D sees one side of the object and invents the rest, so a mesh always fits its own frame; frames that show the
other sides test the invented part. Per candidate and held-out frame more than 15 frames from the candidate's source
frame, FoundationPose registers the mesh from scratch (its own hypothesis search, as at the pipeline's reference frame)
and the fit is scored like the pipeline's size search (FoundationPoseTracker._score_scale): mask IoU and
exp(-median |observed / rendered depth - 1|), each in [0, 1].

register() returns the pose of the uncentred mesh while _score_scale renders the centred mesh, so the scores are taken
at the placed pose, pose @ inv(tf_to_centered); scoring the raw pose renders the object off by the centring offset and
makes the ranking close to noise. The raw-pose numbers are kept as *_unplaced.

  python tools/track3/xkf/score_fp.py --plan <plan.json> --output <scores.json> [--draws <dir>] [--weights <dir>]"""
import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, '/app')
sys.path.insert(0, '/workspace')

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask  # noqa: E402
from v2d.common.datatypes import Image as V2dImage  # noqa: E402
from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker  # noqa: E402
from v2d.mesh.lib.mesh import Mesh  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--draws', help='folder with <id>/mesh_scaled.obj (default: draws/ next to the plan)')
    ap.add_argument('--weights', default='/data/weights_dir/foundation_pose')
    ap.add_argument('--iterations', type=int, default=5)
    a = ap.parse_args()
    P = json.load(open(a.plan))
    draws = a.draws or os.path.join(os.path.dirname(os.path.abspath(a.plan)), 'draws')
    run_dir, gap = P['run'], P['rules']['heldout_exclusion']
    intr = CameraIntrinsics.load(f'{run_dir}/intrinsics_stable.json')
    frames = {}
    for t in P['heldout_frames']:
        bgr = cv2.imread(f'{run_dir}/frames/{t:06d}.png')
        if bgr is None:
            raise SystemExit(f'cannot read frame {t} of {run_dir}')
        frames[t] = (V2dImage(data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), DepthImage.load(f'{run_dir}/depth/{t:06d}.png'),
                     Mask.load(f'{run_dir}/masks/1/{t:06d}.png'))
    res = {'plan': os.path.abspath(a.plan), 'draws': draws, 'frames': sorted(frames), 'iterations': a.iterations,
           'placed': True, 'candidates': {}}
    for c in P['candidates']:
        mesh = os.path.join(draws, c['id'], 'mesh_scaled.obj')
        held = [t for t in sorted(frames) if abs(t - c['frame']) > gap]
        if not os.path.isfile(mesh) or len(held) < 5:
            print(f'{c["id"]}: skipped ({"no draw" if not os.path.isfile(mesh) else "fewer than 5 held-out frames"})')
            continue
        t0 = time.time()
        tracker = FoundationPoseTracker(Mesh.load(mesh), a.weights)
        untf = np.linalg.inv(tracker._est.get_tf_to_centered_mesh().cpu().numpy())
        per = {}
        for t in held:
            rgb, depth, mask = frames[t]
            pose = tracker.register(rgb, depth, mask, intr, iteration=a.iterations)
            placed = type(pose).from_matrix(pose.to_matrix() @ untf)
            per[str(t)] = {'iou': round(float(tracker._score_scale(depth, mask, intr, placed, 1.0, 0.0, 0.0)), 4),
                           'depth': round(float(tracker._score_scale(depth, mask, intr, placed, 0.0, 1.0, 0.0)), 4),
                           'iou_unplaced': round(float(tracker._score_scale(depth, mask, intr, pose, 1.0, 0.0, 0.0)), 4),
                           'depth_unplaced': round(float(tracker._score_scale(depth, mask, intr, pose, 0.0, 1.0, 0.0)), 4)}
            gc.collect()
            torch.cuda.empty_cache()
        s = [v['iou'] + v['depth'] for v in per.values()]
        res['candidates'][c['id']] = {
            'source_frame': c['frame'], 'seed': c['seed'], 'mesh': mesh, 'per_frame': per,
            'mean': round(float(np.mean(s)), 4), 'mean_iou': round(float(np.mean([v['iou'] for v in per.values()])), 4),
            'mean_depth': round(float(np.mean([v['depth'] for v in per.values()])), 4),
            'mean_unplaced': round(float(np.mean([v['iou_unplaced'] + v['depth_unplaced'] for v in per.values()])), 4),
            'seconds': round(time.time() - t0, 1)}
        r = res['candidates'][c['id']]
        print(f"{c['id']:10s} mean {r['mean']:.3f} (iou {r['mean_iou']:.3f}, depth {r['mean_depth']:.3f}) over "
              f"{len(per)} frames, {r['seconds']:.0f} s", flush=True)
        del tracker
        gc.collect()
        torch.cuda.empty_cache()
        json.dump(res, open(a.output, 'w'), indent=1)
    json.dump(res, open(a.output, 'w'), indent=1)


if __name__ == '__main__':
    main()
