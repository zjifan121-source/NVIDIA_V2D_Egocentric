"""Choose the frames: SAM 3D draws at the key frame plus frames spread over the video where the object is well
visible, and the held-out frames that judge every draw. Reads only a finished run folder (frames/, depth/, masks/1/,
intrinsics_stable.json, and run.json for the reference frame)."""
import glob
import json
import os

import numpy as np
from PIL import Image


def _areas(mask_dir, step):
    paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))[::step]
    return {int(os.path.basename(p)[:6]): int((np.asarray(Image.open(p)) > 0).sum()) for p in paths}


def _median(area):
    nz = [a for a in area.values() if a > 0]
    if not nz:
        raise SystemExit('every object mask is empty')
    return float(np.median(nz))


def _spread(frames, n):
    return sorted({frames[int(round(i))] for i in np.linspace(0, len(frames) - 1, n)}) if frames else []


def run(run_dir, output, episode=None, object=None, key_frame=None, seeds=(1, 2), n_frames=6, min_gap=40,
        n_heldout=10, heldout_step=2, min_ratio=0.6, min_area=2000):
    run_dir = os.path.abspath(run_dir)
    masks = f'{run_dir}/masks/1'
    rec = json.load(open(f'{run_dir}/run.json')) if os.path.isfile(f'{run_dir}/run.json') else {}
    if key_frame is None:
        if 'reference_frame' not in rec:
            raise SystemExit(f'no run.json with a reference_frame in {run_dir}: pass --key-frame')
        key_frame = int(rec['reference_frame'])
    elif 'reference_frame' in rec and int(rec['reference_frame']) != key_frame:
        print(f"warning: --key-frame {key_frame} differs from the run's reference frame {rec['reference_frame']}")
    if not os.path.isfile(f'{run_dir}/intrinsics_stable.json'):
        raise SystemExit(f'missing {run_dir}/intrinsics_stable.json')

    def present(t):
        return all(os.path.isfile(f'{run_dir}/{d}/{t:06d}.png') for d in ('frames', 'depth'))

    # draw frames: every mask; the key frame plus n_frames - 1 frames spread over those far enough from it
    area = _areas(masks, 1)
    if not area:
        raise SystemExit(f'no object masks in {masks}')
    med = _median(area)
    ok = sorted(t for t, a in area.items()
                if a >= min_ratio * med and a >= min_area and abs(t - key_frame) > min_gap and present(t))
    spread = _spread(ok, n_frames - 1)
    # the key frame is drawn only if its mask is at least 0.3 x the median of every fifth mask (as the pipeline's
    # draws were checked); otherwise its candidates are left out
    key_ok = present(key_frame) and area.get(key_frame, 0) >= 0.3 * _median(_areas(masks, 5))
    if not key_ok:
        print(f'warning: key frame {key_frame} has no usable mask, frame or depth: its candidates are left out')
    draw_frames = ([key_frame] if key_ok else []) + spread
    if len(spread) < n_frames - 1:
        print(f'warning: only {len(spread)} frames qualify besides the key frame')
    # held-out frames: n_heldout spread over the frames (every heldout_step-th mask) where the object is well visible
    area2 = _areas(masks, heldout_step)
    med2 = _median(area2)
    heldout = _spread(sorted(t for t, a in area2.items() if a >= min_ratio * med2 and a >= min_area and present(t)),
                      n_heldout)
    if len(heldout) < 5:
        raise SystemExit(f'only {len(heldout)} held-out frames qualify (frames/ or depth/ missing in {run_dir}?)')
    plan = {'run': run_dir, 'episode': episode, 'object': object, 'key_frame': key_frame, 'seeds': list(seeds),
            'draw_frames': draw_frames, 'heldout_frames': heldout, 'median_mask_px': med,
            'candidates': [{'id': f'f{f}s{s}', 'frame': f, 'seed': s} for f in draw_frames for s in seeds],
            'rules': {'min_ratio': min_ratio, 'min_area': min_area, 'min_gap': min_gap, 'heldout_step': heldout_step,
                      'heldout_exclusion': 15}}
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    json.dump(plan, open(output, 'w'), indent=1)
    print(f'draw frames {draw_frames} x seeds {list(seeds)} = {len(plan["candidates"])} candidates; '
          f'held-out frames {heldout}')
