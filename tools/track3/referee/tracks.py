"""CoTracker3 2D point tracks of one object (GPU). The tracks depend only on the video and the object mask, not on any
reconstruction, so they are computed once per episode and object.

Queries: up to 64 points inside the eroded object mask at every 30th frame, tracked forward and backward over the whole
episode with the offline model at 1/4 resolution (window 60). Output npz: tracks (T, N, 2) and vis (T, N) in full-
resolution pixels, queries (N, 3) = (frame, x, y), size = (W, H)."""
import time
from pathlib import Path

import numpy as np


def _frames(frames_dir=None, video=None):
    import cv2
    if frames_dir:
        files = sorted(Path(frames_dir).glob('*.png'))
        if not files:
            raise SystemExit(f'no PNG frames in {frames_dir}')
        return [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files]
    cap = cv2.VideoCapture(str(video))
    imgs = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    if not imgs:
        raise SystemExit(f'could not read frames from {video}')
    return imgs


def run(masks, checkpoint, output, frames=None, video=None, every=30, points=64, scale=4, window=60, seed=0):
    import cv2
    import torch
    from cotracker.predictor import CoTrackerPredictor

    if bool(frames) == bool(video):
        raise SystemExit('give exactly one of --frames or --video')
    output = str(output) if str(output).endswith('.npz') else f'{output}.npz'   # np.savez appends it anyway
    imgs = _frames(frames, video)
    H, W = imgs[0].shape[:2]
    small = np.stack([cv2.resize(im, (W // scale, H // scale), interpolation=cv2.INTER_AREA) for im in imgs])
    clip = torch.from_numpy(small).permute(0, 3, 1, 2)[None].float().cuda()   # (1, T, 3, h, w)
    rng = np.random.default_rng(seed)
    Q = []
    for q in range(0, len(imgs), every):
        m = cv2.imread(str(Path(masks) / f'{q:06d}.png'), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = cv2.erode((m > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        ys, xs = np.nonzero(m)
        if len(xs) < 20:
            continue
        k = rng.choice(len(xs), min(points, len(xs)), replace=False)
        Q += [[q, xs[i] / scale, ys[i] / scale] for i in k]
    if not Q:
        raise SystemExit(f'no query points: masks in {masks} are empty at every {every}th frame')
    Q = np.array(Q, np.float32)
    print(f'{len(imgs)} frames {W}x{H}, {len(Q)} queries at frames {sorted(set(Q[:, 0].astype(int)))}', flush=True)
    model = CoTrackerPredictor(checkpoint=str(checkpoint), offline=True, window_len=window).cuda()
    t0 = time.time()
    tracks, vis = model(clip, queries=torch.from_numpy(Q)[None].cuda(), backward_tracking=True)
    print(f'tracked in {time.time() - t0:.0f} s', flush=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, tracks=tracks[0].cpu().numpy() * scale, vis=vis[0].cpu().numpy(),
                        queries=np.c_[Q[:, :1], Q[:, 1:] * scale], size=[W, H])
    print('TRACKS_OK', output, flush=True)
