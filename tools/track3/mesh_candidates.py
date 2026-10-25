"""Candidate object meshes from clearer views than the reference frame (R7).

SAM3D builds each object's mesh from the reference frame alone. A thin or flat object seen edge-on,
or touching a hand or another object there, comes out as a blob (6 of the 10 scored spoons, ep1's
lid, ep19's dust pan). For each object this picks the frames where its SAM2 mask is large, fully
inside the image and clear of every hand mask and the other object's mask (at least 70% of it), and runs SAM3D on
each, with the run's stereo depth, exactly as the pipeline does on the reference frame. The meshes
are written with the SAM3D transform applied (mesh_pretransformed.obj), ready for
run_baseline_reconstruction.py --object-mesh, which estimates their size and tracks them.

Frame rule (same for every object): over frames whose mask touches no image edge and covers at
least 30% of the object's largest mask, score = clear pixels (mask minus hands and the other
object, each dilated 15 px) times the fraction of the mask that is clear; take the best frame, then
the best frame at least a quarter of the clip away from it and from the reference frame.
Usage (reconstruction host venv, GPU node, container env set):
    python tools/track3/mesh_candidates.py <src_ns> <dst_ns> <ep>:<object> [...] [--n 2]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'baseline/reconstruction/data/outputs'
WEIGHTS = REPO / 'baseline/reconstruction/data/weights/sam3d'
EDGE, DILATE, MIN_AREA, MIN_CLEAR = 4, 15, 0.3, 0.7


def mask(path, shape=None):
    if not path.is_file():
        return np.zeros(shape, bool) if shape else None
    return np.asarray(Image.open(path).convert('L')) > 0


def choose_frames(ep, obj, src, n=2, reference=0):
    """Clear views of `obj`, by the rule in the module docstring."""
    run = OUT / src / f'episode_{ep:06d}' / obj
    others = [p for p in (OUT / src / f'episode_{ep:06d}').iterdir() if p.name != obj]
    tracks = json.loads((run / 'hand_tracks.json').read_text())['tracks']
    hand_ids = [t['object_id'] for t in tracks if t.get('role', 'hand') == 'hand']
    frames = sorted(int(p.stem) for p in (run / 'masks/1').glob('*.png'))
    rows = []
    for f in frames:
        m = mask(run / 'masks/1' / f'{f:06d}.png')
        area = m.sum()
        if area == 0:
            continue
        ys, xs = np.nonzero(m)
        h, w = m.shape
        inside = xs.min() > EDGE and ys.min() > EDGE and xs.max() < w - 1 - EDGE and ys.max() < h - 1 - EDGE
        block = np.zeros_like(m)
        for i in hand_ids:
            block |= mask(run / 'masks' / str(i) / f'{f:06d}.png', m.shape)
        for o in others:
            block |= mask(o / 'masks/1' / f'{f:06d}.png', m.shape)
        clear = m & ~binary_dilation(block, iterations=DILATE)
        rows.append((f, int(area), bool(inside), int(clear.sum()), clear.sum() / area))
    top = max(r[1] for r in rows)
    ok = [r for r in rows if r[2] and r[1] >= MIN_AREA * top and r[4] >= MIN_CLEAR]
    score = {r[0]: r[3] * r[4] for r in ok}
    gap = max(1, len(frames) // 4)
    picked = []
    for f in sorted(score, key=score.get, reverse=True):
        if abs(f - reference) >= gap and all(abs(f - g) >= gap for g in picked):
            picked.append(f)
        if len(picked) == n:
            break
    return picked, {r[0]: dict(area=r[1], inside=r[2], clear=r[3], clear_fraction=round(r[4], 3)) for r in rows}


def generate(ep, obj, src, dst, frame):
    sys.path.insert(0, str(REPO / 'baseline/reconstruction/modules/v2d_pipelines'))
    from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh
    from run_ego_wilor import _apply_sam3d_transform
    run = OUT / src / f'episode_{ep:06d}' / obj
    out = OUT / dst / f'episode_{ep:06d}' / obj / f'frame_{frame:06d}'
    final = out / 'mesh_pretransformed.obj'
    if final.is_file():
        return final
    out.mkdir(parents=True, exist_ok=True)
    m = run / 'masks/1' / f'{frame:06d}.png'
    run_image_to_mesh(
        image_path=str(run / 'frames' / f'{frame:06d}.png'), mask_path=str(m),
        mesh_path=str(out / 'mesh.obj'), transform_path=str(out / 'transform.json'),
        intrinsics_path=str(out / 'intrinsics.json'), weights_dir=str(WEIGHTS),
        with_texture_baking=True, with_mesh_postprocess=True,
        depth_path=str(run / 'depth' / f'{frame:06d}.png'),
        depth_intrinsics_path=str(run / 'intrinsics_stable.json'), depth_mask_path=str(m), dev=True)
    _apply_sam3d_transform(str(out / 'mesh.obj'), str(out / 'transform.json'), str(final))
    return final


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('src')
    p.add_argument('dst')
    p.add_argument('targets', nargs='+', metavar='EP:OBJECT')
    p.add_argument('--n', type=int, default=2)
    p.add_argument('--select-only', action='store_true', help='print the chosen frames, run nothing')
    args = p.parse_args()
    for t in args.targets:
        ep, obj = t.split(':')
        ep = int(ep)
        frames, stats = choose_frames(ep, obj, args.src, args.n)
        rec = OUT / args.dst / f'episode_{ep:06d}' / obj
        rec.mkdir(parents=True, exist_ok=True)
        (rec / 'candidate_frames.json').write_text(json.dumps(dict(frames=frames, source=args.src,
            stats={str(f): stats[f] for f in frames + [0] if f in stats}), indent=1))
        print(f'{ep}:{obj} frames {frames} ' + ' '.join(f'{f}:{stats[f]}' for f in frames), flush=True)
        if not args.select_only:
            for f in frames:
                print(f'  -> {generate(ep, obj, args.src, args.dst, f)}', flush=True)


if __name__ == '__main__':
    main()
