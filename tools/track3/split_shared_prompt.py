"""Give each object its own box when two objects of an episode share a detection prompt (R5).

The two wooden pieces are both prompted "wooden board"; Grounding DINO's single best box for that is
the round wooden table under both boards, so both runs track the table. Given Grounding DINO's
detections for a shared prompt at a lower threshold (--detections, e.g. "board" at box 0.25 / text
0.2), this drops every box that contains two or more of the others (the table), keeps the most
confident remaining boxes, one per object, orders them left to right, and gives the k-th object
sharing the prompt (episode list order) the k-th box. The box is written as dino_detections.json into <dst_ns>, next to
hard links of the run's frames, depth, camera and hand detections, so rerunning the pipeline there
(run_baseline_reconstruction.py --branch <dst_ns>) re-prompts SAM2 per object and rebuilds the masks,
mesh, track and hands. Pass --order right to give the first object the rightmost box.
Usage: python tools/track3/split_shared_prompt.py <src_ns> <dst_ns> <episode> --detections <dino.json> [--order left|right]
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'baseline/reconstruction/data/outputs'
MANIFEST = Path(__file__).with_name('episodes.json')
KEEP = ('frames', 'depth', 'anycalib', 'intrinsics', 'intrinsics_stable.json', 'depth_scale.json', 'stereo_seed.json',
        'wilor_raw', 'hand_detections.json', 'geocalib', 'slam_poses', 'slam_trajectory.txt')


def link(src: Path, dst: Path) -> None:
    if dst.exists() or not src.exists():
        return
    if src.is_dir():
        dst.mkdir(parents=True)
        for p in src.iterdir():
            link(p, dst / p.name)
    else:
        os.link(src, dst)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('src')
    p.add_argument('dst')
    p.add_argument('episode', type=int)
    p.add_argument('--detections', type=Path, required=True, help='Grounding DINO output for the shared prompt')
    p.add_argument('--order', choices=('left', 'right'), default='left')
    args = p.parse_args()
    ep = next(e for e in json.loads(MANIFEST.read_text())['episodes'] if e['episode'] == args.episode)
    groups = defaultdict(list)
    for o in ep['objects']:
        groups[o['prompt']].append(o['name'])
    for prompt, names in groups.items():
        if len(names) < 2:
            continue
        dets = json.loads(args.detections.read_text())

        def inside(b, c):  # b within c (5 px slack)
            return b['x0'] >= c['x0'] - 5 and b['y0'] >= c['y0'] - 5 and b['x1'] <= c['x1'] + 5 and b['y1'] <= c['y1'] + 5

        keep = [d for d in dets if sum(inside(e['box'], d['box']) for e in dets if e is not d) < 2]
        keep = sorted(keep, key=lambda d: -d['confidence'])[:len(names)]
        if len(keep) != len(names):
            raise SystemExit(f'{prompt!r}: {len(keep)} separate boxes for {len(names)} objects {names}; not splitting')
        boxes = [d['box'] for d in keep]
        det = keep[0]
        boxes.sort(key=lambda b: b['x0'] + b['x1'], reverse=args.order == 'right')
        for name, box in zip(names, boxes):
            dst = OUT / args.dst / f'episode_{args.episode:06d}' / name
            dst.mkdir(parents=True, exist_ok=True)
            src = OUT / args.src / f'episode_{args.episode:06d}' / name
            for item in KEEP:
                link(src / item, dst / item)
            (dst / 'dino_detections.json').write_text(json.dumps(
                [{'label': det['label'], 'confidence': det['confidence'], 'box': box,
                  'note': f'shared prompt split: {args.order}-to-right order, containing box dropped'}], indent=1))
            print(f'ep{args.episode} {name}: box {[round(v) for v in box.values()]} ({args.order} order)')


if __name__ == '__main__':
    main()
