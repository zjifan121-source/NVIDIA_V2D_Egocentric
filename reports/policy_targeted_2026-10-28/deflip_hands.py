"""Remove impossible wrist snaps from a hub of per-frame hand fits (Path B: WiLoR fits each frame alone).

    python deflip_hands.py SRC_NS DST_NS [--eps 1,5] [--max-step 40]

A wrist cannot turn more than about 40 deg between frames at 20 fps (a fast human wrist is ~18 deg/frame). A frame is
bad when both a forward and a backward walk reject it: each walk accepts a frame whose rotation is within
max_step (+15 deg per skipped frame) of the last accepted one, and re-anchors after 15 straight rejections (a real,
lasting change of the hand's pose). Bad frames are refilled by the loader's own fill_hand (slerp of the world wrist
between good frames, fingers interpolated, written back in camera frame). Objects and cameras are untouched; files
other than result.npz are hard-linked. Writes deflip.json per episode.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, '/home/edmondwa/NVIDIA_V2D_Egocentric/tools/track3')
from run_load_retarget import OUTPUTS, fill_hand  # noqa: E402


def walk(rot, valid, order, max_step, rerun=15):
    ok = np.zeros(len(rot), bool); last = None; run = []
    for f in order:
        if not valid[f]:
            continue
        if last is None:
            ok[f] = True; last = f; continue
        lim = max_step + 15 * (abs(f - last) - 1)
        if np.degrees((rot[f] * rot[last].inv()).magnitude()) <= lim:
            ok[f] = True; last = f; run = []
        else:
            run.append(f)
            if len(run) >= rerun:  # a lasting change: accept from where it began
                ok[run] = True; last = run[-1]; run = []
    return ok


def bad_frames(data, side, max_step):
    c2w = data['camera_to_world_transform'].astype(np.float64)
    rot = Rotation.from_matrix(c2w[:, :3, :3] @ Rotation.from_rotvec(data[f'hand_{side}_wrist_orient_in_camera'].astype(np.float64)).as_matrix())
    valid = data[f'hand_{side}_is_valid'].astype(bool)
    n = len(valid)
    fwd = walk(rot, valid, range(n), max_step); bwd = walk(rot, valid, range(n - 1, -1, -1), max_step)
    return valid & ~fwd & ~bwd


def max_step_deg(data, side):
    T = data[f'hand_{side}_wrist_to_world_transform'].astype(np.float64); v = data[f'hand_{side}_is_valid'].astype(bool)
    r = Rotation.from_matrix(T[:, :3, :3]); d = np.degrees((r[1:] * r[:-1].inv()).magnitude())[v[1:] & v[:-1]]
    return round(float(d.max()), 1) if len(d) else None, int((d > 60).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('--eps'); ap.add_argument('--max-step', type=float, default=40.0)
    a = ap.parse_args()
    src, dst = OUTPUTS / a.src, OUTPUTS / a.dst
    eps = sorted(src.glob('episode_*')) if not a.eps else [src / f'episode_{int(e):06d}' for e in a.eps.split(',')]
    for ep in eps:
        info = {}
        for res in sorted(ep.glob('*/result_slam_gravity_aligned/result.npz')):
            obj = res.parent.parent.name; out = dst / ep.name / obj / 'result_slam_gravity_aligned'; out.mkdir(parents=True, exist_ok=True)
            for f in res.parent.iterdir():
                if f.name != 'result.npz' and not (out / f.name).exists():
                    os.link(f, out / f.name)
            for f in res.parent.parent.iterdir():
                if f.is_file() and not (out.parent / f.name).exists():
                    os.link(f, out.parent / f.name)
            data = dict(np.load(res)); info[obj] = {}
            for side in ('left', 'right'):
                before = max_step_deg(data, side); bad = bad_frames(data, side, a.max_step)
                fill_hand(data, side, bad)
                info[obj][side] = dict(bad_frames=int(bad.sum()), max_step_before=before, max_step_after=max_step_deg(data, side))
            np.savez(out / 'result.npz', **data)
        (dst / ep.name / 'deflip.json').write_text(json.dumps(info, indent=1))
        first = next(iter(info.values()))
        print(ep.name, {s: (v['bad_frames'], v['max_step_before'], v['max_step_after']) for s, v in first.items()}, flush=True)


if __name__ == '__main__':
    main()
