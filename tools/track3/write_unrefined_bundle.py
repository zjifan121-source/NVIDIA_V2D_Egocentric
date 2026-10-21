"""Result bundle without the joint refinement, written by the pipeline's own writer.

The joint gsplat refinement can jitter the camera (ep13: 0.17 m of DROID-SLAM motion becomes 14.7 m
of path), so static objects wander and hands shrink (scale down to 0.2). On the 8 practice clips,
bundles without it play back at 0.625 official AUC against 0.511. Use it when the refined camera
path is more than 6x the raw DROID-SLAM path (2 of 8 practice clips keep the refinement).

Same as run_ego_wilor.py's final packaging when no refinement outputs exist: raw DROID-SLAM camera
(slam_poses, camera_to_world), FoundationPose track (poses_smoothed) on the scale-estimated mesh
(mesh_scaled.obj), depth-aligned HaMeR hands (hamer_aligned_filled), then the same gravity
alignment (geocalib/calibration.json). Writes outputs/<dst>/<episode>/<object>/{result,
result_slam_gravity_aligned}/ and object_track.json. CPU only.
Usage (reconstruction host venv): python tools/track3/write_unrefined_bundle.py <src_ns> <dst_ns> <ep>:<object> [...]
"""
import json
import shutil
import sys
from pathlib import Path

from v2d.common.result_bundle import gravity_align_result_bundle, write_result_bundle

OUT = Path(__file__).resolve().parents[2] / 'baseline/reconstruction/data/outputs'


def hand_dirs(run: Path) -> tuple[str | None, str | None]:
    """Left/right hand track folders, as run_ego_wilor._result_hand_dirs picks them."""
    root = run / 'hamer_aligned_filled'
    tracks = json.loads((run / 'hand_tracks.json').read_text()).get('tracks', [])
    out = []
    for right in (False, True):
        ids = [int(t['object_id']) for t in tracks if t.get('role', 'hand') == 'hand' and bool(t.get('is_right')) == right]
        out.append(next((str(root / str(i)) for i in ids if (root / str(i)).is_dir() and any((root / str(i)).iterdir())), None))
    return out[0], out[1]


def build(src: str, dst: str, ep: int, obj: str) -> str:
    run = OUT / src / f'episode_{ep:06d}' / obj
    out = OUT / dst / f'episode_{ep:06d}' / obj
    out.mkdir(parents=True, exist_ok=True)
    for name in ('result', 'result_slam_gravity_aligned'):
        if (out / name).exists():
            shutil.rmtree(out / name)
    left, right = hand_dirs(run)
    write_result_bundle(
        result_dir=str(out / 'result'),
        frames_dir=str(run / 'frames'),
        intrinsics_path=str(run / 'intrinsics_stable.json'),
        mesh_path=str(run / 'mesh_scaled.obj'),
        object_poses_dir=str(run / 'poses_smoothed'),
        object_scale=1.0,
        camera_to_world_dir=str(run / 'slam_poses'),
        camera_pose_convention='camera_to_world',
        left_hand_dir=left,
        right_hand_dir=right,
        source_manifest={'pipeline': 'ego_wilor', 'variant': 'no_refinement', 'source_run': str(run)},
    )
    shutil.copytree(out / 'result', out / 'result_slam_gravity_aligned')
    gravity_align_result_bundle(
        result_dir=str(out / 'result_slam_gravity_aligned'),
        geocalib_calibration_path=str(run / 'geocalib' / 'calibration.json'),
        target='z_up',
    )
    shutil.copy(run / 'object_track.json', out / 'object_track.json')
    return f'{ep}:{obj} hands L={Path(left).name if left else None} R={Path(right).name if right else None}'


if __name__ == '__main__':
    src, dst = sys.argv[1:3]
    for a in sys.argv[3:]:
        ep, obj = a.split(':')
        print(build(src, dst, int(ep), obj), flush=True)
