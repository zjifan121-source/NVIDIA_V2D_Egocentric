"""Official scores for one training run's rollout, at full or half speed (task.motion_speed=0.5).

At half speed the evaluator writes one row per control step of the reference stretched to twice its
length, so reference_frame runs 0..2N-1 and row 2k is video frame k. This rewrites the parquet to
the even rows with reference_frame // 2 and scores it with the official metric code. Local diagnostic
only; the kit's packer does not do this conversion.
Usage (tools venv, repo root): python tools/track3/score_rollout.py <run_dir> <episode> [scene_variant, default fs_track_attach]
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_local as S  # noqa: E402

LR = Path(__file__).resolve().parents[2] / 'baseline/reconstruction/data/outputs/load_retarget'


def main():
    run, ep = Path(sys.argv[1]), int(sys.argv[2])
    e = next(x for x in S.public_episodes() if x['episode'] == ep)
    table = pd.read_parquet(run / 'results' / f'episode_{ep:06d}.parquet')
    span = int(table['reference_frame'].max()) + 1
    print(f'reference_frame span {span}, video frames {e["num_frames"]}')
    if span > e['num_frames']:  # stretched: keep even rows
        table = table[table['reference_frame'] % 2 == 0].copy()
        table['reference_frame'] //= 2
        last = e['num_frames'] - 1  # the stretched clip ends one row short: repeat its final pose
        if table['reference_frame'].max() < last:
            tail = table[table['reference_frame'] == table['reference_frame'].max()].copy()
            tail['reference_frame'] = last
            table = pd.concat([table, tail])
    with tempfile.TemporaryDirectory() as tmp:
        table.to_parquet(Path(tmp) / f'episode_{ep:06d}.parquet')
        poses = S.load_predictions(Path(tmp), [e])
    variant = sys.argv[3] if len(sys.argv) > 3 else 'fs_track_attach'
    meshes = {(ep, o['name']): next((LR / f'episode_{ep:06d}' / variant / 'score_input' / 'meshes')
                                    .rglob(f"{o['name']}*.obj")) for o in e['objects']}
    print({k: round(v, 3) for k, v in S.score([e], poses, meshes).items()})


if __name__ == '__main__':
    main()
