# assets/ — shared inputs

The inputs both paths use. Code lives in `baseline/` (NVIDIA + Path A) and `path_b/` (Path B).

```
assets/
└── data/        [local only, gitignored] Track 3 dataset and other large inputs
```

Model weights and run outputs are **not** here. The baseline expects them under
`baseline/reconstruction/data/` (`weights/`, `outputs/<branch>/<episode>/`), which is also gitignored.

## Track 3 dataset

Source: [`nvidia/video_to_data_challenge`](https://huggingface.co/datasets/nvidia/video_to_data_challenge/tree/main/track_3) → `track_3/` (CC-BY-4.0, ~1.5 GB, 196 files).

Download from the repo root:

```bash
huggingface-cli download nvidia/video_to_data_challenge --repo-type dataset \
    --include "track_3/*" --local-dir assets/data/
```

Resulting layout:

```
assets/data/track_3/
├── README.md, v2d-challenge-track3.md     dataset card (read once)
├── public/                                20 episodes, scorable locally
│   ├── data/chunk-000/episode_XXXXXX.parquet
│   ├── videos/chunk-000/observation.images.ego_cam_{a,b,c}/episode_XXXXXX.mp4
│   ├── mesh/<object>/<object>.glb         10 objects (see note below)
│   ├── urdf/<object>_rigid.urdf
│   └── meta/info.json, episodes_metadata.jsonl, camera_calibration.json
└── evaluation/                            20 episodes, videos only (ground truth hidden)
    └── videos/chunk-000/observation.images.ego_cam_{a,b,c}/episode_XXXXXX.mp4
```

Notes:

- **Episode indices are not contiguous.** Public: 0, 2, 11, 12, 13, 18, 21, 22, 23, 24, 26, 27, 31, 39–45. Evaluation episodes use the remaining indices.
- **Three objects have split meshes.** `mini_sweeper`, `wooden_piece_1` and `wooden_piece_2` ship `<object>_visual.glb` + `<object>_collision.glb` instead of `<object>.glb`. Use `_visual.glb` for tracking.
- **`ego_cam_a`** (2028×1520, colour) is the camera to feed the baseline. Its calibration has `rotation_deg: 180`, so check frames are upright first (README §5.6).
- There is no `meta/` under `evaluation/`. Use `public/meta/camera_calibration.json` for both splits (same rig).
