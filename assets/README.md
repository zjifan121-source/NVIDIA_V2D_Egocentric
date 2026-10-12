# assets/ — everything about the challenge

Official material only, copied unmodified. No team work lives here.

```
assets/
├── docs/
│   ├── nvidia/                 NVIDIA's docs/ from video_to_data release/0.2.0
│   └── challenge/              official challenge website (rules, FAQ PDF, submission kit zip)
└── data/
    ├── v2d_submission_kit/     official evaluator: scorers, policy evaluator, packer, uploader
    └── track_3/                [local only, gitignored] Hugging Face dataset: inputs + evaluation split
```

Model weights and run outputs are **not** here. The baseline expects them under
`baseline/reconstruction/data/` (`weights/`, `outputs/<branch>/<episode>/`), which is also gitignored.

## Sources and versions

| Folder | Source | Version |
|---|---|---|
| `docs/nvidia/` | `docs/` of [nvidia-isaac/video_to_data](https://github.com/nvidia-isaac/video_to_data) | `release/0.2.0` (`43a3f5b`) |
| `docs/challenge/` | `docs/v2d_challenge/` of NVIDIA's repo, published at https://nvidia-isaac.github.io/video_to_data/v2d_challenge/ | NVIDIA `main` @ `193382a` (2026-10-08) |
| `data/v2d_submission_kit/` | `v2d_submission_kit.zip` from the challenge website, unzipped | updated by NVIDIA 2026-10-06 (`7806e31`) |
| `data/track_3/` | [`nvidia/video_to_data_challenge`](https://huggingface.co/datasets/nvidia/video_to_data_challenge/tree/main/track_3) | downloaded by each person |

To refresh the challenge site and kit from NVIDIA's current `main` (on your branch, then a PR into `main_buffer`):

```bash
git fetch upstream
rm -rf assets/docs/challenge assets/data/v2d_submission_kit
git archive upstream/main docs/v2d_challenge | tar -x -C /tmp && mv /tmp/docs/v2d_challenge assets/docs/challenge
unzip -q assets/docs/challenge/assets/v2d_submission_kit.zip -d assets/data/
```

## Track 3 dataset

CC-BY-4.0, ~1.5 GB, 169 files. Download from the repo root:

```bash
hf download nvidia/video_to_data_challenge --repo-type dataset \
    --include "track_3/*" --local-dir assets/data/
```

`hf` comes with `pip install -U huggingface_hub` (older versions call it `huggingface-cli download`). The download also writes `assets/data/.cache/`, which is gitignored.

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
- The submission kit expects quaternions in **xyzw** order; the dataset stores them **w-first**.
