# Packing ep5 (T8): commands as run on 4 Nov 2026, 02:48:46 EST

The ep5 rows were packed once, here, for swap test T8. The final combined file reused this exact `one.parquet`
(`~/v2d_ops/kenc/final1104/combine_final.sh`; see the ledger, 4 Nov ~13:55).

## 1. Ship the rollout and score meshes to Babel

This ran in local Git Bash. `C:\v2d_demo\tests\ship.sh` has sha256 `82eee05b…`.

```
bash /c/v2d_demo/tests/ship.sh r_ep5_rl_K_s43 rl2_ep5 5 95006720
```

It copies these files into `~/v2d_ops/kenc/tests/r_ep5_rl_K_s43/in/`:

- orchard `runs/r_ep5_rl_K_s43/early_95006720/results/episode_000005.{parquet,json}`, plus `best.txt` and `sweep.json`;
- `scenes/rl2_ep5/score_input/meshes`.

## 2. Pack, splice into bob1, dry-run, then submit to the AUC board only

This ran on Babel in a CPU step of job 11035728. `~/v2d_ops/swap_submit.sh` has sha256 `a7d9fb9d…`, the same as
`C:\v2d_demo\tests\swap_submit.sh`.

```
ONLY=add_auc bash ~/v2d_ops/swap_submit.sh 5 r_ep5_rl_K_s43 \
  /home/haisenk/v2d_ops/kenc/tests/r_ep5_rl_K_s43/in/runs/r_ep5_rl_K_s43/early_95006720/results \
  /home/haisenk/v2d_ops/kenc/tests/r_ep5_rl_K_s43/in/scenes/rl2_ep5/score_input/meshes \
  "Ken swap test T8 (AUC only): bob1 + ep5 <- r_ep5_rl_K_s43@95M (...)" 1
```

`swap_submit.sh` calls the following. The shared part stages these under `../../packing/`.

- `~/v2d_ops/pack_one.sh 5 RES MESH URL one.parquet`
  - sha256 `a895130f…`
  - URL is `https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/commit/1689cf6396bee0f6a3bf4cced610f2fe0f71ebf6`.
  - It runs the team relabel and the kit packer.
- `~/v2d_ops/swap_episode.py ~/v2d_ops/kenc/bob1/candidate_bob1.parquet one.parquet 5 track3_submission.parquet --adopt-commit-url`
  - sha256 `d7d93ab4…`
  - It replaces bob1's 576 ep5 rows. The other 8,544 rows stay byte-identical.
- The kit's `submit/v2d_submit.py --track track_3 --only add_auc --dry-run`, then the real submit.

## Outputs

| file | sha256 | notes |
|---|---|---|
| `one.parquet` | `11183810af8d275adac7e6575d38f6d4c6ead952e5b9c392d35ec630b8068066` | 576 rows, 178,101 B |
| `track3_submission.parquet` | `562a4a492dc7289c2e3d4dab8639103e05b9de30c483531daef3440035610e45` | 9,120 rows, 2,683,866 B |

- Kaggle AUC board submission 57976828, public score **0.75799**. See `kaggle_receipt_T8.json` and `submit.log`.
- `pack.log` holds the packer's provenance hashes:
  - checkpoint_sha256 `d351dbae…`
  - eval_script_sha256 `160d24f5…`
  - eval_code_sha256 `b1b3ebe3…`
  - packer_script_sha256 `49b13052…`
  - uploader_script_sha256 `d31d4a7c…`
  - code_commit_url = bob1's `1689cf63…`. That commit does **not** contain the ep5 scene code staged here.
