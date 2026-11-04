# Training run r_ep5_rl_K_s43 (the T8 policy)

## Driver and protocol

The run used `kq.sh`, staged by the shared part as `../../training/kq.sh`. Its sha256 is
`9360ba9b705d71b9c857ef633ab563cc07db1b75cce43364a40666054ad7b65c`, the same on orchard, on babel-s5-16
`/tmp/haisenk_v2d/kq.sh` and in the local copy `C:\v2d_demo\orchard\kq.sh`.

`kq.sh` runs Edmond's training protocol (TRAINING_GUIDE.md / campaign.py @ de66fea3) without changes:

- flash_chord source de66fea3 via `PYTHONPATH=/tmp/haisenk_v2d/fc_src_de66`;
- recipe KEN;
- 100,003,840 environment steps;
- curriculum `tracking_100m_object2x`;
- the kit evaluator on every checkpoint from 70M on.

The run was on orchard flame-21, GPU 6, from 02:05:50 to 02:44 EST on 4 Nov 2026. It took 38 minutes with 0 nefc
overflow lines.

## Queue line

This line was appended to `/tmp/haisenk_v2d/kq.txt` at 01:49:48 EST; `NAME EP PARQUET SEED RECIPE PRIO`:

```
r_ep5_rl_K_s43 5 /tmp/haisenk_v2d/scenes/rl2_ep5/assets/human_motion_data/ego_recon/processed/sequence_id=ep5_hw_w7_attcom_ep5/robot_name=sharpa_wave 43 KEN 2
```

Sister seeds on the same scene were `r_ep5_rl_K_s42` and `r_ep5_rl_K_s44`. Later, on babel-s5-16, seeds s45 and s46
also ran as `b_ep5_rl_K_s45/s46`.

## Training command that kq.sh built for this line

`$FC` is `/tmp/haisenk_v2d/v2d_train/baseline/robotic_grounding/flash_chord`. `$FC_PY` is `$FC/.venv/bin/python`.
The repo checkout `v2d_train` is commit 1a6bae7.

```
cd $FC && PYTHONPATH=/tmp/haisenk_v2d/fc_src_de66 CUDA_VISIBLE_DEVICES=6 $FC_PY scripts/train_flash_sac.py experiment=sharpa_flash_sac \
  "task.parquet='/tmp/haisenk_v2d/scenes/rl2_ep5/assets/human_motion_data/ego_recon/processed/sequence_id=ep5_hw_w7_attcom_ep5/robot_name=sharpa_wave'" \
  reset.seed=43 curriculum=tracking_100m_object2x training.checkpoint.save_replay=false logging.mode=offline \
  logging.upload_checkpoints=false logging.run_name=r_ep5_rl_K_s43 output_dir=/tmp/haisenk_v2d/runs/r_ep5_rl_K_s43/train \
  training.total_environment_steps=100003840 \
  task.motion_speed=1.0 sim.fps=20.254119873046875 termination.wrist_position.threshold=0.3 \
  training.checkpoint.save_interval_environment_steps=5000192 termination.object_position.threshold=0.1 \
  termination.object_orientation.threshold=0.5 training.replay.capacity=4194304 \
  training.replay.observation_storage_dtype=float32 sim.njmax=1024 sim.nconmax=512
```

`recipe.txt` holds the KEN override string, recorded by `kq.sh`.

## Checkpoint evaluation

For each checkpoint S of 70M or more, `kq.sh` runs the kit evaluator:

```
$FC_PY $KIT/examples/run_policy_evaluation.py --checkpoints runs/r_ep5_rl_K_s43/early_S/checkpoints.json \
  --results runs/r_ep5_rl_K_s43/early_S/results --runtime-root $FC
```

- `$KIT` is `/tmp/haisenk_v2d/v2d_train/assets/data/v2d_submission_kit`.
- `checkpoints.json` for 95M was `{"5": "/tmp/haisenk_v2d/runs/r_ep5_rl_K_s43/train/policy_95006720.safetensors"}`.
- The Hydra overrides are in `eval_overrides.yaml` and `eval_config.yaml`.
- The 95M result is `eval_episode_000005.json`: add_auc 0.90885 [pot 0.91949, spoon 0.89822], chord_sr 1.0,
  maniptrans_sr 1.0, spider_sr_uncentered 1.0, mppe_cm 2.056, 288 steps, no early termination.

## kq.sh driver log (kq.out)

```
2026-11-04 02:05:50 gpu6 start r_ep5_rl_K_s43 (ep5 seed 43 KEN)
  r_ep5_rl_K_s43 train exit=0 minutes=38 nefc_overflow_lines=0
  sweep r_ep5_rl_K_s43: best 0.910 @ 90005504 [0.925, 0.896] (last 0.907 @ 100003840, 7 ckpts)
2026-11-04 02:46:30 gpu6 done r_ep5_rl_K_s43
```

## Selection

The kq.sh sweep (`sweep.json`, `best.txt`) ranks checkpoints by plain own-reference add_auc. The best by that measure
was 90M at 0.9104.

The checkpoint actually used is **95M (95,006,720 steps)**. It has the best *anchored* own-reference score: 0.9093,
against 0.9012 for 90M. The anchored score aligns the whole rollout on the pot's frame-0 pose, as the Kaggle scorer
does. See `../selection/`.
