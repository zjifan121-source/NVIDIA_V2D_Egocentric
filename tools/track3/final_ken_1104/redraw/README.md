# `redraw/`: best-of-N rollout re-evaluation used for FINAL_B

The official FlashCHORD evaluator is not deterministic. Running it again on the same checkpoint and the same scene gives a
different recorded rollout. For the final file (FINAL_B) we therefore re-ran the selected checkpoint N times per episode. We then
submitted the recorded rollout with the highest own-reference `add_auc`, which is the evaluator's ADD-AUC against the episode's
own reconstructed reference. No hidden data was used.

| Ep | Checkpoint (run @ step) | Draw kept | N | Own-ref |
|---|---|---|---|---|
| 1 | `b_ep1_wh_K_s48` @100003840 | a10 | 64 | 0.8414 |
| 3 | `rf_ep3_f3w_K_s42` @100003840 | a7 | 48 | 0.9547 |
| 5 | `r_ep5_rl_K_s43` @95006720 | g5 | 56 | 0.9184 |
| 8 | `rf_ep8_ds_K_s43` @95006720 | a10 | 32 | 0.9946 |
| 10 | `n_ep10_nb_K_s42` @100003840 | e2 | 64 | 0.9852 |
| 15 | bob1 `ep15_v2_250M_RS` | d39 | 40 | 0.9563 |
| 16 | bob1 `ep16_WHCc_..._s101` | d22 | 40 | 0.9391 |
| 20 | bob1 `ep20_h5objrel_s43` | d04 | 36 | 0.9214 |
| 32 | bob1 `ep32_bm_s42_RS` | d23 | 40 | 0.9018 |
| 35 | bob1 `ep35_kenC_s42` | d34 | 72 | 0.9705 |
| 36 | bob1 `ep36_v2_250M` | d02 | 40 | 0.9254 |
| 37 | bob1 `ep37_v5_obj2x_s43` | d36 | 40 | 0.9921 |

Episode 6 was not re-drawn. Its submitted rollout is the checkpoint-sweep evaluation of `x_ep6_d2_K_s44` at step 85004288.

## Files

- **`reeval.sh`, `reeval2.sh`**: run the official evaluator again on one checkpoint and scene. Each run writes a draw folder
  (prefix `DPFX` plus a draw index) holding `results/episode_XXXXXX.json` and the rollout. `reeval2.sh` is the later version,
  used for the second round (series `a` to `g`).
- **`reeval_ours.txt`, `redraw3.txt`, `redraw4.txt`**: the queue lines (scene, run, step, draw ids) for our episodes.
- **`taskA/`**: the same procedure for the teammates' bob1 checkpoints.
  - `setup_ep.sh` installs a scene.
  - `patch_ckpt.py` rewrites only the scene path stored in a bob1 checkpoint's metadata (`user.resolved_config_json`). Tensor
    bytes are unchanged, but the checkpoint file's sha256 differs from bob1's.
  - `gen_jobs*.sh` and `launch_*.sh` enqueue the draws, and `drv*.sh` and `poll.sh` drive them.
  - `agg*.py` collect every draw's own-ref and pick the best.
  - `pack_best.sh` packs the chosen draw with the team packer.
- **`bon2_pack_logs/`**: `pack_one.sh` logs of the five one-episode files for our episodes in FINAL_B.

## Reproducing

Re-running a draw reproduces the procedure, not the exact recorded rollout, because the evaluator is not deterministic. The
recorded rollouts, draw summaries and checkpoints are archived on the team Hugging Face dataset under
`experiments/haisenk/final_ours_20261104/`.
