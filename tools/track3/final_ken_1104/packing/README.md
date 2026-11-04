# packing/: one-episode packing, swap into bob1, combine (final1104)

These scripts turn a selected rollout (the evaluator output of the chosen checkpoint) plus its scene meshes into Track 3 Kaggle
rows. They change only our episodes' rows and keep every other row byte-identical.

## What it does and why

- **Base file**: Edmond's `candidate_bob1.parquet`, the team baseline: Kaggle AUC 0.75166, 9,120 rows over 20 episodes,
  sha256 `46fe967a5aafc4e5078731a480e64105efdab9c48b466687c5632592a7f00fa4`. On 3 Nov the packed bob1 parquet was in neither git
  nor HF. With Ken's OK we downloaded Edmond's Kaggle submission 57934736 using an authenticated API script. That script is not
  included because it reads the Kaggle token.
- `pack_one.sh EP RESULTS_DIR MESH_DIR COMMIT_URL OUT` packs one episode the same way the team's full packer does:
  1. Narrows copies of the kit's `config/tracks.json` (`scored_episodes=[EP]`) and of `data/track_3_sample_submission.parquet` to that episode.
  2. Relabels the rollout's `object_body_name` to the roster names with `relabel()` from the team wrapper `tools/track3/pack_submission.py`.
  3. Copies the scene's `score_input/meshes/episode_XXXXXX/*`.
  4. Runs the kit's `tools/pack_submission.py --track track_3`.

  Why: the swap keeps the other 19 episodes' rows untouched, so a Kaggle change can be attributed to this one episode
  (episode change ~ 20 x dAUC).
- `swap_episode.py BASE ONE EP OUT [--adopt-commit-url]` splices the one-episode rows into a full file. It asserts:
  - identical columns and the same row grid;
  - **uniform provenance**: `eval_script_sha256`, `eval_code_sha256`, `packer_script_sha256`, `uploader_script_sha256` and
    `code_commit_url` must equal the base's, because Kaggle rejects mixed values. `--adopt-commit-url` copies the base's URL;
  - all other rows are unchanged.
- `swap_submit.sh EP NAME RES MESH MSG [SUBMIT]` runs `pack_one.sh` and `swap_episode.py` into bob1, then the kit dry-run. With
  `SUBMIT=1` it also submits; `ONLY=add_auc` limits the run to the AUC board.
  - Edit history from the session logs (EST):
    - created 3 Nov 22:36;
    - quota guard in the SUBMIT block 4 Nov 00:18, refusing to submit at <= 5 left, raised to <= 8 at 00:19;
    - `ONLY` option 00:35, which adds `--only` to the dry-run and submit lines;
    - guard lowered to <= 3 at 01:38 (this is the staged file).
  - None of these edits touched the `pack_one.sh` or `swap_episode.py` lines that produce `one.parquet`.
  - T5 (01:29) ran with the 00:35 version (guard 8). T8 (02:48) and ep10 NEST (05:02) ran with the staged version.
- `ship.sh NAME SCENE EP STEP` (local PC) copies an orchard run's `early_<STEP>/results` rollout and its scene's
  `score_input/meshes` to Babel `~/v2d_ops/kenc/tests/NAME/in/`.
- `combine.sh OUT EP:CAND ...` chains `swap_episode.py` over bob1 with `tests/CAND/one.parquet`, then runs the dry-run. This is the
  Babel version (4 Nov 06:25, written by `_combo_run.sh`), which built:
  - `combos/base4`: bob1 + ep1 T2 `o_ep1_hold_s44` + ep3 T5 `h_ep3_sc_K_s42` + ep5 T8 `r_ep5_rl_K_s43` + ep8 T6 `h_ep8_sc_K_s42`.
    This is the same combination as the team's best AUC entry, 0.76526 (Wentao submitted that one as a row-level merge; we did not
    compare the bytes).
  - `combos/base4_n10`: base4 + ep10 NEST `n_ep10_nb_K_s42`, submitted on MPPE only as N1: 3.42999 -> **3.33289**.
- `final1104/`: the final local-best build (4 Nov 13:47-13:51 EST on babel-s5-16; **not submitted**, because Ken stopped all
  submissions at 10:05).
  - `pack_new.sh` packs ep1 `b_ep1_wh_K_s48@100003840` (scene `w_hold_ep1`) and ep8 `rf_ep8_ds_K_s43@95006720` (scene `rf_ep8_ds`)
    with `pack_one.sh`.
  - `run_combine2.sh` writes `combine_final.sh` and runs it.
  - `dryrun_final.sh` and `dry2_archive.sh` validate the file and stage the ep1/ep8 archives.

## Team-repo code this depends on (not copied)

- The kit `assets/data/v2d_submission_kit` at 1a6bae7b:
  - `tools/pack_submission.py`: sha256 `49b13052...`, which is the rows' `packer_script_sha256`.
  - `submit/v2d_submit.py`: sha256 `d31d4a7c...`, which is the rows' `uploader_script_sha256`. Used for `--dry-run`, `--only`, the
    CD-O extraction and submission.
  - `config/tracks.json`, `data/track_3_sample_submission.parquet`, `data/track_3_evaluation_objects.json`.
- The team wrapper `tools/track3/pack_submission.py`. `pack_one.sh` loads it from `/home/haisenk/v2d_ops/team_pack_submission.py`,
  a byte-identical copy (git blob `8276045d`, the same at 1a6bae7b, 1689cf63 and de66fea3; md5 also equals Babel's repo file). To
  re-run, point that path at the repo file.
- The rollouts come from the kit evaluator through `../training/kq.sh`. The meshes are the scene's `score_input/meshes`, written by
  the team loader `tools/track3/run_load_retarget.py` and edited by our scene fixes; see the episode parts.

## Exact runs that produced the final rows

All per-episode files cite `code_commit_url = https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/commit/1689cf6396bee0f6a3bf4cced610f2fe0f71ebf6`.
That is bob1's URL, passed explicitly and kept uniform with the base. **Note for the code release:** commit 1689cf63 (Edmond, "Track
reports/ code, notes and small results") does not contain these tools or our scene-fix code. The code is in this
`final_ken_1104/` release.

| ep | one.parquet (Babel `~/v2d_ops/kenc/...`) | rows | sha256 | checkpoint_sha256 | produced by |
|---|---|---|---|---|---|
| 1 | `final1104/b_ep1_wh_K_s48/one.parquet` | 670 | `7150f4e5...` | `6aaab5d6...` | `final1104/pack_new.sh` (13:47) |
| 3 | `tests/h_ep3_sc_K_s42/one.parquet` | 626 | `4a3d974d...` | `45c4736b...` | `swap_submit.sh` (T5, 01:29, submitted AUC-only) |
| 5 | `tests/r_ep5_rl_K_s43/one.parquet` | 576 | `11183810...` | `d351dbae...` | `swap_submit.sh` (T8, 02:48, submitted AUC-only) |
| 6 | bob1 rows unchanged (Edmond) | | | | none |
| 8 | `final1104/rf_ep8_ds_K_s43/one.parquet` | 438 | `6da1e7bf...` | `3d33ff6b...` | `final1104/pack_new.sh` (13:47) |
| 10 | `tests/n_ep10_nb_K_s42/one.parquet` | 322 | `5b3eb640...` | `2bd79300...` | `swap_submit.sh` (05:02, dry-run only) |

Commands as run. `ship.sh` runs on the PC and `swap_submit.sh` runs in a CPU step of A6000 holder 11035728. The message strings are
shortened here; each one gave the candidate and scene description:

```bash
bash ship.sh h_ep3_sc_K_s42 sc_ep3 3 95006720
ssh babel "srun --jobid=11035728 --overlap -n1 -c2 --mem=8G bash -c 'ONLY=add_auc bash ~/v2d_ops/swap_submit.sh 3 h_ep3_sc_K_s42 \
  /home/haisenk/v2d_ops/kenc/tests/h_ep3_sc_K_s42/in/runs/h_ep3_sc_K_s42/early_95006720/results \
  /home/haisenk/v2d_ops/kenc/tests/h_ep3_sc_K_s42/in/scenes/sc_ep3/score_input/meshes \"Ken swap test T5 (AUC only): ...\" 1'"
bash ship.sh r_ep5_rl_K_s43 rl2_ep5 5 95006720
#   then the same with: 5 r_ep5_rl_K_s43 .../early_95006720/results .../scenes/rl2_ep5/score_input/meshes "Ken swap test T8 (AUC only): ..." 1
bash ship.sh n_ep10_nb_K_s42 k10_ep10_nest_b 10 100003840
#   then the same with: 10 n_ep10_nb_K_s42 .../early_100003840/results .../scenes/k10_ep10_nest_b/score_input/meshes "bob1 + ep10 <- ..." 0
# final local best (every script below ran as: ssh babel "srun --jobid=10774619 --overlap -n1 -c4 --mem=16G bash -s" < X > X.log):
#   final1104/pack_new.sh -> final1104/run_combine2.sh -> final1104/dryrun_final.sh -> final1104/dry2_archive.sh
bash ~/v2d_ops/kenc/final1104/combine_final.sh local_best 1:$F/b_ep1_wh_K_s48/one.parquet 3:$TS/h_ep3_sc_K_s42/one.parquet \
  5:$TS/r_ep5_rl_K_s43/one.parquet 8:$F/rf_ep8_ds_K_s43/one.parquet 10:$TS/n_ep10_nb_K_s42/one.parquet
#   (F=~/v2d_ops/kenc/final1104, TS=~/v2d_ops/kenc/tests)
```

Result. Records are in `final1104/records/`: `manifest.txt` lists the inputs and their sha256, `swap.log` the per-step row
replacements, plus `sha256.txt`.

- `~/v2d_ops/kenc/final1104/local_best/track3_submission.parquet`: 9,120 rows / 20 episodes, sha256
  **`5d1d534f4b3e6ba5dcd165203aebc001faa4f2823c78238b83e9df287e529edd`**. Its kit-derived CD-O file is `bcf967d1...`,
  byte-identical to base4_n10's.
- Checked against base4_n10: only the ep1 and ep8 rows differ, and each equals its `one.parquet` except the commit URL. Every
  uniform column has one value.
- Kit dry-run on all 6 boards: the file "looks valid", but preflight exits 1 on the kit's Kaggle identity check, because the last
  submitter differs between boards (not a file problem). Split dry-runs (`--only add_auc spider_sr maniptrans_sr rpe_cm` and
  `--only mppe_cm cd_o_cm`) both return "looks valid ... nothing submitted" (logs in `records/`).

Kaggle evidence, from `C:\v2d_demo\tests\ledger.md`; each swap is bob1 with one episode replaced:

- T2 ep1 0.75413 (+0.049 on ep1)
- T5 ep3 0.75305 (+0.028)
- T6 ep8 0.75505 (+0.068)
- T8 ep5 0.75799 (+0.127)
- bob1 + T2 + T5 + T8 + T6 = **0.76526**: the final AUC, team best on all 6 boards at 06:25 EST
- N1 (base4_n10) MPPE 3.33289

The final1104 file itself was never submitted. Its ep1 and ep8 rows replace T2 and T6 based on local GT-free judges only.

## Files

| file | md5 | exactness evidence |
|---|---|---|
| `pack_one.sh` | `16f82b5c6ff8970a69cfc2a7272f0979` | = Babel `~/v2d_ops/pack_one.sh` (3 Nov 21:54), unchanged since before T1 |
| `swap_episode.py` | `b1b1656483bc8920a4967ea62615eaf9` | = Babel `~/v2d_ops/swap_episode.py` (3 Nov 21:55) |
| `swap_submit.sh` | `816dc28d8622bbe293fc72295589449b` | = Babel `~/v2d_ops/swap_submit.sh` (4 Nov 01:38); edit history above |
| `ship.sh` | `f10dac271bfc357315be059fd831fdb1` | local-only, last edited 3 Nov 22:42, before every use |
| `combine.sh` | `28b0f8d361b1f4f89025a64c277109c2` | = Babel `~/v2d_ops/kenc/combine.sh` (4 Nov 06:25), i.e. the heredoc in `_combo_run.sh` (both extractions give this md5). The older local `C:\v2d_demo\tests\combine.sh` (3 Nov 23:14, md5 `03f552a1`, no overwrite guard) is not staged; nothing it built remains. |
| `_combo_run.sh` | `832c11f150291004bfa2e05e281a90d2` | the runner that wrote `combine.sh` and built base4 / base4_n10 (06:25) |
| `final1104/combine_final.sh` | `61addd4f7d8d6a296303ae58e1fa50a9` | = Babel `~/v2d_ops/kenc/final1104/combine_final.sh` (13:48:57) |
| `final1104/pack_new.sh`, `run_combine2.sh`, `dryrun_final.sh`, `dry2_archive.sh` | `84a370b2...`, `d3e103d3...`, `9d74dcc6...`, `dc527e56...` | Local copies, sent through `ssh ... bash -s` (never stored on Babel). The first `run_combine.sh` attempt had CRLF line endings; it left stray CR-named entries, which `run_combine2.sh` removes. It is not staged. |

## Hard-coded paths and side effects (kept as they ran)

- `/home/haisenk/v2d_ops/kaggle_env/bin/python`: a venv with kaggle, pandas and pyarrow; no credentials in any of these files.
- `/data/user_data/haisenk/v2d/v2d_train/assets/data/v2d_submission_kit`: the kit.
- `/home/haisenk/v2d_ops/team_pack_submission.py`, `$HOME/v2d_ops/kenc/{tests,combos,final1104,bob1}`, `/tmp/haisenk_v2d`
  (final1104), and `ssh orchard` with job 157612 (`ship.sh`).
- Side effects:
  - `pack_one.sh` leaves its work dir `/tmp/pack_one_XXXX`.
  - `swap_submit.sh` removes its own previous `one.parquet` and `track3_submission.parquet` in `tests/NAME`.
  - `ship.sh` replaces `tests/NAME/in`.
  - `run_combine2.sh` deletes the CRLF strays in `final1104/local_best`.
  - Real submission needs the kit's Kaggle credentials, which are not part of this release.
