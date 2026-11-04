# cdo_transfer/: CD-O-only mesh row transfers (Kaggle entries C1 and C2)

These two entries went to the **CD-O board only** (`v2d_submit.py --track track_3_mesh`). They change mesh rows; no pose rows and
no policies are involved.

## What it changes and why (confirmed diagnosis)

- CD-O is shape-only. The kit's `metric_code/track_3/CD-O.py` works per object:
  1. scale our mesh to the hidden GT scan's RMS surface radius;
  2. align it with 25 PCA starts and trimmed ICP;
  3. take the symmetric chamfer distance in cm.

  The board is the mean over the 36 scored objects. Meshes are budgeted to 4,096 vertex rows and 4,096 face rows, with
  `row_id = t3m_<ep 6 digits>_<object>_<0=V|1=F>_<i>`.
- The same physical object appears in several scored episodes and is scored against **one** scan. For example, white_pot is in
  episodes 1, 3, 5, 6, 8 and 28, and beige_cup in 8 and 10. So a mesh change whose per-object effect Kaggle has measured moves the
  board by the same amount in every episode that has that object.
- C1 confirmed this transfer: predicted 0.3820, scored 0.38183. CD-O-only files therefore carry meshes that differ from the
  pose-board files: ep10 beige_cup, and white_pot in ep1, 3, 6 and 8. **Never copy them into pose-board files**, because the frames
  differ.
- **These choices were selected on the leaderboard.** The per-object deltas were measured on Kaggle, not GT-free. The technical
  report should say so.

| entry | Kaggle submission | file (Babel `~/v2d_ops/kenc/cdo/`) | change | CD-O |
|---|---|---|---|---|
| base | 57979229 (team's best CD-O entry at the time: "current base + ep33 probe") | `w33/57979229.raw` sha256 `c08d4cbc...` | none | 0.39161 |
| **C1** (4 Nov ~07:10 EST) | **57983875** | `final/cdo_w33_ep10beigeT6.parquet` sha256 `bb2b98096433f4516c7ce1c52fd837519b9515e9c0843c2a48b3aa162ff775cb` (294,912 rows) | ep10 `beige_cup` rows <- the ep8 beige_cup of T6 (`sc_ep8` V2: cup mesh +2.8 cm down to the depth-measured table, rim fixed; the same arrays as in ep8) | **0.38183** (rank #22 -> #19) |
| **C2** (4 Nov ~07:25 EST) | **57984124** | `final/cdo_c2_pots.parquet` sha256 `a86e4f38e7f4c7e6ac9c8409906decd1bcb371336e7078f024f5ef0f9f04b4d4` | C1 + `white_pot` V and F arrays in ep1, 3, 6 and 8 <- the ep5 white_pot of T8 (`rl2_ep5`: Wentao v2 pot, -1.0 cm rim-fixed) | **0.36781** (rank #19 -> #18) |

Per-object deltas from the ledger:
- the T6 cup vs bob1's cup: -0.345 cm on beige_cup;
- the T8 pot vs the donor pot used in ep3, 6, 8 and 28 and in bob1's ep1 and ep5: -0.124 cm;
- Wentao's v2 ep5 spoon vs bob1's ep5 spoon: -0.051 cm (analysis only, not transferred).

After C2 we stopped at Ken's quota line (3 CD-O submissions left). Transfers we did not test: the ep28 pot, the v2 spoon in other
spoon episodes, and blue_cup.

## Scripts, in the order they ran (4 Nov, Babel compute steps, sent through `ssh babel "srun ... bash -s" < script`)

| step | script | what it does | md5 |
|---|---|---|---|
| 0 | `_cdo_build.sh` (**not included**) | Downloads the base entry 57979229 from Kaggle with `KaggleApi().authenticate()` into `w33/57979229.raw`. Left out because it reads the Kaggle token (~/.kaggle). The step only fetches the team's own earlier submission. | n/a |
| 1 | `_cdo_build2.sh` | Packs the ep10 NEST rollout (`n_ep10_nb_K_s42@100M`) with `k10_ep10_nest_b`'s meshes, after replacing `beige_cup.obj` with T6's ep8 beige_cup (`../packing/pack_one.sh`). Splices that into `combos/base4` (`../packing/swap_episode.py`). The kit dry-run (`--only cd_o_cm`) then writes the derived CD-O rows `m10/full.cd-o.parquet` (sha256 `0b499e99...`). | `8b273864...` |
| 2 | `_cdo_cmp.sh` | Compares w33 and ours: same row set; which rows differ; ep8 and ep10 beige arrays. | `6ad0f7ad...` |
| 3 | `_cdo_final.sh` | **C1** = w33 with only the `t3m_000010_beige_cup_*` rows taken from `m10/full.cd-o.parquet`. Keeps w33's single commit URL. Runs the dry-run. | `23a7eb4a...` |
| 4 | `_cdo_submit.sh` | Dry-run, then quota guard (stop at <= 3 left), then submits C1 to `track_3_mesh` -> 57983875. | `d24d6c22...` |
| 5 | `_cdo_poll.sh` | Polls the C1 receipt. | `37766427...` |
| 6 | `_cdo_pairs.sh` | Derives the CD-O rows of bob1 and T3 with the kit. Computes pairwise CD between the white_pot and wooden_spoon variants with the kit's own `_object_chamfer_cm`. | `9ef4070a...` |
| 7 | `_cdo_pots.sh` | Checks which meshes are identical across w33, bob1 and T3: the ep1 lid, the ep33 blue cup, and the ep3, ep5 and ep6 spoons. Also prints the extents of the white_pot variants along their PCA axes. | `468efdaf...` |
| 8 | `_cdo_c2.sh` | **C2** = C1 with the ep1/3/6/8 white_pot V and F arrays set to ep5's arrays. Asserts they are equal, runs the dry-run and the guard, submits -> 57984124, then polls. | `6e2ad20e...` |
| (local) | `cdo_pairs.py A.obj B.obj ...` | The same pairwise CD-O distance on local .obj meshes, using the kit's `CD-O.py` functions. Used on the PC before the Kaggle tests. | `986b74ea...` |

All files are the local copies that were sent; they were never stored on Babel. Their mtimes (07:01-07:17 EST) line up with the
Babel outputs: `m10/` 07:05, `final/cdo_w33_ep10beigeT6.parquet` 07:06, its receipt 07:09, `cdo_c2_pots.parquet` 07:17, its
receipt 07:20. Babel logs: `final/dryrun.log` and `submit.log` (C1, "ACCEPTED ... submission 57983875"); `final/dry_c2.log` and
`submit_c2.log` (C2, "ACCEPTED ... submission 57984124").

## Team-repo code this depends on

- The kit `submit/v2d_submit.py`: `--track track_3_mesh`, `--only cd_o_cm`, `--dry-run`, `--poll-receipt`, and the CD-O
  extraction from a pose file.
- The kit `metric_code/track_3/CD-O.py` (sha256 `c166cf86...` on Babel): `_object_chamfer_cm`, `_surface_moments`.
- `../packing/pack_one.sh`, `../packing/swap_episode.py`, and `../packing/combine.sh` (for `combos/base4`).

## Hard-coded paths (kept as they ran)

- `$HOME/v2d_ops/kenc/cdo/{w33,m10,ref,final}`, `/home/haisenk/v2d_ops/kaggle_env/bin/python`,
  `/data/user_data/haisenk/v2d/v2d_train/assets/data/v2d_submission_kit`, and the receipt paths under
  `/home/haisenk/v2d_ops/kenc/cdo/final/`.
- `cdo_pairs.py` loads the kit from a local Windows copy, `C:\v2d_demo\diag\ep10_verify\scorer_2\kit\metric_code\track_3\CD-O.py`.
- `_cdo_build2.sh` deletes its own earlier `m10/{one,full,full.cd-o}.parquet` before rebuilding.
- Submitting needs the kit's Kaggle credentials, which are not part of this release.
