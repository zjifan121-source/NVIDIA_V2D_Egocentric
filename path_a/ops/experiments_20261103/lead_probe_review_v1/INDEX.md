# lead_probe_review_v1 index

Tools that produce or verify the final Track 3 file. Hashes are the first 16 hex of SHA-256. The watcher was checked against its deployed copy on 2026-11-04 at 10:19 UTC, the rest at 08:47 UTC unless added later.

| File | sha256 | Purpose |
|---|---|---|
| `merge_swaps.py` | `646dc19fb9718291` | Row-level merge of packed files from the raw common base; exact checks, refuses stray differences and overwrites. Produces the final file. |
| `board_watch.py` | `1999c0b7a2350751` | Reads the six boards, classifies every team file against bob1, and builds INCUMBENTS.md: rank-sum selection, per-board matrix, run dirs. |
| `board_watch_loop.sh` | `da3617147399bf8e` | Runs board_watch.py in a loop; stops at the freeze. |
| `check_probe_plan.py` | `94d34df287ec94b5` | Gate for pack plans: roster, results, meshes, scene-policy link, diff against bob1, expected score, commit URL format. |
| `scene_provenance.py` | `304f98b12b162568` | Lists the scripts and inputs behind a scene from its marker files; flags false grasps and hold release gaps. |
| `export_score_input.py` | `75170aa3473c0fca` | Adds score_input meshes and results to scenes built without the loader final step, for packing. |
| `submit_list.sh` | `0f58c71271d4bcdc` | Submits already-scored single-swap files to their missing boards, with duplicate protection. |
| `table_depth_check.py` | `b1282e9a3f041fe1` | Fits the table plane from depth and compares it with support tops and mesh bottoms. |
| `table_frame_check.py` | `970b9d1e4db7caf5` | Checks that depth and object poses share a frame by projecting our meshes. |
| `table_tilt_all.py` | `a50cb0dacc135de0` | Measures table tilt against world Z at frame 0 for every episode. |
| `mesh_size_audit.py` | `65b70e1d3164dc94` | Compares object mesh sizes across episodes by class. Information only. |
| `shape_audit.py` | `01531839ad55e117` | Compares mesh shapes with the kit CD-O function, no GT. Information only. |
| `mesh_swap_probe.py` | `a3092b3b3e581322` | CLOSED. Mesh-only swaps are barred by project rule; never submitted, kept as a record. |
| `rollout_proxy.py` | `bd6c1eb9c9edf3a9` | Scores a rollout against depth and masks, GT-free: constant mapping, two-sided fit for other body frames, or poses aligned by board_emulate.py. |
| `board_emulate.py` | `7b1ee98393a9d61b` | GT-free board stand-in: the kit pose metrics with a bundle or a team rollout as the reference. Screening only, see BOARD_EMULATOR.md. |
| `scan_compare.py` | `faaf96961ffed5c4` | Post-mortem: extents, area and CD-O of team meshes against public scans. Enters no selection. |
| `scan_floor.py` | `915aef1f0ee78a56` | Post-mortem: distance a decimated scan gets, the floor for scan_compare.py. |
| `decode_own.py` | `8235d12942d62a49` | Writes the meshes of a submission file as OBJ. |
| `score_candidates.py` | `6ee25c836f187d00` | The automatic judge: both GT-free checks on every new checkpoint from 70M, against the episode incumbent, with the rollout's own scene meshes. Writes scores_v2.jsonl. |
| `score_new.sh` | `58756562b9008004` | Relays A100 rollouts and meshes to the A6000, runs the judge there, then writes CHECKPOINT_LEDGER.md and proxy_scores.jsonl. |
| `score_loop.sh` | `23a561ee2a2ec883` | Repeats score_new.sh until a run first passes, a verdict is better, or 21:30 UTC. |
| `ledger_from_scores.py` | `3aba3e8e37d4ea1d` | Rewrites CHECKPOINT_LEDGER.md from the judge's rows. |
| `termination_audit.py` | `7726a19b65e2a132` | Reads what the termination limits did to a rollout: frames counted, crossings on the same two limits for every run, and own-ref recomputed from the parquet. |
| `term_fields.py` | `7b79af8db2e6c124` | Writes those fields for the judge, under the CPU venv. |
| `term_backfill.py` | `6d481c010b2e38c4` | One-off: added those fields to the rows scored before the judge carried them. |
| `pair_penetration.py` | `d789acff65259a0d` | Depth of one object inside another along a track: mesh surface, convex hull, and the simulator's decomposed pieces. |
| `cup_measure.py` | `ab312614c2befaaa` | Cup rim diameter and height from depth and masks. Observation only. |
| `cup_silhouette.py` | `a6ac6634a7a3d5e5` | Observed cup silhouette against our rendered mesh, pixel for pixel. Observation only. |
| `scene_lift_diff.py` | `16740974e07fac60` | Compares two built scenes around the first motion of an object, per hand. |
| `mask_audit.py` | `2f4eeea22d6aecf3` | Audits frame package masks against rendered reference objects, over all objects and frames. |
| `selection_rescore.py` | `4812cf56256992f0` | Re-scores past track selections with neutral gating and depth fit, to find flips. |
| `hold_gap.py` | `468abe822b8b85a7` | Wrist, palm and fingertip distance to the held object during each hold, from a scene's reference parquet. |

`SCENE_SCRIPT_DEPENDENCIES.md` lists which scene-building scripts each scene type depends on. Scripts from the A100 and A6000 are in `../lead_handspin_v1/`, with their own `SCRIPTS_INDEX.md`, including `evaluate_all70m.py`.

These files hold no tokens or network addresses; paths under `/root` and `/mnt` are host-local. `submit_list.sh` relies on the Kaggle login already on the host.
