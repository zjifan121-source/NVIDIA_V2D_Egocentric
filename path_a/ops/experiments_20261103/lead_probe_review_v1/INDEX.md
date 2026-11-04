# lead_probe_review_v1 index

Tools that produce or verify the final Track 3 file. Hashes are the first 16 hex of SHA-256, checked against the deployed copies on 2026-11-04 at 08:47 UTC.

| File | sha256 | Purpose |
|---|---|---|
| `merge_swaps.py` | `646dc19fb9718291` | Row-level merge of packed files from the raw common base; exact checks, refuses stray differences and overwrites. Produces the final file. |
| `board_watch.py` | `9df01766412bb5f2` | Reads the six boards, classifies every team file against bob1, and builds INCUMBENTS.md: rank-sum selection, per-board matrix, run dirs. |
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

`SCENE_SCRIPT_DEPENDENCIES.md` lists which scene-building scripts each scene type depends on. Scripts from the A100 and A6000 are in `../lead_handspin_v1/`, with their own `SCRIPTS_INDEX.md`, including `evaluate_all70m.py`.

These files hold no tokens or network addresses; paths under `/root` and `/mnt` are host-local. `submit_list.sh` relies on the Kaggle login already on the host.
