# tools/: scene relocation and candidate archiving

## relocate_scene.py: make a copied scene point at its own files

- **What and why.** A scene from the team loader (`tools/track3/run_load_retarget.py`) stores absolute paths from the machine
  that built it: `raw_motion_file`, `object_mesh_paths` and `object_urdf_paths` in the processed parquet, plus mesh paths inside
  the `.urdf`, `.usda` and `.mtl` files. FlashCHORD loads those paths. Our scenes moved between Babel `/data`, teammates' builds,
  orchard `/tmp` and back to babel-s5-16 `/tmp`, so every install had to rewrite them.
- **How it works.** Each absolute path is matched to the longest trailing part of it that exists under the scene copy. It
  rewrites string and list-of-string columns of every `assets/**/*.parquet` (schema kept) and the path text in
  `assets/**/*.{urdf,usda,mtl}`. It keeps the original next to each changed file as `<file>.orig`, reports paths with no match and
  leaves them unchanged. `--dry-run` only reports.
- **Usage**: `python relocate_scene.py <scene dir> [--dry-run]`. It needs pyarrow; we ran it with the flash_chord venv python.
- **Used by** `../training/install_scene.sh` (orchard: k10_ep10_nest_b and others) and `../training/install_scene_b.sh` (Babel:
  rf_ep8_ds and other scenes installed on 4 Nov). It also ships with the HF package `experiments/haisenk/hand_fixes_wentao_v1_20261103/`.
- **Exactness.** md5 `d8319c83091b9ac11d982869d142e5f4` is identical in four places: the local `C:\v2d_demo\orchard\relocate_scene.py`,
  babel-s5-16 `/tmp/haisenk_v2d/orchard_home/v2d_orchard/relocate_scene.py` (the copy the installers call), `~/v2d_ops/relocate_scene.py`
  and `~/v2d_ops/kenc/hand5w/ship2/relocate_scene.py`.
- **Not the team's file.** This is not `tools/track3/relocate_scene.py` in the team repo (Edmond, commit 6770789d, blob `1cb82e16`),
  which replaces the builder's scene-root prefix in parquet string cells only. Ours matches by trailing path parts and also fixes
  the URDF, USDA and MTL text. Our blob is `8c3bf252`.

## archive.sh: archive a candidate before anything is re-run

- **What and why.** Ken's rule was "never re-run a recorded fix". For a candidate it stores the scene tarball, the policy at the
  chosen checkpoint, the evaluator results of that checkpoint, `sweep.json`, `best.txt`, `recipe.txt`, `train.log` and a
  `manifest.json` (own-ref, per body, policy sha256, rollout sha256, recipe, note). It writes three copies: the orchard node
  (`/tmp/haisenk_v2d/archive/CAND`), the local PC (`C:\v2d_demo\archive\CAND`), and Babel's packed files and Kaggle receipts
  (`~/v2d_ops/kenc/tests/CAND` -> `babel/`).
- **Usage** (local Git Bash): `bash archive.sh CAND SCENE EP STEP "NOTE"`. `STEP` is the checkpoint step, or `final` for the old
  driver's results.
- **Used for the finals**:
  - `archive.sh h_ep3_sc_K_s42 sc_ep3 3 95006720 ...` (4 Nov 01:30)
  - `archive.sh r_ep5_rl_K_s43 rl2_ep5 5 95006720 ...` (02:54)
  - `archive.sh n_ep10_nb_K_s42 k10_ep10_nest_b 10 100003840 ...` (05:03)

  Logs: `C:\v2d_demo\tests\archive_r_ep5.log` and `archive_n_ep10.log`. md5 `caa53903287e820e91a1e76fad69e49d`; the local file was
  last edited 3 Nov 23:04, before all three uses.
- **ep1 and ep8 finals.** These were archived on Babel after orchard was gone, by the `arch()` function in
  `../packing/final1104/dry2_archive.sh`. That function writes the same contents plus `scene_tgz_sha256`, the packed `one.parquet`
  sha256 and a `SHA256SUMS` file. The archives are copied to `C:\v2d_demo\archive\b_ep1_wh_K_s48\` and `...\rf_ep8_ds_K_s43\`.
- **Hard-coded paths and side effects**: `ssh orchard`, job 157612, `/tmp/haisenk_v2d/archive`, `/c/v2d_demo/archive`, and
  `ssh babel` reading `~/v2d_ops/kenc/tests/CAND`. Inside its own archive dir it replaces `results/` atomically: `rm -rf` of its own
  `results.tmp` and `results`, then `mv`. Orchard no longer exists, so the script cannot run as written.

Team-repo dependencies: none at run time, beyond the scene layout of the team loader and the run layout of `../training/kq.sh`.
