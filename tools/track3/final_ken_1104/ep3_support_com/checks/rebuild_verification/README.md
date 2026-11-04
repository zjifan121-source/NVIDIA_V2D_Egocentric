# Rebuild check of sc_ep3 (release staging, 4 Nov 2026 ~14:30 EST)

This check was added while staging the release. It was not part of producing the submission.

It shows that `scene_build/scenefix.py` and `scene_build/comfix.py`, run with the recorded arguments, turn the parent scene into the archived `sc_ep3`.

## How it was run

The check ran locally on Windows, using Python 3.12 with numpy 2.5.2, scipy 1.18.1, trimesh 5.1.1 and pyarrow 26.0.1.

1. **Parent scene.** The parent `b1_ep3` was rebuilt as follows:
   - Started from `sf_ep3` in `C:\v2d_demo\cupfix\sf_scenes.tgz`. That is the tarball of the original orchard `sf_ep3`, made at 00:14 EST on 4 Nov.
   - Replaced its support USD with the parent's own USD, fetched byte-exact from `babel-s5-16:/tmp/haisenk_v2d/scenes/b1_ep3` (md5 `0ee9f14e7dfbed38c55480f93a964a2c`).
   - Removed `scenefix_report.json`.
   - Every other file of `b1_ep3` and `sf_ep3` has the same md5 on Babel, except the parquet. That file differs only in its absolute path strings.
2. **Rebuild.** Ran the following:
   - `python run_posix_glob.py scene_build/scenefix.py b s --support-top support_0:0.9990`
   - `python run_posix_glob.py scene_build/comfix.py fix s c white_pot`

   `RELOCATE` was set to `scene_build/relocate_scene.py`.

   `run_posix_glob.py` only makes `glob.glob` return `/` separators. The scripts parse `sequence_id=([^/]+)` from glob paths, and that would break on Windows `\` paths.

   `comfix` printed `white_pot: inertial origin [0. 0. 0.] -> [-0.008451 -0.062887  0.049392]`, which is identical to the orchard log.
3. **Comparison.** Ran `python repro_compare.py c <archived sc_ep3>`. The archived scene is `C:\v2d_demo\archive\h_ep3_sc_K_s42\scene.tgz`, sha256 `154922eb...`. That archive is itself md5-identical to `babel-s5-16:/tmp/haisenk_v2d/scenes/sc_ep3` for all 31 non-PNG files.

## Result

The full output is in `compare_repro_vs_archive.txt`.

- **Identical files.** Every file is identical, including both edited files:
  - support USD, md5 `bc823a849edfb0339d7ec8730973ec9f`;
  - `white_pot_rigid.urdf`, md5 `b1577693c7a1fff05442dc2c06149a8f`.

  Text files were compared after changing CRLF line endings to LF, because Windows Python writes CRLF.
- **Expected differences:**
  - `scenefix_report.json` holds local paths.
  - The scene parquet has 54/54 non-string columns identical. Only the path columns `raw_motion_file`, `object_mesh_paths` and `object_urdf_paths` differ.
- **Parquet versus the original bob1/h5_ep3_A1 parquet** (`.parquet.orig`): 63 of 63 common non-path columns are identical, so object poses, hand motion and contact labels are unchanged. `relocate_scene.py` adds the two hive-partition columns `sequence_id` and `robot_name`, whose values equal the directory names.

## Two caveats

- **Script version.** This check cannot tell the exact `scenefix.py` from its later version (`../../scene_build/scenefix_later_edits.diff`). The later version also gives the same support USD here (checked: md5 `bc823a84...`). Which version actually ran is known from the session log, not from this check: it was written at 05:10:54Z, sf_ep3 was built at 05:12:13Z, and the first edit came at 05:12:52Z.
- **Rebuilt parent.** The parent was rebuilt from `sf_ep3` plus the original USD; the full original directory was not downloaded. The md5 comparison on Babel covers every file, so this is equivalent.
