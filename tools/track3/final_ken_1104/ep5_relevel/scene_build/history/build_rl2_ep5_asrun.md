# rl2_ep5: the commands as they actually ran (4 Nov 2026, EST)

These commands come from the session log (tool calls at 06:47:52Z, 06:49:06Z and 06:49:30Z) and the build log on the
node. They ran on orchard flame-21 inside `srun --jobid=157612 --overlap` with these settings:

- `cwd=/tmp/haisenk_v2d/scenes`
- `P=/tmp/haisenk_v2d/v2d_train/baseline/robotic_grounding/flash_chord/.venv/bin/python`

Orchard is gone. A byte-identical copy of its `/tmp/haisenk_v2d` (scenes, tools, runs, archive and
`orchard_home/v2d_orchard`) is on babel-s5-16 `/tmp/haisenk_v2d`. The `RELOCATE` default is
`~/v2d_orchard/relocate_scene.py`, sha256 `d1f55474…`, which is the same file as `../relocate_scene.py`.

## 1. Re-level, 01:47:52 EST

The local `relevel.py` (sha256 `0520b6c9…`, identical to `../relevel.py`) was first piped to
`/tmp/haisenk_v2d/tools/relevel.py`. Then:

```
$P /tmp/haisenk_v2d/tools/relevel.py w7_attcom_ep5 rl_ep5 -0.02000265 -0.09335555 0.99543188 0.020094445 0.093783968 1.015406946 white_pot --drop-other-supports
```

Printed result (identical to `records/relevel_report.json`):

- tilt 5.478626169740843 deg
- table_z_after 1.0176935260577498
- support_top 0.9996677266454285
- frame-0 bottoms: pot 1.0006677, spoon 1.0107393
- dropped `support_1`
- relocate "changed 5 paths"
- invariants: object-relative pose change ≤ 8.3e-8 m; hand-to-anchor and wrist-to-anchor distance change ≤ 9.0e-8 m

## 2. Pot trim, 01:49:06 EST

This step used the scenefix.py then on orchard, `scenefix_asrun_20261104T0649Z.py` (see section 4):

```
$P /tmp/haisenk_v2d/tools/scenefix.py rl_ep5 rl2_ep5 --obj white_pot:-1.00 --support-top support_0:1.0097 > rl2_ep5.log 2>&1
```

The `--obj` part ran. It rewrote these three meshes, rim fixed, with the pot 1.0 cm shorter:

- `processed/ep5_hw_w7_attcom_ep5_white_pot.obj`
- `processed/ep5_hw_w7_attcom_ep5_white_pot_visual.stl`
- `score_input/meshes/episode_000005/white_pot.obj`

The `--support-top` part then crashed. Its USD regex expected a 4-space-indented closing brace, but relevel writes the
prims unindented. See `rl2_ep5.log`: line 108, `AttributeError`. So no `scenefix_report.json` was written, the support
was not moved, and relocate did not run.

## 3. Support top and relocate, 01:49:30 EST

The regex in the local scenefix.py was fixed to accept any indentation. That fixed version is `../scenefix.py`, sha256
`65f7d926…`, and it was uploaded to `/tmp/haisenk_v2d/tools/scenefix.py`. It was **not** re-run on rl2_ep5. Instead:

```
cd /tmp/haisenk_v2d/scenes/rl2_ep5
U=$(ls assets/human_motion_data/ego_recon/reconstructed_stage/*_support.usda)
python3 - $U < support_top_inline_20261104T0649Z.py      # ran as an inline heredoc; same statements
$P ~/v2d_orchard/relocate_scene.py /tmp/haisenk_v2d/scenes/rl2_ep5
```

Printed result:

- `support_0 center z 0.9946677266454285 -> 1.0047000000000001`, so the top is 1.0097
- relocate "changed 5 paths": the copied `rl_ep5` paths now point into `rl2_ep5`
- `object_mesh_paths` now under `/tmp/haisenk_v2d/scenes/rl2_ep5/...`

Check right after the build (`../../diagnosis/outputs/restspots_rl2_ep5_20261104T0149EST.txt`): at frame 0, the pot and
the spoon both have their bottom at z 1.0107.

## 4. Why the one-command form in `../build_rl2_ep5.sh` gives the same scene

`scenefix_asrun_20261104T0649Z.py` is the version on orchard at 06:49:06Z. It was rebuilt from the session log: the
05:10:54Z Write of scenefix.py, plus the 05:12:46Z sed (`copytree(symlinks=True)`; skip symlinked bundle meshes). That
sed version was uploaded at 05:12:46Z. The session logs (main session and sub-agents) show no other upload of
scenefix.py before 06:49:30Z.

The rebuilt file agrees with the record in two ways:

- Its line 108 is the `body = m.group(1)` line in the traceback in `rl2_ep5.log`.
- It differs from `../scenefix.py` only in the two `--support-top` regex lines. The `--obj` code path is byte-identical.

The fixed `--support-top` writes `topz - height/2`, which is `1.0097 - 0.005`. That is the same float the inline snippet
wrote. Section 3's relocate is the same call that scenefix.py makes at its end.

`checks/rebuild_verification/` confirms this. Starting from the bundle's parent scene, the two commands in
`build_rl2_ep5.sh` reproduce the archived `rl2_ep5`:

- all 68 parquet columns are numerically bit-identical;
- the pot meshes are vertex-identical;
- the STL is byte-identical;
- the USD is text-identical.

## 5. Training queue, 01:49:48 EST

These lines were appended to `/tmp/haisenk_v2d/kq.txt` (driver `kq.sh`, see `../../training/`):

```
# 02:05 world re-levelled to the depth-measured table (anchor re-seat 3-6 deg at reset diagnosed in ep1/ep5 rollouts); rl2_ep5 also drops the partial spoon disk and makes pot/spoon rest coplanar (pot mesh -1.0 cm)
r_ep5_rl_K_s42 5 /tmp/haisenk_v2d/scenes/rl2_ep5/assets/human_motion_data/ego_recon/processed/sequence_id=ep5_hw_w7_attcom_ep5/robot_name=sharpa_wave 42 KEN 2
r_ep5_rl_K_s43 5 /tmp/haisenk_v2d/scenes/rl2_ep5/assets/human_motion_data/ego_recon/processed/sequence_id=ep5_hw_w7_attcom_ep5/robot_name=sharpa_wave 43 KEN 2
r_ep5_rl_K_s44 5 /tmp/haisenk_v2d/scenes/rl2_ep5/assets/human_motion_data/ego_recon/processed/sequence_id=ep5_hw_w7_attcom_ep5/robot_name=sharpa_wave 44 KEN 2
```
