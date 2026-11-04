# Rebuild check: parent scene + scene_build/build_rl2_ep5.sh gives the archived rl2_ep5 (4 Nov 2026, ~14:25 EST)

## What was done

Everything ran locally on Windows in a scratch directory, with nothing written outside it. The input and reference
came from these archives:

- **parent:** `C:\v2d_demo\handoff\ep1_ep5_20261104\episode_000005\scene\w7_attcom_ep5.tgz`
  - sha256 `ae61a76f…`
  - the orchard training copy of `w7_attcom_ep5`, the scene the re-level ran on
- **reference:** `C:\v2d_demo\archive\r_ep5_rl_K_s43\scene.tgz`
  - sha256 `c0159b0d…`
  - the trained `rl2_ep5`; same sha256 as babel-s5-16 `/tmp/haisenk_v2d/archive/r_ep5_rl_K_s43/scene.tgz`
  - its processed parquet has sha256 `f3badc29…`, equal to babel-s5-16 `/tmp/haisenk_v2d/scenes/rl2_ep5`

The commands ran with local Python: numpy 2.5.2, scipy 1.18.1, pyarrow 26.0.1, trimesh 5.1.1. Orchard's venv had
numpy 2.4.6, scipy 1.18.0, pyarrow 24 and trimesh 4.12.2.

```
RELOCATE=<copy of scene_build/relocate_scene.py>
python relevel_win.py  <in>/w7_attcom_ep5 <work>/rl_ep5  -0.02000265 -0.09335555 0.99543188 0.020094445 0.093783968 1.015406946 white_pot --drop-other-supports
python scenefix_win.py <work>/rl_ep5 <work>/rl2_ep5 --obj white_pot:-1.00 --support-top support_0:1.0097
python cmp_scene.py <work>/rl2_ep5 <ref>/rl2_ep5   > compare_rebuild_vs_archive.json
python cmp_cols.py  <work>/rl2_ep5 <ref>/rl2_ep5   > compare_parquet_columns.txt
```

`relevel_win.py` and `scenefix_win.py` are copies of `scene_build/relevel.py` and `scene_build/scenefix.py`. Each has
one change, needed only on Windows paths: `sequence_id=([^/]+)` became `sequence_id=([^/\\]+)`. The scratch path was
passed in its 8.3 short form to stay under MAX_PATH.

## Result

The relevel report printed the same numbers as the archived `records/relevel_report.json`: tilt 5.478626169740843 deg,
the pivot, support centre and top, and the frame-0 bottoms. The invariant check (object-relative pose change) differs by 2e-16 m.

The scenefix report (`rebuild_scenefix_report.json`) shows:

- white_pot height 13.34 cm, delta -1.0 cm;
- frame-0 bottom 1.0007 -> 1.0107, last frame 1.0139;
- support_0 top 0.9997 -> 1.0097.

Against the archived `rl2_ep5`:

| item | result |
|---|---|
| processed parquet | All 68 columns numerically bit-identical, NaN-aware. Only the path strings differ (`raw_motion_file`, `object_mesh_paths`, `object_urdf_paths`): local Windows paths vs `/tmp/haisenk_v2d/scenes/rl2_ep5/...`. |
| `white_pot.obj` (processed and score_input) | Vertex- and face-identical (max \|dv\| = 0.0). The bytes differ only because Windows text mode writes CRLF. |
| `white_pot_visual.stl` | Byte-identical. |
| support USD | Text-identical apart from CRLF: support_0 translate (0.10026413237265833, -0.037574307062152534, 1.0047000000000001); support_1 removed. |
| all other scene files | Byte-identical: spoon meshes, URDFs including the COM-edited pot URDF, `bundle_cleaned/result.npz`, `.orig`, `report.json` and the rest. |
| only in the rebuild | `scenefix_report.json`; the archived build crashed before writing it (see `scene_build/history/`). Also `bundle_cleaned/mesh.obj`, an empty file left where tar could not make a dangling symlink on Windows. |

On Windows, relocate printed "changed 0 paths" in the scenefix step because local paths do not start with "/". On Linux
it repoints the paths into rl2_ep5, as the 01:49 EST build did.
