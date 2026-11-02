# path_b/cache/

Files that no code, stage, script or test of the best pipeline reads, moved out of the way (James, 2026-11-02:
"On Github repo, move redundant or non-useful contents to cache folder"). Each keeps its old path under `cache/`,
and nothing here is collected by pytest (`testpaths = ["tests"]`) or imported.

To bring one back: `git mv path_b/cache/<path> path_b/<path>`.

| File (old path under `path_b/`) | Last commit | Why it moved |
|---|---|---|
| `vm/cpu_chain.sh` | `3f0f65e5` (2026-10-24) | The v4 CPU chain of Oct 24–25 (`run.py canonical`, then S3a2 → S4b and the waterfall on the Mac; BUILD_LOG rows of 2026-10-25). It does not run S3e, so it cannot rebuild the best pipeline (p4s). Nothing calls it. `vm/phase_b.sh` (the chain on the GPU instance) and `vm/s3e_regate.py` (S3e's keep rule re-applied on the CPU, then S4a and S4b) do that work now |
| `reports/checkpoints/public_2026-10-30_gd1.json` | `0f55b31d` (2026-10-30) | Public checkpoint of run `v4_gd1` (G-D's moving carrier): `ref_inverted` 0.7329 playback. The best is now `public_2026-11-02_p4s` (0.8239 playback, run `v4_p4s`). `vm/score_checkpoint.py compare` reads only the checkpoint it is given, and no code names this one |

Files removed by earlier cleanups are in git only: the Oct 31 cleanup's 46 files at `git show bb6bc46f:path_b/<path>`.
