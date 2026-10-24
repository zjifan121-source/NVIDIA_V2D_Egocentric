# Path B v3 cache (2026-10-24)

v4 (`path_b/V4_BUILD_SPEC.md`) supersedes v3. This folder holds v3-only material, kept for history.
Code here is not on the import path and its tests are not collected (`pytest path_b/tests` only);
moved with `git mv` in v4 task 1, so `git log --follow` shows each file's history.

| File | Was | Why it is here |
|---|---|---|
| `V3_BUILD_SPEC.md` | `path_b/V3_BUILD_SPEC.md` | the v3 spec, with its 2026-10-24 revision (§12 was the first v4 draft) |
| `SPEC_AUDIT.md` | `path_b/SPEC_AUDIT.md` | audit of the v3 build against the v3 spec; a v4 audit starts fresh |
| `s3a_mesh.py`, `test_s3a.py` | `pathb/stages/`, `tests/` | SAM 3D mesh per view: FAILed its bars on ep40 (not watertight, 0.78× the scan); the canonical mesh (A2) is S3a now |
| `s3b2_track_points.py` | `pathb/stages/` | A2.4 mesh-free point tracking (TAPIP3D): never ran; per-frame tracking is what v4 removes |
| `s3c_fuse.py` | `pathb/stages/` | ESKF/RTS fusion of the per-frame track: dropped from v4's stage list (§1), replaced by `s3c_invert` / `s3d_settle`. `pathb/fuse.py` (the filter and rest anchors) stays |
| `phase2.py`, `test_phase2.py` | `pathb/`, `tests/` | the Do as I Do two-body scene: 0.500 either way on ep26 stand-ins; v4 handles two objects per segment. Its support relation (`disk`, `supports`) moved to `pathb/support.py` with its test (`tests/test_support.py`) |
| `standin.py` | `pathb/` | stand-in GPU outputs (`run.py standin`) from the team bundles; the real dev cache replaces them |
| `a2.py` | `pathb/` | the A2.5 comparison (`run.py a2`): decided 2026-10-23 (canonical mesh = S3a) |
| `test_a2_v3_only.py` | cut from `tests/test_a2.py` | the A2.4 and A2.5 tests; A2.1–A2.3 tests stay in `tests/test_a2.py` |
