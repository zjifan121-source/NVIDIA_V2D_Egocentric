# Path B v3 build log

Append only. One line per task per workflow step (`V3_BUILD_SPEC.md` §9): **local** (Mac: code, unit
tests, commit) → **VM** (L40S: pull, GPU test on ep40 and ep26) → **PR** (into `main_buffer`).
Every public-episode score sits next to the number it replaces (oracle baselines from
`oracles/oracle_2026-10-17.json`: ep40 recon 0.34 / still 0.40; ep26 scene 0.45 / still 0.50; the
`stage2_stereo` column from HF run `0fa525bc`). Secrets never appear here.

| Date | Task | Step | Commit | Test | Result and numbers |
|---|---|---|---|---|---|
| 2026-10-21 | 1 | local | `ae1f0e08` | `pytest path_b/tests`: 18 passed (16 existing + 2 new in `test_runner.py`) | v2 files moved to `cache/v2/` (history kept via `55fda316`); `STAGES` = spec §1; `run.py oracle`/`pack` exit 2 "not built yet". Mac P0–P1 on ep41 vs `4c2eac31`: P0 identical (8 metrics); P1 within 1.3 % (features 2522 vs 2491, row offset 0.3415 vs 0.3434 px, all PASS) and byte-identical to the earlier Mac run `mac_p01`, so the difference is OpenCV on macOS vs Linux; the binding check is P0–P2 on the VM |
| 2026-10-22 | 1 | VM | `f238fec1` | ep41 `run.py run --stages p0_ingest p1_calibrate p2_depth` (out `t1_f238fec`) vs restored `4c2eac31`, `vm/compare_runs.py`; VM `pytest`: 18 passed | **PASS.** All three stages PASS as before. P0 and P1: all 8 + 8 metrics identical. P2: 10 of 11 metrics within 0.1 %, largest-plane RMS 0.4266 → 0.4256 cm (0.23 %); stereo offset −1 frame (same); `ego_cam_a` vs calibration 0.2568° / 5.219 → 5.220 mm; P2 345 s, GPU peak 16.6 GB. P0–P2 code is unchanged since `4c2eac31`; the only changed input is the stereo TensorRT engine, rebuilt on this instance. New VM (31.56.109.90): `setup.sh` needed `f238fec1` (fresh instance: `hub/` missing ended the script under pipefail); `loginctl enable-linger root` set |
