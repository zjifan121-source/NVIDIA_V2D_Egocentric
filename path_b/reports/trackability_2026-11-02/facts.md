# Trackability, step 0: facts (2026-11-02)

Task: "trackability score (predict rollout from reconstruction)", step 0. Everything here comes from the repo and
local run folders. Two things were added, both read-only: F2's own S5 / S6 files, copied from the L40S (none of
them was on the Mac), and nothing was run on a GPU.

**Sources**
- **F2** is run `v4_h1` on the L40S: 60M steps, 19 public rollouts (BUILD_LOG 2026-10-29: G2, F2 and G3 rows).
  - Its `s5_task`, `s6_eval`, `s6_select`, `s6_train` (no weights, no logs) and `*_pre_k7` folders are copied to
    `path_b/data/outputs/v4_h1_vm/` (689 MB; its `README.md`).
  - The VM's md5s of `s4b_reference` and of S5's bundle copies are in `v4_h1_vm/md5_vm.txt`.
- **Trained reference.** S5 rewrites the bundle's hands (its hand repair), so S5's copy of `result.npz` never has
  S4b's md5.
  - The md5 of the VM's `s4b_reference/<object>/<candidate>/result.npz` (S5's input) is matched against the Mac
    runs.
  - On all 39 pairs, S5's object trajectory equals the matched run's exactly (maximum difference 0.0).
- **Playback** is the matched run's `waterfall.json` `ref_<candidate>` (official metric code; GT through
  `pathb/gt.py`).
- **Rollout** is `s6_select/selection.json` `scores`: the official metric on the kit evaluator's rollout, as F2
  recorded it.
- **First frame > 10 cm against the reference:** from the rollout parquet (world 0, the object with the larger
  error).
- **First frame > 10 cm against GT:** `gt.official_frame_errors` on the rollout poses (`score_local.evaluator_poses`)
  with S5's scoring meshes, as `s6_select` scores them.
- **Scripts:** `scripts/facts01.py`, `scripts/facts02.py`, `scripts/tables.py`; contacts and ep39 via
  `pathb/trackability.py` (`scripts/step2_pairs.py`).

## Facts

### 0.1 Pairs

Six metrics in the order AUC / SP-SR / MP-SR / MPPE cm / RPE cm / CD-O cm. `trained ref` = the Mac run whose `s4b_reference/<object>/<candidate>/result.npz` has the md5 of the VM's (S5's input); `k7` = that md5 is `v4_k7`'s.

| ep | candidate | trained ref (md5 per object) | k7 | playback | rollout | tax | GT pick | GT-free pick |
|---|---|---|---|---|---|---|---|---|
| 0 | inverted | `v4_j10`: white_pot `1de02bb2`, white_pot_lid `6c527a76` | **no** | 0.654 / 1.00 / 0.00 / 3.04 / 4.00 / 0.52 | 0.375 / 0.00 / 0.00 / 18.16 / 30.29 / 0.52 | +0.279 | still | still |
| 0 | still | `v4_k7` = `v4_j10`: white_pot `3cdd4d67`, white_pot_lid `4f49ca0b` | yes | 0.498 / 1.00 / 0.00 / 8.06 / 13.13 / 0.52 | 0.403 / 1.00 / 0.00 / 9.29 / 15.29 / 0.52 | +0.094 | still | still |
| 2 | inverted | `v4_j10`: white_pot `e27586d9`, white_pot_lid `665f0621` | **no** | 0.659 / 1.00 / 0.00 / 3.06 / 4.33 / 0.52 | 0.274 / 0.00 / 0.00 / 10.82 / 19.91 / 0.51 | +0.385 | still | still |
| 2 | still | `v4_k7` = `v4_j10`: white_pot `0b504c8a`, white_pot_lid `2174bf6e` | yes | 0.456 / 1.00 / 0.00 / 9.60 / 16.71 / 0.52 | 0.397 / 0.00 / 0.00 / 10.55 / 18.71 / 0.51 | +0.059 | still | still |
| 11 | inverted | `v4_k7`: beige_cup `2ab112f2`, blue_cup `984bb0e4` | yes | 0.883 / 1.00 / 1.00 / 1.77 / 1.84 / 0.37 | 0.258 / 0.00 / 0.00 / 17.14 / 27.28 / 0.38 | +0.625 | still | still |
| 11 | still | `v4_k7`: beige_cup `5c21200e`, blue_cup `52de9bc6` | yes | 0.599 / 1.00 / 0.00 / 8.02 / 15.26 / 0.37 | 0.439 / 0.00 / 0.00 / 30.52 / 59.29 / 0.38 | +0.161 | still | still |
| 12 | inverted | `v4_j10`: blue_cup `0ccf94e9`, plastic_dish_rack `d521ea95` | **no** | 0.718 / 1.00 / 0.00 / 3.20 / 2.02 / 0.59 | 0.449 / 0.00 / 0.00 / 9.82 / 12.52 / 0.59 | +0.269 | inverted | inverted |
| 12 | still | `v4_j10`: blue_cup `8205e5ab`, plastic_dish_rack `64147782` | **no** | 0.448 / 0.00 / 0.00 / 9.19 / 10.23 / 0.59 | 0.083 / 0.00 / 0.00 / 87.20 / 44.26 / 0.59 | +0.365 | inverted | inverted |
| 13 (`_pre_k7`) | inverted | `v4_j10`: blue_cup `2538ab63`, plastic_dish_rack `ddb5d963` | **no** | 0.722 / 1.00 / 0.00 / 2.98 / 7.74 / 0.58 | 0.477 / 0.00 / 0.00 / 8.41 / 16.16 / 0.58 | +0.246 | inverted | inverted |
| 13 (`_pre_k7`) | still | `v4_j10`: blue_cup `87d40e5b`, plastic_dish_rack `b2ff020d` | **no** | 0.479 / 0.00 / 0.00 / 9.85 / 15.02 / 0.58 | 0.076 / 0.00 / 0.00 / 56.92 / 107.89 / 0.58 | +0.403 | inverted | inverted |
| 18 | inverted | `v4_j10`: mini_dust_pan `1d1626f2`, mini_sweeper `d4f3e6b1` | **no** | 0.885 / 1.00 / 1.00 / 1.80 / 4.76 / 0.35 | 0.331 / 0.00 / 0.00 / 7.36 / 31.47 / 0.35 | +0.555 | still | still |
| 18 | still | `v4_j10`: mini_dust_pan `d1ea745f`, mini_sweeper `b29a84a7` | **no** | 0.613 / 1.00 / 0.00 / 4.29 / 9.94 / 0.35 | 0.387 / 0.00 / 0.00 / 6.57 / 27.88 / 0.35 | +0.227 | still | still |
| 21 | inverted | `v4_j10`: blue_cup `10fa7d7a`, water_pitcher `70d591a3` | **no** | 0.387 / 0.00 / 0.00 / 12.14 / 18.18 / 0.79 | 0.059 / 0.00 / 0.00 / 91.53 / 35.07 / 0.79 | +0.328 | still | still |
| 21 | still | `v4_j10`: blue_cup `13411679`, water_pitcher `3bcb46c3` | **no** | 0.143 / 0.00 / 0.00 / 24.93 / 22.38 / 0.79 | 0.120 / 0.00 / 0.00 / 24.35 / 19.47 / 0.79 | +0.024 | still | still |
| 22 | inverted | `v4_j10`: blue_cup `d71defb8`, water_pitcher `9c559217` | **no** | 0.719 / 1.00 / 0.00 / 4.57 / 6.88 / 0.61 | 0.086 / 0.00 / 0.00 / 35.54 / 59.48 / 0.61 | +0.633 | still | still |
| 22 | still | `v4_j10`: blue_cup `ea3ef9c9`, water_pitcher `50b7d8ab` | **no** | 0.381 / 1.00 / 0.00 / 7.55 / 9.48 / 0.61 | 0.315 / 0.00 / 0.00 / 10.95 / 20.91 / 0.61 | +0.066 | still | still |
| 23 | inverted | `v4_k7`: wooden_piece_1 `4141e916`, wooden_piece_2 `6eb54c6b` | yes | 0.494 / 0.00 / 0.00 / 4.31 / 2.68 / 0.32 | 0.010 / 0.00 / 0.00 / 34.30 / 38.85 / 0.31 | +0.484 | still | still |
| 23 | still | `v4_k7`: wooden_piece_1 `7fb98bc8`, wooden_piece_2 `f9490007` | yes | 0.079 / 0.00 / 0.00 / 13.47 / 24.93 / 0.32 | 0.041 / 0.00 / 0.00 / 31.75 / 35.88 / 0.31 | +0.038 | still | still |
| 24 | inverted | `v4_j10`: wooden_piece_1 `4737cd38`, wooden_piece_2 `cf936827` | **no** | 0.029 / 0.00 / 0.00 / 34.11 / 25.19 / 0.31 | 0.017 / 0.00 / 0.00 / 62.79 / 87.57 / 0.32 | +0.012 | still | still |
| 24 | still | `v4_j10`: wooden_piece_1 `095f5991`, wooden_piece_2 `33a1ad12` | **no** | 0.029 / 0.00 / 0.00 / 28.98 / 31.26 / 0.31 | 0.028 / 0.00 / 0.00 / 31.80 / 32.44 / 0.32 | +0.000 | still | still |
| 26 | inverted | `v4_j10`: plastic_dish_rack `8657dc3a`, white_pot `93effb84` | **no** | 0.859 / 1.00 / 1.00 / 1.59 / 2.35 / 0.61 | 0.770 / 1.00 / 0.00 / 2.50 / 3.88 / 0.61 | +0.088 | inverted | inverted |
| 26 | still | `v4_j10`: plastic_dish_rack `ba31add1`, white_pot `eb18ef81` | **no** | 0.660 / 0.00 / 0.00 / 7.16 / 12.95 / 0.61 | 0.620 / 0.00 / 0.00 / 7.83 / 13.21 / 0.61 | +0.041 | inverted | inverted |
| 27 | inverted | `v4_j10`: plastic_dish_rack `25dd7991`, white_pot `49fce672` | **no** | 0.746 / 1.00 / 0.00 / 2.91 / 5.34 / 0.57 | 0.526 / 0.00 / 0.00 / 9.38 / 16.84 / 0.57 | +0.220 | inverted | inverted |
| 27 | still | `v4_j10`: plastic_dish_rack `6116a87f`, white_pot `636d60db` | **no** | 0.650 / 0.00 / 0.00 / 7.34 / 14.16 / 0.57 | 0.500 / 0.00 / 0.00 / 10.73 / 20.59 / 0.57 | +0.150 | inverted | inverted |
| 39 | inverted | `v4_j10`: mini_dust_pan `0bb2d72d` | **no** | 0.816 / 1.00 / 1.00 / 2.56 / 0.00 / 0.39 | 0.594 / 0.00 / 0.00 / 36.46 / 0.00 / 0.38 | +0.222 | inverted | inverted |
| 39 | still | `v4_j10`: mini_dust_pan `07796643` | **no** | 0.175 / 0.00 / 0.00 / 16.73 / 0.00 / 0.39 | 0.175 / 0.00 / 0.00 / 16.74 / 0.00 / 0.38 | +0.000 | inverted | inverted |
| 40 | inverted | `v4_j10`: mini_dust_pan `25520e5a` | **no** | 0.939 / 1.00 / 1.00 / 1.23 / 0.00 / 0.38 | 0.842 / 1.00 / 0.00 / 4.83 / 0.00 / 0.39 | +0.097 | inverted | inverted |
| 40 | still | `v4_j10`: mini_dust_pan `d2e56d3b` | **no** | 0.400 / 0.00 / 0.00 / 12.92 / 0.00 / 0.38 | 0.400 / 0.00 / 0.00 / 12.95 / 0.00 / 0.39 | +0.000 | inverted | inverted |
| 41 | inverted | `v4_j10`: white_pot `64e2db6a` | **no** | 0.831 / 1.00 / 1.00 / 2.08 / 0.00 / 0.51 | 0.689 / 1.00 / 0.00 / 3.17 / 0.00 / 0.51 | +0.142 | inverted | inverted |
| 41 | keyframe | `v4_j10`: white_pot `2530a8a4` | **no** | 0.406 / 1.00 / 0.00 / 9.11 / 0.00 / 0.51 | 0.390 / 1.00 / 0.00 / 9.51 / 0.00 / 0.51 | +0.016 | inverted | inverted |
| 41 | still | `v4_j10`: white_pot `7756e41e` | **no** | 0.356 / 1.00 / 0.00 / 9.79 / 0.00 / 0.51 | 0.356 / 1.00 / 0.00 / 9.83 / 0.00 / 0.51 | +0.000 | inverted | inverted |
| 42 | inverted | `v4_j10`: white_pot `67aef432` | **no** | 0.621 / 1.00 / 1.00 / 3.00 / 0.00 / 0.55 | 0.219 / 0.00 / 0.00 / 9.12 / 0.00 / 0.56 | +0.403 | inverted | still |
| 42 | still | `v4_j10`: white_pot `7ff7608f` | **no** | 0.191 / 0.00 / 0.00 / 7.37 / 0.00 / 0.55 | 0.191 / 0.00 / 0.00 / 8.99 / 0.00 / 0.56 | +0.000 | inverted | still |
| 43 | inverted | `v4_j10`: white_pot `cca5f63b` | **no** | 0.435 / 0.00 / 0.00 / 7.47 / 0.00 / 0.51 | 0.116 / 0.00 / 0.00 / 11.57 / 0.00 / 0.51 | +0.319 | still | inverted |
| 43 | still | `v4_j10`: white_pot `84882cb9` | **no** | 0.132 / 0.00 / 0.00 / 11.84 / 0.00 / 0.51 | 0.132 / 0.00 / 0.00 / 11.84 / 0.00 / 0.51 | +0.000 | still | inverted |
| 44 | inverted | `v4_j10`: white_pot `b7adb8e0` | **no** | 0.830 / 1.00 / 1.00 / 1.33 / 0.00 / 0.55 | 0.850 / 1.00 / 1.00 / 1.25 / 0.00 / 0.56 | -0.020 | inverted | inverted |
| 44 | still | `v4_j10`: white_pot `bb8c8d14` | **no** | 0.596 / 1.00 / 0.00 / 6.96 / 0.00 / 0.55 | 0.594 / 1.00 / 0.00 / 7.20 / 0.00 / 0.56 | +0.003 | inverted | inverted |
| 45 | inverted | `v4_k7` = `v4_j10`: white_pot `25401a2a` | yes | 0.629 / 1.00 / 0.00 / 3.56 / 0.00 / 0.43 | 0.258 / 0.00 / 0.00 / 13.84 / 0.00 / 0.48 | +0.372 | inverted | inverted |
| 45 | still | `v4_k7` = `v4_j10`: white_pot `37ef2206` | yes | 0.391 / 0.00 / 0.00 / 12.91 / 0.00 / 0.43 | 0.195 / 0.00 / 0.00 / 13.22 / 0.00 / 0.48 | +0.196 | inverted | inverted |

**Counts against the task's expectation**
- **Inverted: 19 pairs.** The task expected 13.
  - The 13 = the G-H row's 11 (18 42 45 41 39 27 22 12 40 26 44, md5-checked on 2026-10-31) + G3's 11 and 23 (on
    k7's references).
  - The other six (0, 2, 13, 21, 24, 43) are md5-verified here too.
  - **UNKNOWN: needs James:** which 13 pairs the 2026-11-02 research used. Step 1 reports both sets.
- **Keyframe: 1** (ep41, 0.406 → 0.390, as BUILD_LOG 2026-10-29 G2 row).
- **Still: 19** (≥ 10).
- **S0.1 GT pairs:** BUILD_LOG decisions 2026-10-29 give ep41 GT reference → rollout 0.935 (MPPE 1.26 cm) and ep13
  0.899 (MPPE 1.37 cm).
  - Their rollout and S5 files are not in any local run folder (searched `path_b/data/outputs`).
  - **UNKNOWN: needs James:** where they are. Without an S5 task they have no features, so they are not in the
    fits.
- **Not paired:**
  - ep31: inverted trained, never evaluated, so no rollout.
  - ep13's current `s6_*`: G3's k7 retrain left no rollout. Its `selection.json` names candidates whose S5 bundles
    are absent.
- **G3's `g3d` retrain:** nothing reached the Mac.
  - On the VM, only ep13's folders were set aside (`*_pre_k7` = F2's original ep13 on `v4_j10`, used here).
  - The queued retrains of 18, 12, 44, 26 and 2 left no `*_pre_k7` folders, and their references are still
    `v4_j10`'s by md5. So they did not run.
- **Trained md5 ≠ the current k7 reference:** 31 of 39 pairs (column `k7`).
  - Equal on: ep0 still, ep2 still, ep11 (both), ep23 (both), ep45 (both).
  - ep45 and ep2's still files are both j10's and k7's (same md5). Everything else F2 trained is `v4_j10`'s.

### 0.2 Still playbacks of the run F2 trained on (`ref_still`, playback)

| ep | trained ref | AUC | SP-SR | MP-SR | MPPE cm | RPE cm | CD-O cm | still rollout AUC |
|---|---|---|---|---|---|---|---|---|
| 12 | `v4_j10`: blue_cup `8205e5ab`, plastic_dish_rack `64147782` | 0.448 | 0.00 | 0.00 | 9.19 | 10.23 | 0.59 | 0.083 |
| 13 (`_pre_k7`) | `v4_j10`: blue_cup `87d40e5b`, plastic_dish_rack `b2ff020d` | 0.479 | 0.00 | 0.00 | 9.85 | 15.02 | 0.58 | 0.076 |
| 39 | `v4_j10`: mini_dust_pan `07796643` | 0.175 | 0.00 | 0.00 | 16.73 | 0.00 | 0.39 | 0.175 |
| 22 | `v4_j10`: blue_cup `ea3ef9c9`, water_pitcher `50b7d8ab` | 0.381 | 1.00 | 0.00 | 7.55 | 9.48 | 0.61 | 0.315 |
| 27 | `v4_j10`: plastic_dish_rack `6116a87f`, white_pot `636d60db` | 0.650 | 0.00 | 0.00 | 7.34 | 14.16 | 0.57 | 0.500 |
| 2 | `v4_k7` = `v4_j10`: white_pot `0b504c8a`, white_pot_lid `2174bf6e` | 0.456 | 1.00 | 0.00 | 9.60 | 16.71 | 0.52 | 0.397 |
| 42 | `v4_j10`: white_pot `7ff7608f` | 0.191 | 0.00 | 0.00 | 7.37 | 0.00 | 0.55 | 0.191 |
| 45 | `v4_k7` = `v4_j10`: white_pot `37ef2206` | 0.391 | 0.00 | 0.00 | 12.91 | 0.00 | 0.43 | 0.195 |

### 0.3 Evaluator record per rollout

| ep | candidate | failure_cause_fraction object / wrist | non_finite_worlds | steps | first frame > 10 cm: vs reference | vs GT |
|---|---|---|---|---|---|---|
| 0 | inverted | 1 / 1 | 0 | 373 | 129 | 131 |
| 0 | still | 0 / 0 | 0 | 373 | — | 63 |
| 2 | inverted | 1 / 1 | 0 | 343 | 37 | 67 |
| 2 | still | 0 / 0 | 0 | 343 | — | 65 |
| 11 | inverted | 1 / 1 | 0 | 137 | 31 | 30 |
| 11 | still | 1 / 1 | 0 | 137 | 73 | 30 |
| 12 | inverted | 1 / 1 | 0 | 181 | 39 | 40 |
| 12 | still | 1 / 0 | 0 | 181 | 2 | 3 |
| 13 (`_pre_k7`) | inverted | 1 / 1 | 0 | 190 | 44 | 44 |
| 13 (`_pre_k7`) | still | 1 / 0 | 0 | 190 | 2 | 2 |
| 18 | inverted | 1 / 1 | 0 | 353 | 61 | 40 |
| 18 | still | 1 / 0 | 0 | 353 | — | 56 |
| 21 | inverted | 1 / 1 | 0 | 186 | 2 | 0 |
| 21 | still | 0 / 0 | 0 | 186 | — | 0 |
| 22 | inverted | 1 / 1 | 0 | 189 | 49 | 3 |
| 22 | still | 1 / 0 | 0 | 189 | — | 3 |
| 23 | inverted | 1 / 1 | 0 | 325 | 18 | 30 |
| 23 | still | 0 / 0 | 0 | 325 | — | 50 |
| 24 | inverted | 1 / 1 | 0 | 275 | 16 | 0 |
| 24 | still | 1 / 0 | 0 | 275 | 8 | 0 |
| 26 | inverted | 0 / 0 | 0 | 158 | — | 75 |
| 26 | still | 0 / 0 | 0 | 158 | — | 57 |
| 27 | inverted | 1 / 1 | 0 | 246 | 63 | 59 |
| 27 | still | 1 / 0 | 0 | 246 | 144 | 0 |
| 39 | inverted | 1 / 1 | 0 | 319 | 218 | 214 |
| 39 | still | 0 / 0 | 0 | 319 | — | 55 |
| 40 | inverted | 1 / 1 | 0 | 260 | 234 | 233 |
| 40 | still | 0 / 0 | 0 | 260 | — | 47 |
| 41 | inverted | 1 / 1 | 0 | 299 | 130 | 144 |
| 41 | keyframe | 0 / 1 | 0 | 299 | — | 41 |
| 41 | still | 0 / 0 | 0 | 299 | — | 41 |
| 42 | inverted | 1 / 1 | 0 | 257 | 66 | 82 |
| 42 | still | 1 / 0 | 0 | 257 | — | 108 |
| 43 | inverted | 1 / 1 | 0 | 326 | 219 | 57 |
| 43 | still | 0 / 0 | 0 | 326 | — | 57 |
| 44 | inverted | 0 / 1 | 0 | 249 | — | — |
| 44 | still | 0 / 0 | 0 | 249 | — | 83 |
| 45 | inverted | 1 / 1 | 0 | 290 | 51 | 56 |
| 45 | still | 1 / 1 | 0 | 290 | — | 56 |


### 0.4 How S5 writes the 16 contact slots

- **Writer.** S5's mode A (the default; `pathb/stage23.py` runs NVIDIA's loader through `drivers/nvidia_stage23.py`)
  leaves the slots to NVIDIA's `v2d_task_library_loader` (`baseline/reconstruction/modules/v2d_task_library_loader/lib/`,
  unmodified since `d9264a4e`). Our own code writes no contact; mode B (S4c, off by default) would write DaiD's
  (`pathb/stages/s5_task.py:146-157`).
- **Source hands.** Per frame and hand, the posed MANO mesh vertices `mano_results[side]["vertices"][frame_id]`
  (`dataset_loader_base.py:734`). These are the bundle's MANO hands as the loader poses them: P1d's MANO after
  S4b's hand cleaning and S5's repair.
- **Distance threshold.** `approximate_contact_with_id(..., threshold=0.01)` (`dataset_loader_base.py:750-757`).
  An object surface point is a contact when its nearest hand vertex is under 1 cm (`contact_utils.py:50`).
- **Per-frame count.**
  - The contacts are averaged per MANO link into 16 slots per hand (`compute_hand_link_contact_positions`,
    `contact_utils.py:84-160`). Each slot gets an inverse-distance-weighted point and a voted part id (1 or 2:
    the object's slot, `contact_utils.py:100`).
  - "Links with no contact get zeros" (`contact_utils.py:96`). The object-side point
    (`find_object_contact_positions`) is zeroed where the link point is zero (`contact_utils.py:186-192`).
  - So a frame has 0-16 filled slots per hand. Filled means a non-zero point; here, for one object, also part
    id = its slot + 1.
- **Split by S3c kind.** The table below covers the inverted candidate.
  - Kinds per frame come from the md5-matched run's `s3c_invert/<object>/invert_report.json`: bridge = inside a
    `bridge` span; grasp = a grasp segment, failed or not, outside a bridge; static = a static segment;
    other = the rest (unstable, no bridge).
  - Objects are pooled: each frame of each object counts once.

| ep | static: frames, >= 1, >= 2 | grasp | bridge | other |
|---|---|---|---|---|
| 0 | 300, 0.03, 0.03 | 106, 0.98, 0.96 | 340, 0.52, 0.48 | 0 |
| 2 | 341, 0.27, 0.26 | 149, 0.99, 0.99 | 196, 0.59, 0.57 | 0 |
| 11 | 96, 0.11, 0.11 | 68, 0.91, 0.90 | 110, 0.76, 0.60 | 0 |
| 12 | 150, 0.00, 0.00 | 97, 1.00, 1.00 | 115, 0.22, 0.22 | 0 |
| 13_pre_k7 | 199, 0.01, 0.01 | 0 | 181, 0.77, 0.77 | 0 |
| 18 | 164, 0.00, 0.00 | 464, 0.99, 0.99 | 78, 0.59, 0.59 | 0 |
| 21 | 112, 0.00, 0.00 | 0 | 260, 0.55, 0.53 | 0 |
| 22 | 91, 0.00, 0.00 | 135, 1.00, 0.96 | 152, 0.46, 0.45 | 0 |
| 23 | 182, 0.00, 0.00 | 370, 1.00, 1.00 | 98, 0.73, 0.70 | 0 |
| 24 | 267, 0.60, 0.60 | 168, 0.89, 0.89 | 115, 0.29, 0.27 | 0 |
| 26 | 226, 0.48, 0.46 | 31, 1.00, 1.00 | 59, 0.86, 0.76 | 0 |
| 27 | 267, 0.00, 0.00 | 81, 1.00, 1.00 | 144, 0.42, 0.36 | 0 |
| 39 | 64, 0.17, 0.17 | 207, 1.00, 1.00 | 48, 0.25, 0.21 | 0 |
| 40 | 45, 0.00, 0.00 | 206, 1.00, 0.99 | 9, 0.56, 0.44 | 0 |
| 41 | 51, 0.00, 0.00 | 186, 1.00, 0.99 | 62, 0.97, 0.92 | 0 |
| 42 | 70, 0.26, 0.26 | 40, 1.00, 1.00 | 147, 0.45, 0.44 | 0 |
| 43 | 43, 0.40, 0.37 | 34, 1.00, 1.00 | 249, 0.63, 0.54 | 0 |
| 44 | 65, 0.23, 0.22 | 158, 1.00, 1.00 | 26, 0.46, 0.35 | 0 |
| 45 | 83, 0.49, 0.47 | 62, 1.00, 1.00 | 145, 0.95, 0.94 | 0 |

### 0.5 ep39: where the drop falls

F2's ep39 inverted rollout leaves its reference by > 10 cm at frame 218 (against GT: 214; 0.3). F2's own S5 task
(`v4_h1_vm/episode_000039/s5_task/inverted`) and `v4_j10`'s S3c report give:
- **Bridged: yes.** S3c's grasp 72-207 ends in a slip guard. Frames 208-221 are unstable, guess interpolated,
  bridge 208-245.
- **Hand-far: from 214.**
  - The reference hand holds the pot (≥ 2 fingertip sites within δ = 2.37 cm) up to frame 213.
  - The nearest fingertip is then 2.1 cm away at 214-215, 3.5-3.7 cm at 216-217, 2.6 / 2.4 cm at 218-219,
    10.3 cm at 220 and > 25 cm from 223.
  - The departure against GT (214) is the frame the reference hand lets go.
- **Fast: from 219, in the wrist, not the object.** The right wrist reference runs 0.3-0.65 m/s to 218, then
  3.07 m/s (219), 3.01 (220), 2.25 (221) and 10.4 m/s (223). The pot's own reference moves 0.15-0.3 m/s
  throughout.
- Same frames as the G-H row (BUILD_LOG 2026-10-31: wrist spike 218-222, max 10.4 m/s).
- **Order of events:** the bridge starts (208), the hand leaves (214), the rollout departs from GT (214), the
  rollout passes 10 cm from its reference (218), the wrist spikes (219-223).

### 0.6 What the kit evaluator does with our policy

`baseline/robotic_grounding/flash_chord/src/flash_chord/evaluation/harness.py` (unmodified since `d9264a4e`):
- **Thresholds come first.** `evaluate_checkpoint` reads the checkpoint's own failure thresholds (`thresholds =
  evaluation_thresholds(config)`, `reference_criteria = reference_tracking_criteria(config)`, lines 388-389)
  before it calls `apply_evaluation_composition` (line 390).
- **What `apply_evaluation_composition` sets** (lines 216-251):
  - `scene.world_count` = the requested count (228)
  - `env.auto_reset = False` (229)
  - `env.voc_scale = 0.0`, i.e. VOC off (230); with reset mode `explicit` also `voc.max_force = max_torque = 0`
    (231-233) and `reset.voc_decay_steps = 0` (235-236)
  - `reset.always_reset_to_first_frame = True` (234) and `reset.seed = 42` (237)
  - every failure term `enabled = False` (245-248); a non-recon termination keeps `reference_end`, and the
    composition raises if `reference_end` is off (249-250)
- **Applied afterwards.** The docstring (224-226): "The tracking-failure terms are disabled so a failed world keeps
  advancing … their limits are applied afterwards to the recorded errors, and `reference_end` completes every world".
  `reference_tracking_criteria` (184-212) gives wrist position / orientation and object position / orientation
  for our (non-recon) termination, all active from step -1. They are shifted for a settling preamble when there
  is one (254-262, 476).
- **Our runs' values.** Every one of the 39 rollouts has `world_count` 1, `reset_mode` `explicit` and
  `settling_steps` 0 (each `s6_eval/<candidate>/results/episode_*.json`). `pathb/policy.py` passes
  `evaluation.world_count=1 evaluation.reset_mode=explicit` (the eval's `.hydra/overrides.yaml`).
- **Our recipe's thresholds** (F2's Hydra record `s6_train/<candidate>/hydra/.hydra/config.yaml`, `termination`;
  identical on all 41 F2 training configs, the 2 overflow / retry runs included):
  - `wrist_position` enabled, 0.15 m
  - `wrist_orientation` disabled
  - `object_position` enabled, 0.1 m
  - `object_orientation` enabled, 0.5 rad
  - `reference_end` enabled
- **No override sets them.** None of the 41 `overrides.yaml` names a termination term, so these are the experiment
  defaults, as `pathb/policy.py:8` says ("wrist termination 0.15 m").
- **The spec says otherwise.** V4_BUILD_SPEC.md S6 quotes Edmond's `termination.wrist_position.threshold=0.3`; F2 did
  not run with it.
- **Training-only settings in the same record:** `env.voc` k 400 / d 40, max force 200 N, torque 50 N·m, VOC on
  (`voc_scale` 1.0), `world_count` 4096, `auto_reset` true. Scene: `scene.support` true,
  `scene.decompose_objects` true, `collision.ground` true, `scene.contact_friction` 1.0.
- **The evaluator JSON** holds `object_position_threshold_m` 0.1 and `object_orientation_threshold_rad` 0.5,
  plus `reference_tracking.failure_cause_fraction` per cause.
  - With one world each fraction is 0 or 1.
  - `non_finite_worlds` is empty on all 39 rollouts.

### UNKNOWN: needs James
- Which 13 pairs the 2026-11-02 research meant (0.1; inferred as G-H's 11 + G3's 11 and 23).
- Where the S0.1 GT-reference rollouts and S5 tasks are (ep41 0.935, ep13 0.899; 0.1).
