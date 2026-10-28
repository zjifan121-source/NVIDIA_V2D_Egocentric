# Overnight runs, 2026-10-15

## Short version

1. **The still-object safety net works.** The ep41 still policy holds the pot to 0.1 cm and scores local AUC **0.356**, exactly what a perfectly still pot scores.
2. **We found and fixed a root cause: objects tipped over at the start.** In every one of our scenes the object started 5–27° off any face it can rest on. In simulation it tipped over within 5 frames and stayed 8–24° off its reference for the rest of the clip. NVIDIA's tissue sample stays under 1°. The new `--level` step rotates the scene so the object starts resting flat. That alone raised the ep41 policy from local AUC 0.133 to **0.243**, our best trained result, and took the still policy from 0.196 to 0.356.
3. **Correction this morning: with our own bundles, still scores 0.31, not 0.45.** The second objects of the public episodes arrived at 08:00, so I scored whole public episodes built from our bundles.
   - In a two-object episode Kaggle aligns the scene by one object only, so the other object pays for our placement error. Our objects sit 11–27 cm away from where they should be relative to each other.
   - Mean AUC over the 20 public episodes: still **0.309** (0.329 scan-rescaled). A perfect copy of the reconstruction: 0.293 (**0.347** scan-rescaled). Leaderboard: 2nd 0.387, 3rd 0.307.
   - Last night's 0.447 used the true placement.
4. **Three scored bundles tracked the wrong thing** (new identity check). ep19's and ep30's "mini_sweeper" runs tracked the dust pan and the dish rack; ep25's wooden piece came out as a 0.9 m blob. Public ep18, ep23 and ep24 show the same failures. They cost the most in the dry run (AUC 0.03–0.30).
5. **Grasp loss explained.** While the object is lifted, the reference hands often aren't touching it. A new repair moves the hands onto the object; policies trained on it are running.
6. **A full submission still needs every object of all 20 scored episodes.** Kaggle registers each submitted mesh and scores all 20 episodes, so the 15 missing second objects block any submission, including the still one.

Code on `edm_bundle_audit` (pushed): `73e6a69a Add --level`, then `36baea1a Level scenes by default` (`--no-level` turns it off). Compare: https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/compare/main_buffer...edm_bundle_audit?expand=1

## Results

Standard settings for every run: `task.motion_speed=1.0 sim.fps=20.254 sim.njmax=1024 sim.nconmax=512`, wrist cutoff 30 cm, 250M steps.

How to read the columns:
- **Vs reference:** how well the simulated object follows the reconstruction it was trained on (evaluator).
- **Local AUC vs truth:** our scorer against public ground truth; ep41 only. A still pot scores 0.356 and the scan-rescaled reconstruction itself scores 0.430, the ceiling for a policy.

| Run | Scene | Object error vs reference | AUC vs reference | Local AUC vs truth |
|---|---|---|---|---|
| ep41_clean_wrist30 | cleaned (before last night) | 18.8 cm / 26° | 0.083 | 0.186 |
| ep41_still | still pot, **not levelled** | 4.2 cm / 25° | 0.337 | 0.196 |
| **ep41_still_level** | still pot, levelled | **0.1 cm / 0.1°** | **1.000** | **0.356** |
| ep41_scan | scan rescale | 17.5 cm / 36° | 0.169 | 0.133 |
| ep41_scan_obj20 | + object cutoff 20 cm | 79 cm / 37° | 0.180 | 0.132 |
| ep41_scan_500m | + 500M steps | 84 cm / 119° | 0.204 | 0.143 |
| **ep41_scan_level** | scan rescale + level | 10.1 cm / 31° | 0.324 | **0.243** |
| ep35_clean | spoon | 19.1 cm / 75° | 0.068 | |
| ep35_hands | hand-scale rescale | 16.8 cm / 58° | 0.060 | |
| ep37_clean | spoon | 10.4 cm / 28° | 0.514 | |
| ep37_hands | hand-scale rescale | 9.0 cm / 25° | 0.471 | |
| ep37_level | level | 10.5 cm / **12°** | 0.503 | |
| ep38_scan | dust pan, scan rescale | 43.6 cm / 71° | 0.415 | |
| **ep38_scan_level** | + level | **8.8 cm** / 25° | **0.461** | |
| ep36_clean | spoon | 21.1 cm / 93° | 0.092 | |
| ep36_level2 | level (object frame, 27° tilt) | 22.0 cm / 101° | 0.090 | |
| ep33_merged_scan | first two-object run (cup + spoon) | 32.3 cm / 76° | 0.240 | |
| **ep33_merged_level** | + level | **11.7 cm** / 75° | **0.264** | |

What this says:
- **Levelling helps wherever it applies.**
  - ep38 object error fell from 44 to 9 cm.
  - ep33 two-object error fell from 32 to 12 cm.
  - ep37 rotation error fell from 28° to 12°.
  - ep41 improved on every measure.
  - ep36 is unchanged. Its spoon is lost early either way, and its 27° tilt needed the object-frame fix.
- **A larger object cutoff and longer training make things worse.** The object gets thrown and is never caught again: 80 cm errors.
- **Rescaling by hand size does not help** (ep35, ep37), consistent with last night's reconstruction-only test.
- **The remaining failure is still losing the grasp once the helper force fades to 0.** 85–95% of late training episodes end with the object lost. That is the next thing to attack.

## Why a still object scores so well

Kaggle aligns each trajectory to the truth with one rigid transform taken from the first frame. A still object therefore scores the same whatever its pose or mesh: we checked our mesh against the scan, and our start pose against the true one, and both give 0.356 on ep41.

What a still object scores, per public episode (AUC):

| | |
|---|---|
| Single-object (39–45) | 0.175, 0.400, 0.356, 0.191, 0.132, 0.596, 0.391 |
| Two-object | 0.24–0.67; mean 0.52, because usually only one of the two objects moves |
| All 20 | **0.447** (SP-SR 0.45) |

Per moving object, a still pose against our reconstruction played back perfectly: still wins on 9 of 14. Mean 0.317 against 0.291; taking the better of the two each time gives 0.340. Tracking pays off only where the reconstruction is good: ep22 pitcher 0.58, ep41 pot 0.43, ep27 pot 0.22.

Caveat: in two-object episodes, a still submission also needs the two objects placed correctly *relative to each other*. That placement comes from our reconstruction, so we won't hit the 0.52 exactly.

## Public dry run with our own bundles (this morning)

`tools/track3/dryrun_public.py` builds each of the 20 public episodes exactly as stage 2 would: two objects merged into one scene, then cleaned. It scores every object of the episode together with the official metric code; no simulation is involved.

How Kaggle aligns a scene (`metric_code/track_3/AUC.py`, `_align_registered_scene`): one rigid transform per episode, from the **anchor object's** frame-0 pose (the first object without symmetry). Any error in where we put the second object *relative to the anchor* is never corrected, even if both objects are perfectly still.

| Episodes | Still | Still, scan-rescaled | Reconstruction played back | Reconstruction, scan-rescaled | Better of the two scan-rescaled, per episode |
|---|---|---|---|---|---|
| Single-object (7) | 0.320 | 0.320 | 0.404 | 0.441 | 0.461 |
| Two-object (13) | 0.303 | 0.333 | 0.233 | 0.296 | 0.378 |
| Two-object, right objects (10) | 0.357 | 0.396 | 0.272 | 0.353 | 0.454 |
| **All 20** | **0.309** | **0.329** | 0.293 | **0.347** | 0.407 |

- **Still-object placement costs 0.25 per two-object episode.** With the true placement a still two-object episode averages 0.57; with ours, 0.33. Frame-0 distance between the two objects (ours against the truth):
  - cup + pitcher: 44 cm against 33 cm (ep21)
  - rack + pot: 32 cm against 22 cm (ep27)
  - pot + lid: 1 cm against 6 cm (ep0)
  - Beyond about 10 cm the second object adds nothing to AUC.
- **On single-object episodes the reconstruction beats still** when it is good: ep42 0.74 against 0.19, ep44 0.86 against 0.60. Averaged, 0.441 against 0.320.
- **Rescaling each object by its own scan ratio does not fix placement.** It helps cup + pitcher (ep21 0.24 → 0.34, ep22 0.30 → 0.44) but breaks pot + lid and the rack episodes. Mesh-size errors don't track distance errors reliably.
- **What this means for the plan:** the realistic public score of our current bundles is 0.31–0.35, about third place. The levers, largest first:
  1. Fix the wrong-object reconstructions (next section).
  2. Get the relative placement of two objects right.
  3. Train policies that get close to the reconstruction.

## Identity check: three scored bundles tracked the wrong object

`tools/track3/check_identity.py` projects every bundle's mesh onto frame 0 of its video. The numeric checks in `numeric_qa.json` pass on all of these; it is a visual check (`mesh_identity_visual_check: PENDING` in every bundle).

| Episode | Bundle | What it actually tracked | Use |
|---|---|---|---|
| **19** (scored) | mini_sweeper | the dust pan | Relabel as `mini_dust_pan`, the episode's other object; redo the sweeper |
| **30** (scored) | mini_sweeper | the dish rack (70 cm) | Relabel as `plastic_dish_rack`, the episode's other object; redo the sweeper |
| **25** (scored) | wooden_piece_2 | a 67 × 86 × 87 cm blob over the whole table | Redo |
| 18 (public) | mini_sweeper | the dust pan (both ep18 runs track it) | Redo |
| 23, 24 (public) | both wooden pieces | 50–90 cm blobs | Redo |
| 0, 2 (public) | white_pot | the lid area (the lid sits on the pot) | Ambiguous |

All other scored bundles track the right object: 1, 3, 5, 6, 8, 10, 14, 15, 16, 20, 28, 32, 33 (both), 35–38.

The failing prompts were `'mini sweeper'`, which locks onto the bigger dust pan or rack (it worked once, on ep31), and `'wooden board'`. A more specific prompt such as "small wooden hand brush with bristles" is the first thing to try.

## Safety-net runs in flight

Still scene + `--level` for every scored episode with complete bundles: ep35, ep36, ep37 (spoon), ep38 (dust pan, scan-rescaled) and ep33 merged (cup + spoon). All five finished and were evaluated (helper force 0, as on Kaggle).

| Run | Object error vs reference | AUC vs reference | Object turn over the clip | Training: object lost |
|---|---|---|---|---|
| ep37_still (spoon) | 0.1 cm / 0.3° | 1.000 | 0.2° | 0% |
| ep38_still (dust pan) | 0.0 cm / 0.0° | 1.000 | 0.1° | 0% |
| ep33_still (cup + spoon) | 0.6 cm / 2.5° | 1.000 | cup 0.0°, spoon 7° | 3% |
| ep35_still (spoon) | 0.4 cm / 13° | 1.000 | 12°, all in the first 5 frames, then still | 35% |
| ep36_still (spoon) | 0.7 cm / 16° | 0.956 | rocks 4–12°, then turns to 66° and moves 2.8 cm over the last third | 71% |

- **ep37, ep38 and ep33 are as good as a still object can be.** ep41_still_level (0.1 cm, local AUC 0.356) showed that this is exactly what a perfectly still object scores.
- **The ep35 and ep36 spoons move without being touched.** The reference hands stay at least 25 cm away for the whole clip, and ep35's turn happens within 0.25 s, before a hand could get there. Both spoon meshes are poor (ep35: 28 × 21 × 13 cm, ep36: 33 × 13 × 22 cm, nothing like a spoon), so they rock on their support.
  - ep35: a constant 12° offset costs roughly 1 cm per point on that one episode.
  - ep36: only the last third of the clip is off.
  - Each is at most a few hundredths of AUC on one of 20 episodes, so I've left them for now. If needed, the fix is to start the spoon in the pose the simulator settles it into.

## Why the grasp is lost (found this morning)

The evaluator always runs with the helper force at 0 (`flash_chord/evaluation/harness.py:230`), so a policy must carry the object on its own. Twice the training (500M steps) made things worse, so the problem is in the reference.

**The reference hands often don't touch the object while it is lifted.** Lifted frames on which no robot finger is within 3 cm of the object:

| Scene | Lifted frames with no finger within 3 cm |
|---|---|
| NVIDIA tissue sample | 0% |
| ep41 | 17% |
| ep38 | 28% |
| ep33 cup / spoon | 9% / 58% |
| ep37 | 100% |

The reconstruction fits hands and object separately, so a reference object often floats next to the hand. A policy can't copy a carry that nobody holds.

**Two possible repairs; only one works:**
- *Move the object onto the hand:* worse. Reconstruction-only AUC 0.291 → 0.265, ep41 0.432 → 0.290. Our object tracks are better than our hand tracks.
- *Move the hands onto the object* (new `--attach-hands`): while the object is lifted, each hand that comes within 12 cm of it rides rigidly with it, at the offset from its closest frame. Hands aren't scored, so the score can't drop. Lifted frames with no finger within 3 cm:

| Scene | Before | After |
|---|---|---|
| ep41 | 17% | 3% |
| ep38 | 28% | 16% |
| ep33 cup | 9% | 2% |
| ep33 spoon | 58% | 14% |
| ep37 | 100% | 94% (the spoon "lifts" with no hand within 9 cm: a tracking error) |

Training on ep41, ep38, ep37 and ep33 with attached hands is queued.

**Results (all evaluated, helper force 0):**

| Run | Object error vs reference | AUC vs reference | Training: object lost | Local AUC vs truth |
|---|---|---|---|---|
| ep38_scan_level → **ep38_attach** | 8.8 cm / 25° → **0.6 cm / 1.4°** | 0.461 → **0.988** | 88% → 8% | |
| ep41_scan_level → **ep41_attach** | 10.1 cm / 31° → 9.7 cm / 18° | 0.324 → **0.743** | 91% → 95% | 0.243 → **0.356** |
| ep33_merged_level → ep33_attach | 11.7 cm / 75° → 9.4 cm / 61° | 0.264 → 0.300 | 98% → 95% | |
| ep37_level → ep37_attach | 10.5 cm / 12° → 10.5 cm / 13° | 0.503 → 0.501 | 84% → 84% | |

- **Attaching the hands is the biggest single gain so far.**
  - ep38's dust pan is now tracked almost perfectly.
  - ep41 more than doubles its AUC against the reference.
  - ep41's local score against the truth rises to 0.356: it ties the still pot, and the reconstruction's ceiling is 0.432.
- ep37 is unchanged as expected, because its "lift" is a tracking error.

**Also tried, no gain:** placing the second object of a two-object scene by sliding it along its camera ray onto the table under the first. Mean still AUC 0.303 → 0.301, so placement is not just a depth error.

## What we changed

**`--level`** (stage 2, `run_load_retarget.py`):
- For each object, it finds the nearest stable resting pose of the mesh (`trimesh` stable poses).
- Tilts up to 20° are treated as a world gravity error: the whole scene is rotated about the object, the same as NVIDIA's `level_result_bundle.py`, so hands keep their grip.
- Larger tilts are an error in the object's own pose: rotating the world by ep36's 27° left the spoon 10 cm above its table. So the object is rotated in its own frame instead.
- Results: frame-0 seating 0.0 cm on ep41, ep37, ep38 and ep33; reconstruction-only score unchanged (ep41 AUC 0.432 both ways).

## Decisions for you

1. **Submit the still safety net once all 20 episodes have both objects?** For a still submission, a missing second object only needs a mesh and its start position, not good tracking.
2. **Rules question (still open):** may we use the public scans' size to rescale our scored-episode reconstructions? The wooden spoon has no scan, and the scored episodes reuse the public objects.
3. **Per-object choice between still and tracked.** If Kaggle's scored episodes behave like the public ones, the best submission keeps most objects still and tracks only where our reconstruction is clearly good. Tracking only helps once trained policies get close to the reconstruction's score.

## Next

- Run the attached-hands policies (queued) and compare them with the levelled ones above.
- Close the remaining finger gap: attach holds each hand at its *closest* offset, which can still be a few cm.
- Run still + level on each new bundle as the missing second objects arrive.
