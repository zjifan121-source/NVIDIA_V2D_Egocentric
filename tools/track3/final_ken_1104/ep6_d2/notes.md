# ep6 refine, round 2 (D2 spoon de-smoothing), 4 Nov 2026 12:15-12:45 EST

**Outcome: built, but failed one pre-registered gate.** Nothing was installed or queued, and ep6 stays on bob1 (`eb1_ep6`, `ep6_bm_s42` @100M).
- The D2 spoon track passed every pre-build gate, G1 to G5.
- The built scene failed **G6(c)**, the check that the right-hand grip stays as in bob1.
- Details are in `FAILED.json`. Gates were pre-registered in `gates.json` (Babel copy written 12:22:35, before any D2 code existed; md5 48e4435c).

## What D2 does (rule and diagnosis in gates.json)
D2 keeps bob1's spoon mesh and its 15.4 deg levelling. On f36-171 it multiplies a world-frame correction onto bob1's poses. The correction is built in five steps:
1. Take Ken's FoundationPose smoothed-to-raw motion (round-1 N1).
2. Reject frames more than 30 deg off (13 roll flips) and interpolate across them (N2).
3. Smooth the motion with a Gaussian, sigma 1.5. The sigma was picked by held-out cost among the values that pass the step limits.
4. Choose per frame between the corrected pose and bob1's pose by the round-1 mask/depth cost. The corrected pose was kept on 108 of 124 frames.
5. Smooth the choice and ramp it in and out over 6 frames.

## Gates
| Gate | Result | Numbers |
|---|---|---|
| G1: held-out parity CV, relative to this mesh | pass | IoU_occ 0.332 -> **0.433**. Bar: 0.85 x rest-fit 0.487 = 0.414. Gain +0.101; 1.6% of frames worse. |
| G1s: strict CV | pass | 0.425 (+0.093); 2.4% of frames worse. |
| G2: depth | pass | \|FS depth\| 1.79 -> 0.91 cm (drop 0.88 cm). |
| G3: steps, f35-172 | pass (tight) | Rotation max 14.95 / p90 7.14 deg; translation max 2.91 / p90 1.626 cm. Limits: 15 / 10.6 deg and 3 / 1.63 cm. |
| G4: kit roll and SR margins | pass | Kit self-registration 0.0 deg (0.29 deg resampled). Worst frame vs bob1: 15.4 deg, roll 15.3 deg. Episode means: 2.9 deg, 0.88 cm. |
| G5: raw right hand | pass | 0.254 -> 0.234 cm. |
| G6(a,b,d,e): built scene | pass | Pot, left hand, and spoon outside f36-171 match bob1 bit for bit. Spoon matches the targets to 3e-8. Interpenetration 0.19 cm (bob1 0.20). Task error unchanged. |
| **G6(c): grip** | **FAIL** | attached_1 right [32,185] at **1.6 cm vs bob1 2.2 cm** (limit 0.5 cm). |

## Why G6(c) matters (new finding)
The team attach rule anchors the grip at the hold frame where the unmoved wrist is closest to the spoon. The de-smoothed spoon moved that frame, so the hand now holds the spoon about 4.0 cm / 21.4 deg away from bob1's grip for the whole hold.

This includes **f172-185, where the spoon is unchanged**: there the right wrist is 4.05 cm / 21.4 deg off bob1. The spoon fix silently rewrote the hand reference outside its own window.

## Next round: D2b (design ready, not built)
Keep the same spoon targets (`d2_targets.npz` md5 19c851c7), so G1-G5 carry over unchanged. In the patch:
1. Call the original attach on bob1's unmodified tracks. This gives bob1's grip at 2.2 cm.
2. Restore the left hand.
3. Write the D2 spoon frames.
4. Move the right wrist rigidly with the spoon correction, H' = C_b(t) H(t), and write it back through the camera-frame MANO parameters as attach_hands does.

Preview: the hand shifts by a median of 1.7 cm (max 5.1 cm) on f36-171 and by 0 outside. The grip equals bob1's by construction.

New G6(c): the wrist in the spoon frame equals bob1's within 1e-4 m on every hold frame. Estimated time: about 20 min plus training.

## Other information
- **Video motion.** Correlation between the mask's per-frame image motion and the rendered spoon's motion:
  - x: 0.03 (bob1) -> 0.39 (D2)
  - y: 0.89 -> 0.96
- **Tilt.** The depth-gradient slope grows from -0.154 to -0.199. This is the round-1 N3 over-tilt and remains a judge criterion.
- **Judge pseudo-truth:** `work/out/pseudo_truth_V1_scene_world.npy`.

## Files
- Local: `C:\v2d_demo\diag\refine1104\ep6_d2\`
  - `gates.json`, `FAILED.json`, `notes.md`
  - `scripts\`
  - `babel_out\`: d2_result.json, g6_result.json, g6_extra.json, interpen_*.json, logs, d2_targets.npz, pseudo-truth, build.log, ep6_fixpatch.py, report.json
- Babel: `/data/user_data/haisenk/v2d/refine1104/ep6_d2/`
  - The built scene `scene/rf_ep6_d2` is **not endorsed**; it has a marker file beside it.
