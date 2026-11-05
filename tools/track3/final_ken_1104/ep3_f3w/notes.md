# ep3 round 2: F3w (4 Nov 2026, 12:15-12:35 EST)

**Outcome: FAILED on the pre-registered gate G7 (kit-registration roll consistency).** Nothing was installed or queued, and no gate was overridden.

Every other gate passed:

- G1-G6 from round 1, unchanged
- the new twist-continuity gate (G4)

See `gates.json` (written before the build), `gate_results.json` and `FAILED.json`.

## What F3w is (from `gates.json`)

F3w is a direct edit of a copy of the adopted scene sc_ep3 (T5 = h_ep3_sc_K_s42 @95M). It only touches round 1's F3 windows:

- weights `w` from round 1's `f3_select.npz`
- runs f51-61, f86-114 and f122-200, plus 8-frame ramps, so f43-208 are edited

Inside those windows:

- **Position and long axis.** The spoon takes its centroid and its long axis from Wentao's raw track W. The long axis is the principal axis of the mesh vertices, the same definition as round 1's d2.py.
- **Twist.** The twist about that axis comes from the observed right wrist, combined with the hold's chordal-mean grip:
  - The wrist is bundle_merged with the build's smooth_hand(sigma=2), replicated as in `ep3_skB3\skB3.py`.
  - Gm is the chordal mean of the spoon-in-wrist rotation over the right hold [43,266) from report.json, measured with W.
  - The wrist-predicted axis is rotated onto W's axis by the minimal rotation. The median correction is 14.8 deg (max 30.4).
- **Blend.** The 8-frame ramps SLERP from the adopted track to F3w.
- **Levelling rule.** The levelling is applied inside the windows only if round 1 showed it raises IoU. Round 1 measured W 0.746 -> WL 0.618, so it is not applied.
  - L is 7.93 deg in total: 6.85 deg of it is twist about the long axis (now replaced by the wrist twist) and 3.99 deg is axis tilt (undone in the windows).
- **Right hand.** The hand is moved rigidly with the spoon, M = S_new inv(S_old), only on edited frames inside the hold (all of them are inside it). This covers 11 right-hand columns.
  - Entry 0 of robot_right_frames ('universe') is kept.
  - mano_right_scale_center is kept, because by the stated rule it is not a world point (median 1.13 m from the wrist).

## Gate results

| Gate | F3w | Scene (sc_ep3) | Threshold | |
|---|---|---|---|---|
| G1a IoU f130-185 | 0.743 | 0.390 | >= 0.65 | pass |
| G1b IoU f85-129 | 0.727 | 0.527 | >= 0.65 | pass |
| G1c IoU episode | 0.719 | 0.591 | >= 0.70 | pass |
| G1d IoU P0 / P1 | 0.718 / 0.714 | 0.718 / 0.600 | not below scene - 0.02 | pass |
| G2 depth \|median\|, switched frames | 4.8 mm | 12.5 mm | <= scene + 5 mm | pass |
| G3 spoon-pot penetration frames | 0 | 0 | <= scene | pass |
| G4 frames > 10 deg/frame, f50-255 | 19 (twist-only 4) | 23 (Wentao 45) | <= 30 | pass |
| G5 diff / scope | spoon + 11 right-hand columns, f43-208 only; hand-in-spoon to 1e-7; files identical | | | pass |
| G6 right contact ids | identical | | | pass |
| G7 kit-registration roll | median 68.6 deg, \|twist\| 31.7 deg | OC 77.4 / 55.7; W 74.5 / 44.2 | both < 30 | **FAIL** |

## Why G7 fails, and what that does and does not say

How the check ran (`g7_detail.out`):

- On every 2nd switched frame (60 frames, all usable), the SAM 3-masked FoundationStereo depth was turned into a camera-frame cloud.
- The spoon mesh was registered to it with AUC.py's own `_align_rigid`: PCA starts plus trimmed ICP, best start by symmetric distance.

What came out:

- **The registration is bimodal.** 23 of 60 frames sit near a 180 deg flip, and the median residual is 13.9 mm. One-sided, hand-occluded clouds of a thin spoon do not pin down the roll.
- **The adopted, Kaggle-proven track fails the same check more badly.** It scores 77.4 / 55.7 against F3w's 68.6 / 31.7.
- **On the frames that did not flip (descriptive only, not a gate),** the median angle / |twist| is:

  | Track | Angle | \|Twist\| |
  |---|---|---|
  | NEW (F3w) | 20.8 | 18.7 |
  | OC (adopted) | 36.8 | 31.0 |
  | W (Wentao) | 15.6 | 11.1 |

So, as pre-registered, the check cannot tell a good roll from a bad one, and the T7 risk stays open. The gate stands anyway.

## What would be needed (Ken's OK and a fresh pre-registration)

1. **A well-posed roll check.** For example:
   - Run local trimmed ICP from each track's own pose, with no PCA flip starts.
   - Use a one-sided cloud->mesh distance.
   - Pass if F3w's final roll offset is no worse than the adopted track's on the same frames, and below 30 deg on frames whose converged residual is under 5 mm.
2. **Then build F3w unchanged.** The build is deterministic: `build_f3w.py` regenerates the parquet with md5 a026149e. Queue only if the new check also passes.

## Judge plan (unchanged; applies only if F3w is ever trained)

- **Rollout to camera frame.** Run round-1 `l3_roll.py` on the new rollout, with the FIXED sc_ep3 reference.
- **Proxy scores.** Run `l4_proxy.py` with three pseudo-truths:

  | Pseudo-truth | T5 |
  |---|---|
  | own-ref (OC) | 0.9477 |
  | Wentao in the windows | 0.9149 |
  | Wentao everywhere | 0.9092 |

- **Anchored own-ref.** Run `anch.py` with the m_sc_ep3 meshes against sc_ep3's reference.
- **Video check.** Run round-1 `d2.py` for rollout IoU and depth per window.

**Adopt only if both of these hold:**
- The median over s42/43/44 beats T5 under all three pseudo-truths.
- It keeps the SP-SR/MP-SR pass margins. Under every pseudo-truth where T5 passes, the new run passes too, and the spoon MP-SR margins are not smaller. Pot IoU per window must be within 0.02 of T5.

## Files

Local `C:\v2d_demo\diag\refine1104\ep3_f3w\`:
- `gates.json`, `build_f3w.py`, `build_report.json`, `f3w_track.npz`
- `gate_results.json`, `FAILED.json`, `g7_detail.out`
- `babel\d3.py`, `babel\d3_run.sh(.out)`, `fromBabel\d3.json`, `fromBabel\d3.out`
- `build\rf_ep3_f3w`: the built scene copy. It is NOT installed.

Babel `/data/user_data/haisenk/v2d/refine1104/ep3_f3w/`:
- `gates.json`, `build_report.json`, `code/build_f3w.py`
- `diag/{d3.py, AUC.py, f3w_track.npz, d3.out, d3.json}`
- `FAILED.json`, `notes.md`
