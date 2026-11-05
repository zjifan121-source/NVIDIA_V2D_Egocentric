set -u; D=/data/user_data/haisenk/v2d/refine1104/ep6_d2; date
[ -e $D/FAILED.json ] && { echo "FAILED.json exists - not overwriting"; exit 3; }
cat > $D/FAILED.json << '__F_EOF__'
{
  "episode": 6,
  "round": "refine1104 ep6 round 2 (design D2)",
  "status": "failed",
  "written": "2026-11-04 ~12:45 EST (refine1104 ep6_d2 agent)",
  "decision": "Not installed, not queued. ep6 stays on bob1 (scene eb1_ep6, policy ep6_bm_s42 @100M). The D2 spoon track passed every pre-build gate (G1-G5), but the built scene failed G6(c): when the unchanged team attach rule re-attached the right hand to the de-smoothed spoon, it picked a different grip frame. The pre-registered limit on the grip-distance change was 0.5 cm and the measured change was 0.6 cm. Behind that number, the right wrist sits 4.0 cm / 21.4 deg off bob1's grip relative to the spoon for the whole hold f32-185, including f172-185, where the spoon itself is unchanged. No override.",
  "gates_file": "C:\\v2d_demo\\diag\\refine1104\\ep6_d2\\gates.json (md5 48e4435c; Babel copy /data/user_data/haisenk/v2d/refine1104/ep6_d2/gates.json written 12:22:35 EST, before d2.py existed)",
  "design": {
    "name": "D2 = data-driven de-smoothing of bob1's in-pot spoon; keeps bob1's spoon mesh and levelling",
    "diagnosis": "C:\\v2d_demo\\diag\\refine1104\\ep6\\FAILED.json N1 (the stir was erased by FoundationPose zero-phase smoothing), N2 (raw FP roll flips and jitter), N4 (the motion transfers onto our mesh), N5 (our mesh caps IoU_occ at ~0.49, hence mesh-relative bars), T7 lesson (no mesh swap; kit roll check)",
    "rule": "C(f) = Ken's ken_c_scored smoothed->raw FP spoon motion in the scene world (rotation about his smoothed centroid + centroid displacement) -> reject frames > 30 deg (13 frames: f41, 56, 57, 63, 71, 121-124, 132-135) and interpolate -> Gaussian sigma s -> candidate = C_s(f) left-multiplied onto bob1's levelled reference on W = f42-165 -> per frame keep the candidate only if its round-1 mask/depth cost (1 - IoU_occ + 0.04*min(|dep|, 6)) beats bob1's -> choice smoothed with sigma s -> 6-frame ramps (f36-41, f166-171). s chosen from {1, 1.5, 2}: the lowest held-out CV cost among the s that pass G3 -> s = 1.5 (s = 1 failed G3: rotation step 18.5 deg, translation step 3.26 cm, translation p90 1.72 cm).",
    "chosen_sigma": 1.5,
    "choice": "candidate kept on 108/124 W frames (0.871); bob1 kept at f57, 61-64, 69, 71, 76, 77, 89, 104, 119, 122, 123, 126, 139",
    "scripts": "C:\\v2d_demo\\diag\\refine1104\\ep6_d2\\scripts\\ (d2.py, ep6_fixpatch.py, g6.py, setup.sh, run_d2.sh, build_d2.sh, run_g6.sh, run_g6b.sh, fetch1.sh)"
  },
  "gate_results": {
    "G1_heldout_primary_parity_CV": {"pass": true, "REST": 0.4871, "REST_parts_f0_24_f220_242": [0.5107, 0.4615], "bar_0.85xREST": 0.414, "iou_occ_bob1": 0.3316, "iou_occ_D2": 0.4328, "gain": 0.1012, "ratio_to_rest": 0.889, "frac_worse_gt_0p05": 0.016, "per_block_bob1_vs_D2": {"f42-69": [0.211, 0.334], "f70-99": [0.394, 0.448], "f100-129": [0.347, 0.439], "f130-165": [0.36, 0.492]}},
    "G1s_strict_parity_CV": {"pass": true, "iou_occ_D2": 0.4246, "gain": 0.093, "frac_worse_gt_0p05": 0.024},
    "G2_depth": {"pass": true, "absdep_median_bob1_cm": 1.792, "absdep_median_D2_cm": 0.914, "drop_cm": 0.878},
    "G3_steps_f35_172": {"pass": true, "rot_max_deg": 14.95, "rot_p90_deg": 7.14, "trans_max_cm": 2.913, "trans_p90_cm": 1.626, "limits": "rot max 15 / p90 10.6; trans max 3.0 / p90 1.63", "bob1": {"rot_max": 14.64, "rot_p90": 6.13, "trans_max_cm": 1.496, "trans_p90_cm": 1.162}, "note": "translation p90 passes by 0.004 cm and rotation max by 0.05 deg; sigma 2 had more margin (2.68 cm / 1.56 cm p90) but a worse held-out cost, and the pre-registered rule picks by cost"},
    "G4_kit_roll_and_SR_margins": {"pass": true, "a_kit_self_registration_deg": 0.0, "info_resampled_registration_deg": 0.288, "b_rot_vs_bob1_max_deg": 15.42, "b_roll_about_long_axis_max_deg": 15.25, "at_frame": 136, "c_mean_rot_change_deg_all243": 2.89, "c_mean_centroid_change_cm_all243": 0.875, "in_W_median": "4.78 deg / 1.34 cm, max 5.27 cm"},
    "G5_raw_right_hand": {"pass": true, "changed_frames": 136, "closest_joint_median_bob1_cm": 0.254, "D2_cm": 0.234, "frac_closer": 0.662, "windows": {"f36-69": [0.391, 0.21], "f70-99": [0.31, 0.249], "f100-129": [0.357, 0.26], "f130-171": [0.151, 0.223]}},
    "G6_built_scene": {
      "pass": false,
      "a_bundle_cleaned": {"pass": true, "pot_max_diff": 0.0, "left_hand_max_diff": 0.0, "spoon_outside_f36_171_max_diff": 0.0, "spoon_vs_targets_max": 3.0e-08},
      "b_scene_parquet": {"pass": true, "pot_pos_wxyz_max_diff": 0.0, "left_wrist_max_m": 0.0, "spoon_changed_frames": "f36-171 (136)", "spoon_steps": "rot max 14.95 / p90 7.14, trans max 2.913 / p90 1.626 cm", "right_wrist_shift_max_cm": 6.415},
      "c_report_cleaning": {"pass": false, "attached_1": [["right", 32, 185, 1.6]], "bob1_attached_1": [["right", 32, 185, 2.2]], "grip_distance_diff_cm": 0.6, "limit_cm": 0.5, "attached_0": null, "tilt_levelling": "1.4 / 15.4 / world / object = bob1", "why_it_matters": "the attach rule anchors the grip at the hold frame where the unmoved wrist is closest to the spoon surface; the de-smoothed spoon moved that frame, so the hand-in-spoon offset changed by 4.0 cm median (max 4.9) and 21.4 deg for the whole hold. Right wrist vs bob1 in the world: f32-35 1.7 cm / 9.0 deg; f36-171 4.1 cm / 22.2 deg (max 5.9 cm / 27.6 deg); f172-185, where the spoon is unchanged, 4.05 cm / 21.4 deg"},
      "d_interpen": {"pass": true, "spoon_in_pot_max_cm": 0.192, "bob1": 0.204, "frame": 62, "frames_gt_0p5": 0},
      "e_task_error": {"pass": true, "right_mean_max_cm": [1.93, 5.54], "bob1": [1.93, 5.54], "frame0_seating_cm": [0.0, 0.58]}
    }
  },
  "info": {
    "video_motion_agreement_W": "per-frame image motion of the spoon mask vs the rendered visible silhouette, corr(dx) / corr(dy): bob1 0.033 / 0.887 -> D2 0.389 / 0.957 (in-sample; information only)",
    "depth_gradient_slope_W": "bob1 -0.154 -> D2 -0.199 (held-out -0.199). The levelling over-tilt (round-1 N3) grows slightly when the stir is restored; it stays a judge criterion (|slope| growth <= 0.05)",
    "no_new_interpenetration": "spoon-in-pot 0.19 cm max (bob1 0.20), pot-in-spoon 0.28 cm (bob1 0.28), both only at f62",
    "pseudo_truth_for_judging": "/data/user_data/haisenk/v2d/refine1104/ep6_d2/work/out/pseudo_truth_V1_scene_world.npy (round-1 V1 construction: flip-rejected sigma-1 raw FP motion on bob1's mesh, no per-frame choice)"
  },
  "built_but_not_endorsed": "/data/user_data/haisenk/v2d/refine1104/ep6_d2/scene/rf_ep6_d2. Built in 354 s at 12:33 EST, report.json OK. Not installed in job 10774619. Marker: scene/rf_ep6_d2.FAILED_G6c_NOT_INSTALLED.txt",
  "next_round_D2b_ready_design": {
    "change": "Keep the D2 spoon track exactly: the same targets, d2_targets.npz md5 19c851c7, so G1-G5 carry over unchanged. Change only how the right hand follows. Do not re-run the attach rule on the moved spoon. In the H5_PATCH wrapper: (1) call the original attach_hands on bob1's unmodified tracks, which reproduces bob1's right attach [32,185] at 2.2 cm and its grip; (2) restore hand_left_* (EP6_NO_LEFT_ATTACH=1); (3) write the D2 spoon frames f36-171; (4) on those frames move the right wrist rigidly with the spoon correction: H_new(t) = C_b(t) H(t), with C_b(t) = T_D2_b(t) T_bob1_b(t)^-1 in the bundle frame. Write the result back through hand_right_wrist_orient_in_camera / trans_in_camera (with the MANO j0 offset) and the wrist_to_camera / world transforms, exactly as attach_hands does.",
    "preview_numbers": "right wrist shift vs bob1: f36-171 median 1.70 cm, max 5.14 cm (the stir); outside f36-171 2e-14 cm; grip relative to the spoon = bob1's on every frame by construction",
    "gates_to_preregister": "G1-G5 as here (identical spoon track). G6 as here, with (c) replaced by: right wrist in the spoon frame equal to bob1's within 1e-4 m / 0.01 deg on every hold frame f32-185; right-hand arrays bit-identical to bob1 outside f36-171; attached_1 = bob1's. Keep (a), (b), (d), (e).",
    "cost": "about 10 min to write the patch, 6 min to build, 4 min for checks, then install and 3 KEN seeds"
  },
  "judge_plan_if_built_later": "As pre-registered in gates.json judge_plan_preregistered. Compare rf_ep6_d2b_K_s42/43/44 (checkpoint per the team own-ref sweep >= 70M) with bob1 ep6_bm_s42 @100M on four judges. J1: round-1 D6 video judges on the rollout, per phase. J2: anchored own-ref against the FIXED bob1 reference, inside f36-171 and outside it. J3: AUC against the V1 de-smoothed pseudo-truth on f42-165. J4: rollout spoon roll and margins against the bob1 rollout. Adopt only if all of these hold in at least 2 of 3 seeds: in-pot spoon IoU_occ >= bob1 + 0.03 (>= 0.317); |FS depth| not worse by more than 0.1 cm; |slope| growth <= 0.05; J3 in-window AUC above bob1's; pickup and rest IoU not lower by more than 0.02; J2 outside-window AUC within 0.01; pot IoU within 0.02; rotation vs the bob1 rollout <= 30 deg on every frame (roll included); mean change <= 10 deg / 2 cm; no rotation step > 15 deg/frame; no terminations. No Kaggle.",
  "outputs": {
    "local": "C:\\v2d_demo\\diag\\refine1104\\ep6_d2\\ (gates.json, FAILED.json, notes.md, scripts\\, babel_out\\ = copies of the Babel work/out JSONs, logs, d2_targets.npz, pseudo-truth, build.log, the patch, report.json)",
    "babel": "/data/user_data/haisenk/v2d/refine1104/ep6_d2/ (gates.json, FAILED.json, notes.md, work/ = d2.py g6.py kit_AUC.py interpen.py and out/, build/ = b_d2 bundle copy, repo copy, ep6_fixpatch.py md5 f6f63f3b, d2_targets.npz md5 19c851c7, build.log, scene/rf_ep6_d2 NOT endorsed)"
  },
  "safety": "GT-free: no scans, GT meshes or GT poses. No Kaggle, HF, git. Nothing outside my own dirs was modified or deleted. Inputs were copied, not edited: round-1 fit6.py, d4_rawfp.json and f1d_joints_cam.npz; b_ctl and the repo tree from /scratch/haisenk-v2d/diag/ep6_plan on t9-28; the kit AUC.py from the local copy. CPU only, in job 11035728 (babel-t9-28); CUDA_VISIBLE_DEVICES=1 was set and no GPU was used. Job 10774619 was not touched: no install, no queue."
}
__F_EOF__
cat > $D/notes.md << '__N_EOF__'
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
__N_EOF__
python3 -c "import json;json.load(open('$D/FAILED.json'));print('FAILED.json ok')"
md5sum $D/gates.json $D/FAILED.json $D/notes.md $D/build/d2_targets.npz $D/build/ep6_fixpatch.py | cut -c1-8,33-
ls -la $D $D/scene; date
