# Path B v2: physics-grounded video inversion

Created Oct 19, 2026 · last updated Oct 20, 2026 · exported from the [Claude doc](https://claude.ai/code/artifact/afa70a10-81ae-43e2-a850-0549839a491a), which is the canonical copy

Track 3 scores a simulated rollout, so Path B v2 treats the whole task as one inference problem: find the physically executable demonstration that best explains the ego video, enforced by the same physics the kit's evaluator runs, with the FlashCHORD policy trained inside the loop. Approved by James on 2026-10-19 as the plan to the Nov 1 deadline.

## TL;DR

Today's chain is open-loop: perception → repairs → retarget → RL → rollout, and each stage hands the next a reference it was never asked to make feasible. Path B v2 inverts the order of authority: the simulator is the model of the world, the demonstration (hands, object motion, mesh, support, contacts) is the hidden state, and the video is evidence. Perception becomes an observation model; it no longer decides anything.

- **What is scored** is a rollout: the FAQ requires "object trajectories from learned policies" and welcomes other embodiments. The kit's evaluator (`flash_chord/evaluation/harness.py`) accepts only FlashCHORD `flash_sac` or `ppo` checkpoints, and using it is the team's decision (James, 2026-10-19). Either way the rollout is the quantity the loop optimizes.
- **Physics constrains the reference; it does not replace cleaning it.** A demonstration realized in MuJoCo Warp is feasible by construction: the object moves only through contact and rests on its support otherwise. It is not correct by construction: SPIDER warns that a misaligned reference "may cause the optimizer to converge to a semantically incorrect contact mode despite physical consistency" (§4.5), and CHORD that "effective contact guidance requires relatively clean human demonstrations" (§6). So stage 2's cleaning, rest hold and levelling stay as pre-processing of the kinematic reference, and S4 adds feasibility on top.
- **Same solver family, both ends, with a transfer check.** The kit's evaluator runs on Newton, whose solver is MuJoCo Warp; SPIDER's physics retargeting runs on MuJoCo Warp with a 22-DoF floating Sharpa hand, the joint count of FlashCHORD's Sharpa Wave URDF. Settings differ (FlashCHORD RL: sim 0.01 s, control 0.05 s, `njmax` 512; SPIDER with guidance: sim 0.005 s, control 0.05 s), so feasibility is not assumed to transfer: gate 0 replays S4's output in FlashCHORD's physical replay and records the object error.
- **Change from Path B v1:** v1 fixed scale (stereo) and planned VLM boxes and SAM 3 masks for Stage 2. v2 keeps v1's P0–P4 as the evidence layer and puts physics-consistent demonstration inference plus policy-in-the-loop in place of Stages 2–3. The learned egocentric 4D hand-object model (block 1 in full) is after Nov 1.

## Why not better stages

Our own numbers put the loss between stages, not inside one: a policy tracking a jittery, contact-inconsistent reference scored below doing nothing.

| Measurement (19 of the 20 public episodes unless noted; ep24 is excluded because P3 errored on it) | AUC | Source |
| --- | --- | --- |
| Team bundles played back | 0.270 | Path B waterfall, run `0fa525bc` |
| + stage 2 repairs (rest hold, cleaning, levelling) | 0.314 | same |
| + Path B stereo scale | 0.420 | same |
| Residual after the scan-oracle scale (tracking, mesh, placement) | ≈0.6 of the scale | same |
| A still object: our two-object placement vs the true placement (20 episodes) | 0.309 vs 0.447 | public dry run `ef058c7e` |
| ep41: trained tracking policy / playback / **still policy** | 0.243 / 0.287 / **0.356** | V2D\_PROJECT\_CONTEXT.md §4 |

CHORD's own paper says why: "effective contact guidance requires relatively clean human demonstrations" (§6), and in its noise study "when noise is additionally introduced into contact guidance, performance decreases more noticeably" than with hand-pose noise (Appendix A.1) ([CHORD](https://arxiv.org/html/2607.00033v2)). A rollout is terminated at 15 cm / 40° object error in the paper (§4.1) and at **10 cm / 0.5 rad** (wrist 15 cm) in the kit's Sharpa recipe (`configs/termination/tracking.yaml`). The repairs the team bolts on (rest hold, levelling, attach-hands, seat, merge) are hand-written physics; v2 keeps them as pre-processing and lets the simulator decide what is feasible.

## Architecture

*Architecture diagram (ego video → 1 evidence → 2 object model → 3 physics-consistent demonstration → 4 policy in the loop → kit evaluator, with the mesh route to CD-O and the post-Nov 1 feedback loop) is in the [live doc](https://claude.ai/code/artifact/afa70a10-81ae-43e2-a850-0549839a491a).*

Block 3 (accented) is where the chain stops being open-loop: the demonstration handed to the policy is realized in the same physics the evaluator runs. The dashed loop, rollout failures fed back into block 3, is the part deferred past Nov 1.

## Evidence

The kit's two success metrics are named after ManipTrans and SPIDER, so the organizers' own lineage is physics retargeting plus residual RL.

| Claim | Evidence |
| --- | --- |
| The organizers' lineage is physics retargeting + residual RL | FAQ names "Maniptrans Success Rate (MP-SR)" and "Spider Success Rate (SP-SR)". [ManipTrans](https://arxiv.org/abs/2503.21860) (CVPR 2025) "pre-trains a generalist trajectory imitator… then fine-tunes a specific residual module under interaction constraints"; [SPIDER](https://arxiv.org/html/2511.09484v3) does "physics-based sampling with curriculum-style virtual contact guidance", "10x faster than RL baselines"; its [code](https://github.com/facebookresearch/spider) runs on MuJoCo Warp with the Sharpa hand and has processors for HOT3D, OakInk, OakInk-v2, Arctic-v2 and GigaHands, among others |
| Blocks 3 and 4 share a solver family with the evaluator, not its settings | FlashCHORD README: "GPU-parallel robotic reference tracking with Newton, Warp, JAX, PPO, and FlashSAC"; its code imports `mujoco_warp`; `objectives/` holds `contact.py`, `wrench.py`, `tracking.py`. FlashCHORD RL sim: 20 fps × 5 substeps, `njmax` 512 (`configs/sim/rl.yaml`); SPIDER with guidance: sim 0.005 s, control 0.05 s (Table 5) |
| SPIDER's own Sharpa results are weak, so embodiment is a risk of its own | SPIDER Table 3 (ADD-AUC, seed 0, failed grasps included): HOT3D Allegro 71.2, XHand 70.1, Inspire 69.4, **Sharpa 40.7**; aggregated 67.9 / 66.8 / 66.4 / **44.5**. SPIDER's Sharpa is a floating hand with 22 finger control dimensions (A.4); FlashCHORD's Sharpa Wave URDF has 22 revolute joints per hand |
| Physics in the loop removes jitter, flips and penetration at the source | [SimPoE](https://arxiv.org/abs/2104.00683) (CVPR 2021): a simulated character driven from video, state-of-the-art accuracy with physically plausible motion; [HOIC](https://dl.acm.org/doi/10.1145/3641519.3657505) (SIGGRAPH 2024): RL plus a surface contact model to reconstruct hand-object interaction; [ViTaM-D](https://openaccess.thecvf.com/content/ICCV2025/papers/Yu_Dynamic_Reconstruction_of_Hand-Object_Interaction_with_Distributed_Force-aware_Contact_Representation_ICCV_2025_paper.pdf) (ICCV 2025): force-aware contact optimization that removes penetrations. All three from their abstracts |
| The rollout should be the optimized object | [DexTrack](https://arxiv.org/abs/2502.09614) (ICLR 2025): per-trajectory homotopy optimization with the learned controller, "over a 10% improvement in success rates"; [DexMachina](https://arxiv.org/abs/2505.24853): "virtual object controllers with decaying strength"; FlashCHORD ships the same idea as a `voc_scale` curriculum (ten stages, 1.0 → 0.75 → 0.5 → 0.25 → … → 0, `configs/curriculum/`) |
| A learned ego 4D model can be trained on data that matches our rig (after Nov 1) | [HOT3D](https://arxiv.org/abs/2411.19167) (CVPR 2025): 833 minutes, 19 subjects, 33 objects, marker-based ground truth, recorded on Aria glasses (RGB plus mono stereo cameras, like our OAK rig); multi-view beats single-view on its benchmarks. FAQ: pretrained models and external data are allowed in all tracks |
| Metric scale comes from stereo, not from priors | Path B P3: within 3% of the scan oracle on ep41, +0.106 AUC over 19 public episodes; [VideoMimic](https://arxiv.org/abs/2505.03729) (arXiv 2025) had only monocular video: "the metric human height prior in the SMPL body models serves as the metric reference" (§3.2) |
| Video → sim → policy is a proven pattern | VideoMimic: 4D human and scene from phone video, RL tracking in IsaacGym, a distilled policy deployed on "a 23-DoF Unitree G1 humanoid" (§5.2) |

## What each block buys on the six metrics

| Metric | Today's loss | Block that removes it |
| --- | --- | --- |
| AUC, MPPE | scale (fixed in v1), flips, drift while held | 3: the object moves only under contact; 1: stereo scale |
| MP-SR (3 cm / 30°) | 0.158 now; jitter and early termination | 4: a policy trained on a demonstration it can execute |
| SP-SR (10 cm / 0.5 rad) | 0.474 now; lost on flips | 3 |
| RPE (two-object) | our placement costs 0.14 AUC | 3: one scene with a support graph, the anchor object at frame 0 |
| CD-O | 0.74 cm; meshes 0.5–0.75× the scan after rescaling | 2 |

## Pipeline S1–S7 (the Nov 1 experiment)

S1–S2 exist (Path B Stage 1); S3–S7 are new. Every step writes a `report.json` in the Path B format, so `loss_report.md` keeps naming the first failing step.

### S1. Evidence from the video

**How (built: P0–P2, `path_b/pathb/stages/`):**

1. **P0 ingest:** decode `ego_cam_a/b/c`; frame counts and resolutions must match the manifest and the calibration.
2. **P1 calibrate:** rectify the head pair (left `ego_cam_c`, right `ego_cam_b`) and undistort `ego_cam_a` with the 14-coefficient factory model. The calibration does not say whether its intrinsics are native or upright, so all four readings are tried and the best kept; the per-episode row drift (\~2 px) is removed.
3. **P2 depth:** FoundationStereo (the baseline's `v2d_foundation_stereo`, read-only) on the rectified pairs; the table plane in every frame; the stereo-to-colour frame offset; depth reprojected into `ego_cam_a`.
4. **Bundle:** load the team's bundle (MANO parameters, object pose track, mesh) from HF as noisy observations, never as truth.
5. **Run:** `run.py run --split evaluation` on the L40S, \~5 min per episode, outputs under `path_b/data/outputs/<commit>/`.

**In → out:** `ego_cam_a/b/c`, factory calibration, HF bundle → `rect_left/`, `rect_right/`, `cam_a/`, `calib.json`, `depth_rect/`, `planes.json`, `alignment.json`, `depth_a_half/`, and a `report.json` per stage.

**Check:** row offset after correction < 0.5 px; features within 5 px ≥ 0.5; plane RMS. Nothing new.

| Risk | Handling |
| --- | --- |
| P0–P2 have never run on the 20 evaluation episodes | First run before S2 (\~2 h); `loss_report.md` names the first failing stage, fixed before S2 starts |
| The calibration comes from `public/meta/` (evaluation ships no `meta/`) | Both splits were recorded with the same head rig (`pathb/paths.py`); the rectification bars run per episode, and an episode that fails them is flagged, not used |
| Wrong intrinsics reading | P1 tries all four readings and keeps the one with the lowest row offset |
| Stereo use challenged | NVIDIA approved it (2026-10-18); the organizers were emailed on 2026-10-20 to confirm it in writing |

### S2. Metric scale and scene frame

**How (built: P3–P4):**

1. **P3 scale:** over the frames before a hand reaches an object, divide the camera's height above the table from stereo (the largest horizontal plane at most 35 cm under the object) by the same height in the bundle (its lowest mesh point). That ratio is one scale factor per episode, cross-checked along the ray to the object.
2. **P4 export:** rescale each object's bundle by that factor and write it in the `BUNDLES.md` layout with `mesh_scaled.obj`, fps, the table's world height, `run.json`, `numeric_qa.json` and `status.json`.
3. **Frame:** the support plane becomes the table, with gravity. The scorer aligns each scene on its roster-first object at frame 0, so S2 never moves that pose.

**In → out:** S1 + bundle → `scale.json`, rescaled bundles `episode_XXXXXX/<object>/`, table height, gravity.

**Check:** spread over rest frames < 5%; support plane vs gravity < 8°. Public only: scale vs the ground-truth trajectory scale (the scan-size ratio stays as information).

**Key idea:** scale is the one thing physics cannot recover; everything downstream assumes metres.

| Risk | Handling |
| --- | --- |
| The wrong plane is taken (floor, shelf) | The 35 cm rule plus the gravity check; flagged episodes are reviewed on the `visualize` page before S3 |
| A hand already touches the object at frame 0 (ep24) | Flagged; the opening frames serve as rest |
| Two objects at different heights | The ruler stands on the lowest object; table placement is left to stage 2's `--seat`, which Edmond's still runs needed for ep1, 3, 6, 14, 15, 16, 20 and 30 (`tools/track3/jobs/still_submit.sh`) |
| Ground truth leaks into evaluation | P3 reads ground truth and scans only when the episode is public; the GT firewall (R4) turns any other read into an error |
| One anchor object across the rest frames is still open (P3 fix list) | Fixed before S4 runs at scale |

### S3. Object model

**How (new; SAM 3D Objects access granted 2026-10-20):**

1. **Pass 1 (always):** the bundle mesh at S2's scale (`mesh_scaled.obj` from P4).
2. **Pass 2 (candidates):** route A first: the ego pipeline's SAM3D (`run_ego_reconstruction.py`) at several `--seed` and `--reference_frame` values, which runs as today with no adapter. Route B only if A loses on public CD-O: the HOI module's SAM3D mode (`run_reconstruction.py --mode sam3d --sam3d_capture_mode stationary --sam3d_use_depth`). It is built for object scans and needs an adapter that writes our frames and CuSFM poses into its mapping-data layout, and it runs its own FoundationStereo.
3. **Choose:** scale every candidate on stereo; keep the one that best fits stereo depth and the object silhouettes. Public scans never take part in the choice on evaluation episodes.
4. **Physics model:** `mesh_to_usd` (CoACD colliders; mass from `--mass-kg` = mesh volume × a per-class density) for FlashCHORD, and SPIDER's `decompose.py` (CoACD) for S4. The density table is fixed on the public split before any evaluation run.
5. **Drop test:** `mesh_to_usd/run_drop_test.py` (Isaac Sim), plus a settle test in MuJoCo Warp, the engine family S4 and S6 use.

**In → out:** frames, masks S3 makes itself (the bundles carry none), stereo depth, S2 scale, roster names → `mesh.glb` (visual), convex pieces (collision), USD with mass and inertia, and the mesh's SHA-256 (R3).

**Check:** watertight; number of convex pieces; settles within 1 cm of its frame-0 pose. Public only: CD-O vs the scan and size ratio 0.8–1.25×.

**Key idea:** a mesh that is too small or not watertight makes S4 fail physically before it fails CD-O.

| Risk | Handling |
| --- | --- |
| The HOI module expects scan captures, not egocentric clips | Route A first; route B gets one day for the adapter, only if A loses on public CD-O |
| Single-view SAM3D hallucinates unseen geometry | Several seeds and reference frames; the stereo-fit choice drops candidates that disagree with depth |
| The drop test runs in Isaac Sim, not in MuJoCo Warp | The extra MuJoCo Warp settle test decides; Isaac Sim's result is information |
| A public scan slips into a submission | Never on evaluation (R2, R4); the mesh hash is checked at S7 |
| `wooden_spoon` has no public scan | Health checks only |

### S4. Physics-consistent demonstration (the heart)

**How (new; SPIDER as a pinned dependency):**

1. **Kinematic reference (built, `pathb/spider_scene.py`):** read from stage 2–3's output, the ManoSharpa partition the runner writes after rest hold, cleaning, levelling and the MANO load: world-frame `mano_*_joints` (tips at 4, 8, 12, 16, 20), `object_body_*` poses and the support disks. S4 and S5 therefore share FlashCHORD's world. Two-object episodes reach stage 2–3 through `pathb/merge_runner.py`, the team runner's `--merge` pointed at Path B's P4 bundles (built).
2. **V2D processor (built):** writes SPIDER's processed layout: `trajectory_keypoints.npz` (wrist pose in SPIDER's hand-frame convention, checked against its HOT3D processor to 2e-16; five fingertips; both object slots), `task_info.json`, one `visual.obj` per object. SPIDER ships processors for HOT3D, OakInk, OakInk-v2, Arctic-v2, GigaHands and others, none for V2D.
3. **Scene (built, `stages/s4_scene.py`):** SPIDER's own `generate_xml.py` in bimanual mode already holds two objects (`right_object`, `left_object`, free bodies tracked by the optimizer, colliding with each other and the floor), as its OakInk path uses. The support top under the objects becomes z = 0, so SPIDER's floor is the table; hands collide with it (`--hand-floor-collision`). Each object takes the slot of the hand that touches it most; contact guidance is marked unsafe when one hand handles both. An object resting on neither the table nor the other object is flagged. `pathb/spider_settle.py` then lifts the hands and checks the frame-0 objects drift ≤ 1 cm and turn ≤ 2° in 1 s.
4. **Solve (next):** `decompose.py` (S3's mesh) → `ik_fast.py` → `run_mjwp.py`. Contact guidance on for clean episodes and off for noisy ones; SPIDER itself runs its large-scale generation "without contact guidance" because "noisy demonstrations can make contact signals unreliable" (§4.4).
5. **Export (next):** Sharpa joint trajectories, the physically realized object trajectory as `s4_physics/bundle/result.npz` (the waterfall's `physics` column reads it), the contact sequence and a pass/fail flag per episode.
6. **Environment:** a separate Python 3.12 venv beside the repo (`vm/README.md`). SPIDER's `pyproject.toml` pins mujoco 3.7.0, mujoco-warp 3.7.0.1 and warp-lang 1.12.1, and its `decompose.py` also needs `coacd` and `trimesh`; FlashCHORD pins mujoco 3.8.1, mujoco-warp 3.8.1 and warp-lang 1.14.0.

**In → out:** S2 bundle (keypoints, object poses), S3 collision model, table → robot hand trajectories, physics object trajectory, contacts, success flag.

**Check, four layers:** (0) **gate 0:** replay one S4 output in FlashCHORD's physical replay (`scripts/debug/replay.py`) with `voc_scale=0`, because `configs/replay/physical.yaml` defaults to 1.0 and would push the object along the reference, hiding failures; object error in cm and degrees. (1) Tracking: physics vs kinematic object trajectory, per frame. (2) Validity: penetration depth, zero object motion before first contact, maximum object speed. (3) **Gate 1 (Oct 25):** the `physics` column beats `stereo` on public AUC / MPPE / RPE.

**Runtime:** SPIDER+C runs at 0.4–0.5 FPS (Table 1), about 10 min per 262-frame episode with guidance and seconds without; retries double it on failure (A.3); resampling to SPIDER's 12.5 Hz (Table 6) cuts it by a third.

**Key idea:** one mechanism gives feasibility where rest hold, object cleaning, levelling, attach-hands and seat each patched one symptom. Flips and drift while held cannot happen. Correct contacts still come from the reference, which is why the cleaning stays.

| Risk | Handling |
| --- | --- |
| Two-object scenes (16 of 20 evaluation episodes) depend on SPIDER's bimanual two-slot path in `generate_xml.py` | Adapter and settle test built (2026-10-20); an episode whose settle test fails, or with an unsupported object, keeps stage 2; more than two objects is refused (the roster has at most two) |
| SPIDER's repo has no Sharpa MJCF (its LIFTED release says Sharpa assets "must come from the benchmark code archive"), and its Sharpa may differ from FlashCHORD's Sharpa Wave (Sharpa 40.7 vs Allegro 71.2 on HOT3D) | The scene is validated with SPIDER's XHand first; a Sharpa \`bimanual.xml\` comes from that archive or is converted from FlashCHORD's Sharpa Wave URDF, with SPIDER's \`collision\_hand\_\*\` names. Compare joints, limits and gains with `embodiment/sharpa_hands.yaml`; the Allegro/XHand control at gate 1 separates hand problems from physics; stay on Sharpa and tune gains (James, 2026-10-20) |
| Feasible in SPIDER's settings (sim 0.005 s) but not in FlashCHORD's (0.01 s, `njmax` 512) | Gate 0 before S4 runs at scale; match timesteps where possible |
| A noisy reference leads to the wrong contact mode | Stage 2 cleaning stays; guidance off on noisy episodes |
| Dependency conflict with FlashCHORD | Separate venv; S4 exchanges files only |
| Physics does not beat stereo | Gate 1 drops S4 from the submission path |
| SPIDER is CC BY-NC 4.0 | Pinned dependency, never copied; organizers asked on 2026-10-20 |

### S5. Demonstration → FlashCHORD task

**How (new):**

1. **Mode B (first, on ep41):** write the task parquet in FlashCHORD's unversioned ManoSharpa format (`flash_chord/data/loader.py`) straight from S4: `robot_{left,right}_wrist_position` / `_wxyz`, `robot_*_finger_joints` with their joint names, `robot_*_frames` (forward kinematics of FlashCHORD's Sharpa frames), `object_body_*`, and the contact fields `mano_*_object_contact_positions` / `_normals` / `_part_ids`, filled with the SPIDER hand's contacts against the object. The IK re-solve is skipped, so the hand FlashCHORD imitates is the hand that moved the object.
2. **Mode A (fallback):** write S4's realized object trajectory back as a cleaned bundle and run the team's `run_load_retarget.py` unchanged. This breaks co-generation: the IK hand did not produce the object motion.
3. **Scene files:** URDF and support USD built from S3's mesh, so trajectories come from our reconstructed objects (R3).

**In → out:** S4 output, S3 mesh → `task.parquet` + URDF + support USD; `score_input/` for the local scorer.

**Check:** the runner's checks (fingertip error < 6 cm max, no link > 2 cm inside the object, seating gap < 1 cm); the reconstruction-only score of the reference (public); for mode B a kinematic replay in FlashCHORD ("Show kinematic reference") must play back what S4 produced.

| Risk | Handling |
| --- | --- |
| SPIDER's joint names or order differ from FlashCHORD's Sharpa Wave | Map by name; any unmatched joint stops the write |
| `robot_*_frames` needs FlashCHORD's frame set | Forward kinematics with FlashCHORD's own URDF, checked by the kinematic replay |
| Contact fields are named for MANO contacts | Filled with the SPIDER hand's contacts; the replay compares them with S4's contact sequence |
| Mode B fails on ep41 | Mode A, recorded in the context file |

### S6. Policy in the loop

**How (new):**

1. **Train** one FlashSAC policy per episode with FlashCHORD's `sharpa_flash_sac` recipe and Edmond's 60M-step overrides (`tools/track3/jobs/still_submit.sh`): `task.motion_speed=1.0`, `sim.fps=20.254`, `termination.wrist_position.threshold=0.3`, and the VOC curriculum compressed from 250M to 60,006,400 steps. The curriculum has ten helper-force stages, 1.0 → 0.75 → 0.5 → 0.25 → 0.1 → 0.05 → 0.025 → 0.01 → 0; compressed, the last \~27M steps are unassisted, as the evaluator runs. `sim.njmax` stays at the recipe's 512 (James, 2026-10-20), since Edmond's 1024 peaks at 45.5 GB, above the L40S's 46 GB.
2. **Evaluate** with the kit's `evaluate_policy.py`: deterministic, start frame 0, unassisted, the kit unmodified (R5).
3. **Select** per episode the better of {tracking, still}: by local score on public episodes; by the evaluator's rollout-vs-reference error and termination frame on evaluation episodes, never ground truth or single-episode leaderboard tests.
4. **Rerun** the \~8 episodes where tracking is close at 250M steps with the default curriculum.

**In → out:** S5 parquet → `policy_*.safetensors`, evaluator `episode_XXXXXX.parquet` + metrics JSON.

**Check:** official metrics on the rollout minus the same metrics on the S5 reference (public); today −0.044 on ep41 (0.243 vs 0.287), target ≈ 0. Termination frame at the evaluator's thresholds (object 10 cm / 0.5 rad, wrist 15 cm) and MPPE on both splits.

**Key idea:** the rollout is what Kaggle scores, so it is the quantity the loop optimizes.

| Risk | Handling |
| --- | --- |
| `njmax` 512 overflows in contact-rich scenes, the reason Edmond raised it | First runs: ep41 and one two-object episode (ep6); check for constraint overflow; move to Babel (A6000 or ≥ 80 GB) if either fails |
| A short run without the compressed curriculum never reaches VOC 0 (10M/20M runs knocked objects away; ep33: 63 cm, 75°) | Always the compressed thresholds at 60M |
| 60M steps underfit | Per-episode fallback to the still policy; 250M reruns where tracking is close |
| L40S throughput is unmeasured (21,600 steps/s was the A6000) | Measured on the first run; the budget is redone from it |
| Still policies questioned | Trained only from our reconstruction; disclosed in the technical report; organizers asked on 2026-10-20 |

### S7. Pack and submit

**How (new; Kaggle token ready and repo shared, 2026-10-20):**

1. **Compliance preflight (ours):** all 20 rollouts present; mesh hashes match S3; no ground-truth reads in any evaluation `report.json`; clean, pushed commit; kit folder unmodified (R1–R5). Any failure blocks the pack.
2. **Pack:** the kit's `pack_submission.py --track track_3` with the rollouts and S3 meshes, then `preflight.py`; every `MESH WARNING` is reviewed.
3. **Upload:** `submit/v2d_submit.py --track track_3 --wait` to all six competitions; tag the commit `submit/b-YYYY-MM-DD-N`.
4. **Archive:** checkpoints, task parquets and meshes to HF under the tag, for later validation (R5).

**In → out:** evaluator parquets, meshes → `track3_submission.parquet` (+ `track3_submission.cd-o.parquet`), receipt, leaderboard scores.

**Check:** leaderboard vs local scores on the public split, the calibration of everything above.

| Risk | Handling |
| --- | --- |
| Mesh and pose frames disagree | Hash check plus the packer's validation; the same mesh feeds S4–S6 |
| Submission cap (5 per week) | Edmond's still submission is already point 1; at most 2 more this week and 3 next week |
| Scoring takes up to 24 h | Final upload Oct 30; Oct 31 for verified fixes only; Nov 1 hard stop |
| Local and leaderboard scores drift | Compared on every upload; a gap over \~0.05 AUC stops new uploads until explained |

## Rule compliance

Path B v2 meets every published Track 3 rule as written. Seven rules apply; each is enforced at a named step, and S7's preflight fails the pack if any check fails. Checked on 2026-10-20 against the challenge FAQ (`assets/docs/challenge/assets/v2d_challenge_2026_faqs.pdf`), the kit's `tracks/track_3/README.md` and the [challenge page](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/).

| # | Rule (official wording) | Steps | How the pipeline enforces it |
| --- | --- | --- | --- |
| R1 | "Object trajectories from learned policies … for all 20 egocentric videos" (FAQ) | S6, S7 | Every packed trajectory is an `episode_XXXXXX.parquet` from the kit's `evaluate_policy.py` on a FlashCHORD checkpoint, run "unassisted" (no VOC). S4 demonstrations and bundle tracks are never packed. Preflight fails if any of the 20 is missing. |
| R2 | Participants "should reconstruct the objects … and submit their own reconstructed meshes", even where public data has a mesh (FAQ); every object on the roster (kit README) | S3, S7 | One S3 mesh per `(episode, object)` in `data/track_3_evaluation_objects.json`, two-object episodes included. A public scan is never submitted. Every packer `MESH WARNING` is reviewed before upload. |
| R3 | Trajectories are generated with the reconstructed objects; "meshes in the same body coordinates as their rollout poses" (FAQ, kit README) | S3–S7 | The same mesh file flows S3 → S4 collision model → S5 URDF/USD → S7 packer. Each step records its SHA-256 in `report.json`; preflight fails on any mismatch. |
| R4 | Evaluation ground truth is held out (challenge page) | S1–S6 | GT firewall: with `--split evaluation`, loaders refuse public scans, ground-truth trajectories and scan-oracle paths, and each `report.json` lists the files read. S2's GT scale check, S3's CD-O and size checks and S4's `physics` column run on the public split only. Global settings (densities, thresholds) are fixed on the public split before any evaluation run. |
| R5 | A git commit URL, script hashes and the policy checkpoint "for later validation"; private repos invite `Nvidia-V2D-Challenge` (FAQ, kit README) | S7 | Pack only from a clean, pushed commit; the kit's evaluator, packer and uploader are used unmodified, so the hashes the packer records match the release. Checkpoints, task parquets and meshes are archived to HF under the `submit/b-…` tag. |
| R6 | "Up to five times per week, with unlimited submissions during the final three days"; freeze Nov 4, 2026, 5:00 pm EST (challenge page); the limit is per track (FAQ) | S7 | Each upload is logged with its receipt. The plan uses at most 2 this week and 3 next week. Nov 1 is a hard stop (James, 2026-10-20); the unlimited Nov 2–4 window is not used. |
| R7 | Winners release code: "Apache 2.0" (FAQ); "research license" (challenge page) | repo | Team code is Apache 2.0. `THIRD_PARTY.md` lists every dependency and its license. Non-commercial code ([SPIDER](https://github.com/facebookresearch/spider), CC BY-NC 4.0) is installed at a pinned commit and never copied into the repo. |

## Gray areas and how we handle them

None of these breaks a written rule; each is closed with a confirmation, a disclosure or a code check. James emailed v2d\_challenge@nvidia.com with the first three questions on 2026-10-20; the reply is pending.

| Item | Why it is open | Handling | Owner, by |
| --- | --- | --- | --- |
| SPIDER is CC BY-NC 4.0 | The FAQ asks winners for Apache 2.0; the challenge page says "research license" | Ask whether a non-commercial dependency is acceptable in a winning release. Until then S4 keeps SPIDER as a pinned, uncopied dependency. If the answer is no, re-implement S4's sampler on FlashCHORD's Newton/Warp stack before the code release (after Nov 1). | James, emailed 2026-10-20 |
| Stereo stream (`ego_cam_b/c`) | NVIDIA's approval (James, 2026-10-18) is not on file; S2's scale depends on it | Save the reply in `docs/compliance/`; if it cannot be found, ask again in the same email | James, Oct 21 |
| Still policy | It is a learned FlashCHORD checkpoint, so it meets R1's wording, but it does not follow the video's motion | Train it only from our reconstruction (reconstructed initial pose, S3 mesh). Ask in the same email. Report the per-episode choice (tracking or still) in the technical report. | James, emailed 2026-10-20; Claude (report) |
| Per-episode selection on the evaluation split | No ground truth exists there, and per-episode leaderboard probing would tune to held-out data | Select only on the evaluator's rollout-vs-reference error and termination frame. The leaderboard checks whole submissions, never single episodes. | Claude, S6 |
| Manual inputs on evaluation videos | The rules say nothing; hand labels would be hard to defend | Object names come from the kit roster. No hand-drawn masks, boxes or hand-picked frames on evaluation videos. Any manual parameter goes into `manual_inputs.json` and the technical report. | Claude, S1–S3 |
| Kaggle rules for the six Track 3 competitions | Not readable from here; each must be accepted before upload | Read and accept all six; flag anything that contradicts the table above | James, Oct 21 |

Code to add before S3 runs on the evaluation split (Claude, Oct 22): the GT firewall (R4), mesh-hash tracking (R3) and a compliance preflight in S7 that checks R1–R5 and blocks the pack on any failure.

## Plan rules and GPU budget

Nov 1 is the hard deadline for the submission, so the plan keeps a valid submission in hand at every point and puts nothing risky on the critical path.

1. **A valid submission is on the leaderboard by Oct 23**: Edmond's still submission is already there (James, 2026-10-20); the Oct 23 upload adds Path B's stereo scale to the still and tracking policies. Everything after that can only replace it if it scores better locally.
2. **Hard gates with a fallback at each.** Gate 0: one S4 output replays in FlashCHORD's physical replay with object error under the kit's termination thresholds. Gate 1 (Oct 25): physics (S4) must beat the stereo column on the public split, with the Allegro/XHand control run alongside, or it leaves the submission path. References freeze on Oct 27, because policies take hours.
3. **Final submission Oct 30; Oct 31 is a resubmission day; nothing new starts on Nov 1.**

One L40S, about 288 GPU-hours over 12 days; the jobs below use about 90, with the policies running at night.

| Job | Cost | Total |
| --- | --- | --- |
| Path B S1–S2 on the 20 evaluation episodes (never run yet) | \~6 min each | 2 h |
| Still policies, 20 evaluation episodes, 60M steps | \~0.8 h each | 16 h |
| Tracking policies, 20 evaluation episodes, 60M steps | \~0.8 h each | 16 h |
| 250M-step reruns for the \~8 episodes where tracking is close | \~3.2 h each | 26 h |
| Physics retargeting (S4), all 40 episodes, with contact guidance | \~10 min each, retries extra | 8–12 h (seconds each without guidance) |
| SAM3D meshes, \~10 objects | \~20 min each | 4 h |
| Validation runs on public episodes |  | \~20 h |

Step costs come from the team's measurements and SPIDER's paper: FlashCHORD's Sharpa recipe is 250,003,456 steps at \~21,600 environment steps/s on an A6000 (\~3.2 h), a rate not yet measured on the L40S, so the first job on the instance is to measure it; SPIDER+C runs at 0.4–0.5 FPS on one L40S (Table 1); Path B's P2 took \~5 min per episode on the L40S. The total is about 100 of 288 GPU-hours.

## Risks, cuts, and what depends on James

| Risk | Answer |
| --- | --- |
| SPIDER's Sharpa scores 40.7 vs Allegro 71.2 on HOT3D (Table 3) | The Allegro/XHand control at gate 1 separates the hand model and gains from physics in the loop; if the hand is the blocker, tune the Sharpa gains or ask whether another embodiment is worth the switch (FAQ welcomes them) |
| Feasibility in SPIDER's MuJoCo Warp does not transfer to FlashCHORD's Newton settings | Gate 0: physical replay of one S4 output in FlashCHORD before S4 runs at scale; match sim and control timesteps where possible |
| A noisy reference leads S4 to the wrong contacts despite physical consistency | Stage 2 cleaning stays as pre-processing; contact guidance off on noisy episodes, as SPIDER does at scale |
| S4 with contact guidance is slow (\~10 min per episode plus retries) | Budget 8–12 h; resample references to 12.5 Hz; run without guidance where the reference is noisy |
| SPIDER's Sharpa is not FlashCHORD's Sharpa Wave model | Same 22 finger DoF; compare the URDF and the gains on day 1, adapt the embodiment XML if they differ |
| No SAM 3D Objects access by Oct 24 | Resolved: access granted (James, 2026-10-20); bundle meshes stay the fallback (CD-O \~0.74 cm) |
| The L40S dies | `setup.sh` + HF restore, \~2 h; outputs are uploaded with `hf_outputs.py` after each milestone |
| Edmond's packed still submission is unreachable | Resolved: it is on the leaderboard (James, 2026-10-20) |
| SPIDER's code is CC BY-NC 4.0 | Used as a dependency, never vendored; a winning Apache-2.0 release can depend on it but not include it. James asks the organizers by Oct 21 (see Gray areas); re-implementation on FlashCHORD's Newton/Warp stack only if they require it |
| FlashCHORD at 60M steps tracks worse than at 250M | The selection rule falls back to the still policy per episode; 250M reruns go to the episodes where tracking is close |
| `wooden_spoon` has no public scan | CD-O for it rests on S3's health checks alone |

**Cut from the Nov 1 plan:** the retraining flywheel (one pass only: train, evaluate, select), the SPIDER re-implementation, and the learned egocentric 4D model. All three are the first items after Nov 1. S5 mode A (the IK re-solve) stays only as the fallback if mode B is not ready on ep41.

**Depends on James:** done as of 2026-10-20: the Kaggle token, the repo shared with the `Nvidia-V2D-Challenge` org, SAM 3D Objects access, Edmond's still submission on the leaderboard, and the email to the organizers. Still open: their reply, and the go/no-go at gate 1 on Oct 25.

## Decisions that scope this plan (James, 2026-10-19)

- **Policy rule (team decision):** the FAQ requires learned policies and welcomes other embodiments; the team chose the kit's evaluator, which accepts only FlashCHORD (`flash_sac` / `ppo`) checkpoints. The framework may change everything that feeds it.
- **External data:** anything the FAQ allows (pretrained models, external datasets); licenses still checked per item.
- **Objective:** the leaderboard first (six metrics, sum of ranks); physical validity where it raises them.
- **Compute and people:** one Verda L40S; James and Claude.
- **Perception for Nov 1:** Path B P0–P4 plus the team's bundles; the learned 4D ego model and SAM 3 masks come after.
- **Block 3:** SPIDER's code as a dependency; re-implement only if it works and the release needs it.
- **Evidence:** literature plus our measurements; no new GPU runs for the proposal itself.
- **Inputs:** test time `ego_cam_a` + the stereo pair (allowed per NVIDIA's reply to James, 2026-10-18); training and fitting may use everything in the public split and external data.
- **Submissions:** two leaderboard checkpoints at least (an early fallback and the final).
- **Review (2026-10-19):** an adversarial review of this doc was checked item by item; its corrections are folded in above (kit termination thresholds, 19-of-20 episodes, the policy rule as a team decision, physics constrains rather than replaces, SPIDER's Sharpa numbers and the Allegro/XHand control, Mode B before Mode A, the gate-0 replay, L40S speed to be measured, S4 runtime).

**Decisions (James, 2026-10-20):**

- **Training GPU:** policies train on the L40S at FlashCHORD's `njmax` 512; one ep41 run first, and a move to Babel only if it runs out of memory or tracks badly.
- **Fallback:** the Oct 23 still-policy submission goes ahead before the organizers answer the still-policy question; it is disclosed and replaced if they object.
- **Embodiment:** stay on Sharpa Wave; if gate 1 shows a Sharpa-specific failure, tune its gains. Allegro/XHand stays a diagnostic.
- **Deadline:** Nov 1 is a hard stop; nothing is submitted Nov 2–4.

**Re-check against the code (2026-10-20):** S1–S7 now list how, inputs and outputs, and risks per step. Corrections folded in: gate 0 replays with `voc_scale=0` (the physical-replay config defaults to 1.0); 60M-step policies use Edmond's compressed ten-stage VOC curriculum; SPIDER's scene builder has no table, so S4 puts the table top at z = 0 (its two-object bimanual path already exists and is reused); SPIDER ships no V2D processor; SPIDER needs its own venv; S3 tries the ego pipeline's SAM3D before the scan-oriented HOI module; the drop test also runs in MuJoCo Warp; S5 mode B writes FlashCHORD's ManoSharpa columns, including `robot_*_frames`.

## Sources

Pages opened 2026-10-19.

- [CHORD: Learning Dexterous Manipulation Using Contact Wrench Guidance From Human Demonstration (arXiv 2607.00033)](https://arxiv.org/abs/2607.00033) · [full text](https://arxiv.org/html/2607.00033v2) (§4.1, §6, Appendix A.1 read)
- [ManipTrans (arXiv 2503.21860)](https://arxiv.org/abs/2503.21860) · [code](https://github.com/ManipTrans/ManipTrans)
- [SPIDER: Scalable Physics-Informed Dexterous Retargeting (arXiv 2511.09484v3)](https://arxiv.org/html/2511.09484v3) (§4.4, §4.5, §5, Tables 1, 3, 5, 6, Appendix A.3–A.4 read) · [code](https://github.com/facebookresearch/spider)
- [DexTrack (arXiv 2502.09614, ICLR 2025)](https://arxiv.org/abs/2502.09614) · [code](https://github.com/Meowuu7/DexTrack)
- [DexMachina (arXiv 2505.24853)](https://arxiv.org/abs/2505.24853)
- [VideoMimic: Visual Imitation Enables Contextual Humanoid Control (arXiv 2505.03729)](https://arxiv.org/html/2505.03729) (§3.2, §5.2 read)
- [HOT3D (arXiv 2411.19167, CVPR 2025)](https://arxiv.org/abs/2411.19167)
- [SimPoE (arXiv 2104.00683, CVPR 2021)](https://arxiv.org/abs/2104.00683) · [HOIC (SIGGRAPH 2024)](https://dl.acm.org/doi/10.1145/3641519.3657505) · [ViTaM-D (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Yu_Dynamic_Reconstruction_of_Hand-Object_Interaction_with_Distributed_Force-aware_Contact_Representation_ICCV_2025_paper.pdf): abstracts only
- [DeepSimHO (arXiv 2310.07206)](https://arxiv.org/pdf/2310.07206) · [AGILE (arXiv 2602.04672)](https://arxiv.org/pdf/2602.04672) · [DexMan (arXiv 2510.08475)](https://arxiv.org/pdf/2510.08475)
- [Newton: NVIDIA's announcement](https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/) (MuJoCo Warp as its solver)
- V2D Challenge FAQ (`assets/docs/challenge/assets/v2d_challenge_2026_faqs.pdf`); submission kit `tracks/track_3/README.md`, `flash_chord/scripts/evaluate_policy.py`, `flash_chord/evaluation/harness.py` (lines 293–303: supported checkpoints)
- Team repo `baseline/robotic_grounding/flash_chord/`: `README.md` (§3 replay, §4 train), `configs/termination/tracking.yaml`, `configs/sim/rl.yaml`, `configs/curriculum/`, `data/loader.py`, `assets/urdfs/sharpawave/`; `baseline/reconstruction/modules/v2d_hoi_object_reconstruction/`
- `V2D_PROJECT_CONTEXT.md` §4 (Path B results, ep41, public dry run); Path B waterfalls `0fa525bc`, `4c2eac31`
