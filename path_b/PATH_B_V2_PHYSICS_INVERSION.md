# Path B v2: physics-grounded video inversion

Created Oct 19, 2026 · last updated Oct 19, 2026 · exported from the [Claude doc](https://claude.ai/code/artifact/afa70a10-81ae-43e2-a850-0549839a491a), which is the canonical copy

Track 3 scores a simulated rollout, so Path B v2 treats the whole task as one inference problem: find the physically executable demonstration that best explains the ego video, enforced by the same physics the kit's evaluator runs, with the FlashCHORD policy trained inside the loop. Approved by James on 2026-10-19 as the plan to the Nov 1 deadline.

## TL;DR

Today's chain is open-loop: perception → repairs → retarget → RL → rollout, and each stage hands the next a reference it was never asked to make feasible. Path B v2 inverts the order of authority: the simulator is the model of the world, the demonstration (hands, object motion, mesh, support, contacts) is the hidden state, and the video is evidence. Perception becomes an observation model; it no longer decides anything.

- **What is scored** is a FlashCHORD rollout in the kit's evaluator (FAQ: "object trajectories from learned policies"; the evaluator accepts only `flash_sac` or `ppo` checkpoints). So the rollout is the quantity the loop optimizes.
- **Physics replaces repairs.** Rest hold, object cleaning, levelling, attach-hands, seat and merge are hand-written physics; a demonstration realized in MuJoCo Warp has them by construction: an object moves only through contact and rests on its support otherwise.
- **Same engine, both ends.** The kit's evaluator runs on Newton / MuJoCo Warp; SPIDER's physics retargeting runs on MuJoCo Warp with the Sharpa hand. What is feasible in one is feasible in the other.
- **Change from Path B v1:** v1 fixed scale (stereo) and planned VLM boxes and SAM 3 masks for Stage 2. v2 keeps v1's P0–P4 as the evidence layer and puts physics-consistent demonstration inference plus policy-in-the-loop in place of Stages 2–3. The learned egocentric 4D hand-object model (block 1 in full) is after Nov 1.

## Why not better stages

Our own numbers put the loss between stages, not inside one: a policy tracking a jittery, contact-inconsistent reference scored below doing nothing.

| Measurement (20 public episodes unless noted) | AUC | Source |
| --- | --- | --- |
| Team bundles played back | 0.270 | Path B waterfall, run `0fa525bc` |
| + stage 2 repairs (rest hold, cleaning, levelling) | 0.314 | same |
| + Path B stereo scale | 0.420 | same |
| Residual after the scan-oracle scale (tracking, mesh, placement) | ≈0.6 of the scale | same |
| A still object: our two-object placement vs the true placement | 0.309 vs 0.447 | public dry run `ef058c7e` |
| ep41: trained tracking policy / playback / **still policy** | 0.243 / 0.287 / **0.356** | V2D\_PROJECT\_CONTEXT.md §4 |

CHORD's own paper says why: "effective contact guidance requires relatively clean human demonstrations", contact noise "degrades performance more noticeably" than pose noise, and a rollout is terminated at 15 cm / 40° object error ([CHORD §3–6](https://arxiv.org/html/2607.00033v2)). The repairs the team bolts on (rest hold, levelling, attach-hands, seat, merge) are hand-written physics. v2 replaces them with physics.

## Architecture

*Architecture diagram (ego video → 1 evidence → 2 object model → 3 physics-consistent demonstration → 4 policy in the loop → kit evaluator, with the mesh route to CD-O and the post-Nov 1 feedback loop) is in the [live doc](https://claude.ai/code/artifact/afa70a10-81ae-43e2-a850-0549839a491a).*

Block 3 (accented) is where the chain stops being open-loop: the demonstration handed to the policy is realized in the same physics the evaluator runs. The dashed loop, rollout failures fed back into block 3, is the part deferred past Nov 1.

## Evidence

The kit's two success metrics are named after ManipTrans and SPIDER, so the organizers' own lineage is physics retargeting plus residual RL.

| Claim | Evidence |
| --- | --- |
| The organizers' lineage is physics retargeting + residual RL | FAQ names "Maniptrans Success Rate (MP-SR)" and "Spider Success Rate (SP-SR)". [ManipTrans](https://arxiv.org/abs/2503.21860) (CVPR 2025) "pre-trains a generalist trajectory imitator… then fine-tunes a specific residual module under interaction constraints"; [SPIDER](https://arxiv.org/abs/2511.09484) does "physics-based sampling with curriculum-style virtual contact guidance", "10x faster than RL baselines"; its [code](https://github.com/facebookresearch/spider) runs on MuJoCo Warp with the Sharpa hand and has processors for HOT3D v2, DexYCB and OakInk |
| Blocks 3 and 4 can share one physics engine with the evaluator | FlashCHORD README: "GPU-parallel robotic reference tracking with Newton, Warp, JAX, PPO, and FlashSAC"; `objectives/` holds `contact.py`, `wrench.py`, `tracking.py` |
| Physics in the loop removes jitter, flips and penetration at the source | [SimPoE](https://arxiv.org/abs/2104.00683) (CVPR 2021): a simulated character driven from video, state-of-the-art accuracy with physically plausible motion; [HOIC](https://dl.acm.org/doi/10.1145/3641519.3657505) (SIGGRAPH 2024): RL plus a surface contact model to reconstruct hand-object interaction; [ViTaM-D](https://openaccess.thecvf.com/content/ICCV2025/papers/Yu_Dynamic_Reconstruction_of_Hand-Object_Interaction_with_Distributed_Force-aware_Contact_Representation_ICCV_2025_paper.pdf) (ICCV 2025): force-aware contact optimization that removes penetrations |
| The rollout should be the optimized object | [DexTrack](https://arxiv.org/abs/2502.09614) (ICLR 2025): per-trajectory homotopy optimization with the learned controller, "over a 10% improvement in success rates"; [DexMachina](https://arxiv.org/abs/2505.24853): "virtual object controllers with decaying strength"; FlashCHORD ships the same idea as a `voc_scale` curriculum (1.0 → 0.25 → 0) |
| A learned ego 4D model can be trained on data that matches our rig (after Nov 1) | [HOT3D](https://arxiv.org/abs/2411.19167) (CVPR 2025): 833 minutes, 19 subjects, 33 objects, marker-based ground truth, recorded on Aria glasses (RGB plus mono stereo cameras, like our OAK rig); multi-view beats single-view on its benchmarks. FAQ: pretrained models and external data are allowed in all tracks |
| Metric scale comes from stereo, not from priors | Path B P3: within 3% of the scan oracle on ep41, +0.106 AUC over 19 public episodes; [VideoMimic](https://arxiv.org/abs/2505.03729) (CoRL 2025) had only monocular video and resolved scale from a human-height prior |
| Video → sim → policy is a proven pattern | VideoMimic: 4D human and scene from phone video, RL tracking in simulation, a distilled policy deployed on a Unitree G1 |

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

- **Does:** P0–P2 as built: rectify the stereo pair with the per-episode row correction, undistort `ego_cam_a`, run FoundationStereo, fit planes, find the stereo-to-colour frame offset, reproject depth into `ego_cam_a`. Plus the team's bundle for the episode (MANO hands, object pose track, mesh) as the kinematic prior; in the full framework a learned 4D model replaces it.
- **Inputs → outputs:** `ego_cam_a/b/c`, factory calibration, HF bundle → metric depth per frame in `ego_cam_a`, planes, frame offset, hands / object / mesh as noisy observations.
- **Loss:** the existing P1–P2 metrics (row offset after correction < 0.5 px, features within 5 px ≥ 0.5, plane RMS). Nothing new.

### S2. Metric scale and scene frame

- **Does:** P3–P4 as built: one scale factor per episode from the camera's height above the support plane (stereo vs bundle), cross-checked along the object ray; the support plane becomes the table; the scorer's anchor object defines frame 0.
- **Inputs → outputs:** S1 → rescaled bundle, support-plane height, gravity, anchor object.
- **Loss:** scale factor vs the ground-truth trajectory scale on public episodes (a fit of the bundle's object motion to GT; replaces the misleading scan-oracle bar), spread over rest frames (< 5%), support plane vs gravity (< 8°).
- **Key idea:** scale is the one thing physics cannot recover; everything downstream assumes metres.

### S3. Object model

- **Does:** builds the mesh that both the scorer and the simulator see. First pass: the bundle mesh at S2's scale. Second pass: the HOI module's SAM3D mode on the clip's best unoccluded views (`--sam3d_use_depth` with our stereo depth, several seeds, view bins), scale fitted on stereo, best candidate kept. Then a collision model: CoACD decomposition (SPIDER's `decompose.py`), mass from volume × a per-class density, a fixed table friction.
- **Inputs → outputs:** frames, masks from the bundle run, stereo depth, S2 scale → `mesh.glb` (visual), convex pieces (collision), inertial proxy.
- **Loss:** CD-O vs the public scan and size ratio 0.8–1.25× (public); watertightness, number of convex pieces, and a drop test in the simulator (the object must settle within 1 cm of its frame-0 pose; `mesh_to_usd` has one). `wooden_spoon` has no scan, so only the health checks apply to it.
- **Key idea:** a mesh that is too small or not watertight makes S4 fail physically before it fails CD-O.

### S4. Physics-consistent demonstration (the heart)

- **Does:** SPIDER physics retargeting in MuJoCo Warp. The kinematic reference (hand keypoints from the bundle's MANO, object pose track, contacts, support plane) is the target; sampling-based optimization finds Sharpa-hand joint trajectories whose simulated interaction moves the object as close as physics allows to the reference. The object only moves through contact; before the first touch it rests. Two-object episodes become one scene with both objects and the support graph.
- **Inputs → outputs:** S2 bundle (keypoints, object poses), S3 collision model, table → robot hand trajectories, the physically realized object trajectory, contact sequence, a success flag per episode.
- **Loss, three layers:** (1) tracking: physics object trajectory vs the kinematic reference, cm and degrees per frame; (2) validity: penetration depth, object displacement before first contact (must be 0), maximum object speed; (3) ground truth: a new `physics` column in the waterfall on public episodes, AUC / MPPE of the physics trajectory vs GT next to raw / stage 2 / stereo. If `physics` beats `stereo`, the idea holds before any policy is trained.
- **Key idea:** one mechanism replaces rest hold, object cleaning, levelling, attach-hands and seat. Flips and drift while held cannot happen: a 90° turn in 1/20 s needs forces no hand applies. The same engine family runs the kit's evaluator, so what is feasible here is feasible there.

### S5. Demonstration → FlashCHORD task

- **Does:** writes the task parquet FlashCHORD trains on. Mode A (first): write S4's realized object trajectory and hands back as a cleaned bundle and run the team's `run_load_retarget.py` unchanged. Mode B (cut from the Nov 1 plan): write the parquet directly from SPIDER's robot trajectories (`robot_*_wrist_*`, `robot_*_finger_joints`, `object_body_*`, MANO contact points), skipping the IK re-solve.
- **Inputs → outputs:** S4 output → `task.parquet` + URDF + support USD; `score_input/` for the local scorer.
- **Loss:** the runner's checks (fingertip error < 6 cm max, no link > 2 cm inside the object, seating gap < 1 cm) and the reconstruction-only score of the parquet reference (public).

### S6. Policy in the loop

- **Does:** FlashCHORD Sharpa FlashSAC per episode with its VOC curriculum, then the kit evaluator. One pass for Nov 1: train, evaluate, select. A selection rule picks, per episode, the better of {tracking policy, still policy} by local score (public) or by the evaluator's rollout-vs-reference error and termination frame (evaluation split). The retraining flywheel (termination frames fed back into S4) is after Nov 1.
- **Inputs → outputs:** S5 parquet → `policy_*.safetensors`, evaluator `episode_XXXXXX.parquet` + metrics JSON.
- **Loss:** the policy loss = official metrics on the rollout minus the same metrics on the S5 reference (public). Today that gap is negative on ep41 (0.243 vs 0.287); the target is a gap near zero, which means the reference was executable. Also termination frame and MPPE from the evaluator JSON on both splits.
- **Key idea:** the rollout is what Kaggle scores, so it is the quantity the loop optimizes.

### S7. Pack and submit

- **Does:** kit packer and preflight on the 20 evaluation rollouts and the S3 meshes; tag `submit/b-YYYY-MM-DD-N`.
- **Inputs → outputs:** evaluator parquets, meshes → `track3_submission.parquet`, leaderboard scores.
- **Loss:** leaderboard numbers vs our local scores on the public split: the calibration of everything above.

## Plan rules and GPU budget

Nov 1 is the hard deadline for the submission, so the plan keeps a valid submission in hand at every point and puts nothing risky on the critical path.

1. **A valid submission is on the leaderboard by Oct 23**, made with the existing chain (stereo-rescaled bundles, stages 2–3, still and tracking policies). Everything after that can only replace it if it scores better locally.
2. **Hard gates with a fallback at each.** Physics (S4) must beat the stereo column on the public split by Oct 25 or it leaves the submission path. References freeze on Oct 27, because policies take hours.
3. **Final submission Oct 30; Oct 31 is a resubmission day; nothing new starts on Nov 1.**

One L40S, about 288 GPU-hours over 12 days; the jobs below use about 90, with the policies running at night.

| Job | Cost | Total |
| --- | --- | --- |
| Path B S1–S2 on the 20 evaluation episodes (never run yet) | \~6 min each | 2 h |
| Still policies, 20 evaluation episodes, 60M steps | \~0.8 h each | 16 h |
| Tracking policies, 20 evaluation episodes, 60M steps | \~0.8 h each | 16 h |
| 250M-step reruns for the \~8 episodes where tracking is close | \~3.2 h each | 26 h |
| Physics retargeting (S4), all 40 episodes | minutes each | < 4 h |
| SAM3D meshes, \~10 objects | \~20 min each | 4 h |
| Validation runs on public episodes |  | \~20 h |

Step costs come from the team's measurements: FlashCHORD's Sharpa recipe is 250M steps at \~21,600 environment steps/s on an A6000 (\~3.2 h); Path B's P2 took \~5 min per episode on the L40S.

## Day by day, Oct 20 – Nov 1

One L40S, James and Claude. GPU jobs run overnight in tmux. "James" marks what only he can do.

| Day | Work | Gate / fallback |
| --- | --- | --- |
| Oct 20 | New L40S via `setup.sh` + restore. Path B S1–S2 on the 20 evaluation episodes. Two-object path for Path B bundles into stages 2–3 (a hub-style folder the runner's `--merge` reads). James: Kaggle token, share the repo with `Nvidia-V2D-Challenge`, request `facebook/sam-3d-objects` access, ask Edmond for the packed still submission | evaluation bundles rescaled |
| Oct 21 | Stages 2–3 on all 20 evaluation episodes (still scenes). Night: 20 still policies at 60M. Install SPIDER, run its example, check its Sharpa model. James: upload Edmond's still submission if retrievable (leaderboard point 1) | — |
| Oct 22 | Evaluate the 20 still policies, pack, preflight. Write the V2D → SPIDER dataset processor; ep41 through decompose, XML, IK | — |
| Oct 23 | **Submit the fallback** (stereo-rescaled still). S4 on the 7 single-object public episodes; `physics` column in the waterfall. Night: tracking policies at 60M for the 20 evaluation episodes from the stage 2 chain | fallback is on the board |
| Oct 24 | S4 on the 12 two-object public episodes (support graph, anchor). S2's GT scale check. Evaluate the tracking policies; per-episode selection tracking vs still | — |
| Oct 25 | **Gate 1:** physics vs stereo on public AUC / MPPE / RPE. If physics wins: S4 on the 20 evaluation episodes and S5 mode A. If not: physics leaves the path and the chain stays stage 2. Pack and submit the selection (point 2) | decision recorded in the context file |
| Oct 26 | S3: SAM3D multi-view meshes for the objects with bad bundle meshes, drop test, CD-O on public, only if access arrived by Oct 24; otherwise the bundle meshes stay. Night: tracking policies at 60M on the physics references (if gate 1 passed) | mesh swap is independent of the policies |
| Oct 27 | **Reference freeze.** Evaluate, select per episode. Pick the \~8 episodes where tracking is within reach of still for 250M reruns. Night: first 250M batch | no reference changes after today |
| Oct 28 | 250M batch continues, day and night. Local public-split dry run of the exact final pipeline, meshes included | — |
| Oct 29 | 250M batch ends. Final selection; pack; preflight; checksum the submission file. Submit if ready (point 3) | everything after this is replacement only |
| Oct 30 | **Final submission** (or confirm the Oct 29 one). Record scores; local vs leaderboard calibration | leaderboard numbers |
| Oct 31 | Resubmission day: only a fix already verified locally (e.g. a mesh swap). Write results into the context file and this doc | — |
| Nov 1 | Nothing new. Confirm the leaderboard shows the intended submission; archive the instance's outputs to HF | done |

Submissions stay under the 5-per-week limit: at most 2 in the week of Oct 20 and 3 in the week of Oct 27.

## Risks, cuts, and what depends on James

| Risk | Answer |
| --- | --- |
| SPIDER's Sharpa is not Sharpa Wave | One day to adapt the embodiment XML; gate 1 moves to Oct 26 at the latest |
| No SAM 3D Objects access by Oct 24 | Bundle meshes stay, as today; CD-O unchanged at \~0.74 cm |
| The L40S dies | `setup.sh` + HF restore, \~2 h; outputs are uploaded with `hf_outputs.py` after each milestone |
| Edmond's packed still submission is unreachable | The Oct 23 fallback becomes leaderboard point 1 |
| SPIDER's code is CC BY-NC 4.0 | Used as a dependency, never vendored; a winning Apache-2.0 release can depend on it but not include it. Re-implementation on FlashCHORD's Newton/Warp stack only if the organizers require it |
| FlashCHORD at 60M steps tracks worse than at 250M | The selection rule falls back to the still policy per episode; 250M reruns go to the episodes where tracking is close |
| `wooden_spoon` has no public scan | CD-O for it rests on S3's health checks alone |

**Cut from the Nov 1 plan:** the retraining flywheel (one pass only: train, evaluate, select), S5 mode B (the IK re-solve stays), the SPIDER re-implementation, and the learned egocentric 4D model. All four are the first items after Nov 1.

**Depends on James:** the Kaggle token and the repo shared with the `Nvidia-V2D-Challenge` org (Oct 20, for the Oct 21 and Oct 23 submissions); SAM 3D Objects access (by Oct 24, for S3); Edmond's packed still submission from Babel; the go/no-go at gate 1 on Oct 25.

## Decisions that scope this plan (James, 2026-10-19)

- **Policy rule:** the scored rollout comes from a FlashCHORD (`flash_sac` / `ppo`) policy run by the kit's evaluator. The framework may change everything that feeds it.
- **External data:** anything the FAQ allows (pretrained models, external datasets); licenses still checked per item.
- **Objective:** the leaderboard first (six metrics, sum of ranks); physical validity where it raises them.
- **Compute and people:** one Verda L40S; James and Claude.
- **Perception for Nov 1:** Path B P0–P4 plus the team's bundles; the learned 4D ego model and SAM 3 masks come after.
- **Block 3:** SPIDER's code as a dependency; re-implement only if it works and the release needs it.
- **Evidence:** literature plus our measurements; no new GPU runs for the proposal itself.
- **Inputs:** test time `ego_cam_a` + the stereo pair; training and fitting may use everything in the public split and external data.
- **Submissions:** two leaderboard checkpoints at least (an early fallback and the final).

## Sources

Pages opened 2026-10-19.

- [CHORD: Learning Dexterous Manipulation Using Contact Wrench Guidance From Human Demonstration (arXiv 2607.00033)](https://arxiv.org/abs/2607.00033) · [full text](https://arxiv.org/html/2607.00033v2)
- [ManipTrans (arXiv 2503.21860)](https://arxiv.org/abs/2503.21860) · [code](https://github.com/ManipTrans/ManipTrans)
- [SPIDER: Scalable Physics-Informed Dexterous Retargeting (arXiv 2511.09484)](https://arxiv.org/abs/2511.09484) · [code](https://github.com/facebookresearch/spider)
- [DexTrack (arXiv 2502.09614, ICLR 2025)](https://arxiv.org/abs/2502.09614) · [code](https://github.com/Meowuu7/DexTrack)
- [DexMachina (arXiv 2505.24853)](https://arxiv.org/abs/2505.24853)
- [VideoMimic: Visual Imitation Enables Contextual Humanoid Control (arXiv 2505.03729)](https://arxiv.org/abs/2505.03729) · [site](https://www.videomimic.net/)
- [HOT3D (arXiv 2411.19167, CVPR 2025)](https://arxiv.org/abs/2411.19167)
- [SimPoE (arXiv 2104.00683, CVPR 2021)](https://arxiv.org/abs/2104.00683)
- [HOIC (SIGGRAPH 2024)](https://dl.acm.org/doi/10.1145/3641519.3657505)
- [ViTaM-D (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Yu_Dynamic_Reconstruction_of_Hand-Object_Interaction_with_Distributed_Force-aware_Contact_Representation_ICCV_2025_paper.pdf)
- [DeepSimHO (arXiv 2310.07206)](https://arxiv.org/pdf/2310.07206) · [AGILE (arXiv 2602.04672)](https://arxiv.org/pdf/2602.04672) · [DexMan (arXiv 2510.08475)](https://arxiv.org/pdf/2510.08475)
- V2D Challenge FAQ (`assets/docs/challenge/assets/v2d_challenge_2026_faqs.pdf`); submission kit `tracks/track_3/README.md`, `flash_chord/scripts/evaluate_policy.py`, `evaluation/harness.py`
- Team repo `baseline/robotic_grounding/flash_chord/README.md`, `configs/curriculum/`, `data/loader.py`; `baseline/reconstruction/modules/v2d_hoi_object_reconstruction/`
- `V2D_PROJECT_CONTEXT.md` §4 (Path B results, ep41, public dry run); Path B waterfalls `0fa525bc`, `4c2eac31`
