# Path B: End-to-End VLM Framework for Track 3

Created Oct 11, 2026 · last updated Oct 19, 2026 · exported from the [Claude doc](https://claude.ai/code/artifact/0873d10c-acc3-455c-936a-12baf370fabf), which is the canonical copy

## TL;DR

Recommend a **VLM-orchestrated, stereo-metric object reconstructor** (hybrid of candidates A and B): a small VLM makes only the discrete, semantic decisions; specialist geometry models do all metric work; geometric checks, not the VLM, accept or reject each result.

- **Why it should beat v0.4.0:** it attacks the baseline's six documented error sources directly — single-frame MoGe scale, one tracked object, single-view mesh, memoryless re-registration (flips), no metric world frame, and wrong intrinsics.
- **Biggest lever:** the eval split ships the 75 mm stereo pair (`ego_cam_b/c`), which the v0.4.0 ego pipeline never uses (only its multi-view path does). Stereo gives per-frame metric depth instead of a monocular guess.
- **Why not a pure VLA (candidate C):** Track 3 scores a reconstructed object mesh plus object trajectories, a VLA produces neither, and 20 public episodes on 1–2 GPUs is far too little to fine-tune one.
- **"End-to-end" here means** one entrypoint from video to trajectory + mesh, not one network. No single published model outputs metric multi-object 6-DoF poses and meshes from ego video.

## Verified on our public bundles (2026-10-18)

The approach stands: every targeted problem shows up in the team's 33 public bundles (HF `rose-tea-cafe/v2d-t3-recon` @ `e2492c6`, built from NVIDIA `8c778f9`, whose reconstruction code is identical to v0.4.0). Measured from the bundles' poses, meshes, intrinsics and run commands; mesh size is compared with the public scans; score impacts are the team's existing results. No scorer was run and no evaluation episode was used.

| Problem | Verdict | Evidence |
| --- | --- | --- |
| Wrong object / blob mesh | **Confirmed, bigger than expected** | 5 of 33 meshes are 4–12× the scan's size: all four wooden pieces (ep23, ep24) and the ep31 sweeper |
| Object scale off | Confirmed | Other 28 meshes: 0.68–1.45× the scan; 17 off by more than 20%. Hand scale off by more than 10% in 31 of 58 hand tracks; left and right disagree within one clip (ep31: 0.62 vs 1.26). `--rescale-by-scan` raised AUC 0.240 → 0.291 (`1f3daeb2`) |
| Orientation flips | Confirmed | In the scored world frame, 11 of 33 bundles jump more than 30° between consecutive frames (over 600°/s at 20 fps); 5 jump more than 90° (camera frame: 16 and 7). ep26 pot: 83° at frame 82, then about 160° off for the rest of the clip. ep2 pot: 90–117° camera-frame jumps across frames 267–274. ep41 breakdown: 74 cm, 57° |
| One object per run | Confirmed; cause narrower | All 13 two-object episodes are two separate runs. Their camera paths agree (median 0.3 cm; spikes 2–34 cm), so the cost (AUC 0.309 with our placement vs 0.447 true, `ef058c7e`) comes mostly from per-object tracking and placement |
| Wrong intrinsics | Confirmed; impact unmeasured | Optical centre fixed at the image centre (1014, 760) in all 33; focal length 950–1061 px (11% spread) for one physical camera |
| Stereo unused | Confirmed | All 33 run commands read only `ego_cam_a` |
| Unseeded SAM 3D | Code-confirmed; effect unmeasurable here | No `--seed` in any of the 33 commands; every object was reconstructed only once |
| No metric world frame | Smaller than stated | Gravity alignment applied in all 33; the scorer re-aligns the scene at frame 0. Matters for simulation, not score |

**New priority order for Path B:**

1. Right object: VLM names each object, SAM 3 masks each one. Targets the wrong-object meshes, which tuning cannot fix.
2. Scale (stereo depth) and flips (symmetry-aware, pose-gated tracking). Both confirmed and both cost points.
3. Later: world frame; seeding, until a repeat run shows variance.

**Limits:** flips are judged by physical plausibility, not ground truth (a 90° turn in 1/20 s is impossible; 30–90° is very unlikely). ep41 has no `run.json`. Most bundles are "files present, not revalidated" in `inventory.csv`. Script: `check_bundles.py` (Claude session scratchpad; not in the repo).

## v0.4.0 re-check

The approach still stands. Every error Path B targets is still in [v0.4.0](https://github.com/nvidia-isaac/video_to_data/tree/v0.4.0) (commit 33129dd, the latest tag), and v0.4.0 ships code that makes Path B cheaper to build. `main` has no changes to the ego, FoundationPose or stereo code since that tag.

| Check | v0.4.0 finding (verified in code) | Effect on Path B |
| --- | --- | --- |
| Scale from MoGe on one frame | Unchanged (steps 2, 9) | Target stands |
| SAM3D seed | Still no seed in the HaMeR/HaWoR call ([run\_ego\_wilor.py](https://github.com/nvidia-isaac/video_to_data/blob/v0.4.0/reconstruction/modules/v2d_pipelines/run_ego_wilor.py)) | Target stands |
| Flips | No symmetry in the ego tracker; re-register from scratch at IoU < 0.3 | Target stands |
| One object | Unchanged (top Grounding DINO box) | Target stands |
| Stereo depth | [`v2d_foundation_stereo`](https://github.com/nvidia-isaac/video_to_data/tree/v0.4.0/reconstruction/modules/v2d_foundation_stereo) exists (TensorRT, 576×960 ONNX), used only by the multi-view and HOI object pipelines | Reuse it; less to build |
| Multi-frame mesh | [`v2d_hoi_object_reconstruction`](https://github.com/nvidia-isaac/video_to_data/blob/v0.4.0/reconstruction/modules/v2d_hoi_object_reconstruction/docker/run_reconstruction.py) SAM3D mode: seeded SAM3D on several frames, silhouette scale fit, optional stereo depth, best-candidate pick | Reuse for the mesh stage |
| BundleSDF | Present in the same module, but it expects a scan (object still → rotated → still) | Fallback only: a 13 s ego clip rarely shows every side |
| Symmetry code | [`multiview_tracker.py`](https://github.com/nvidia-isaac/video_to_data/blob/v0.4.0/reconstruction/modules/v2d_foundation_pose/lib/multiview_tracker.py) already accepts a symmetry group | Adapt it to one camera |
| Hands | New `--hand_tracking hawor` option | Use HaWoR as the hand source |
| Grounding | `run_e2e.sh`: retarget to Sharpa Wave (`floating`) → RL expert or FlashCHORD → optional GR00T N1.7-3B fine-tune | Replaces our CHORD / G1 + Dex3 assumption; skip the GR00T step, since the scored policy is one FlashCHORD checkpoint per episode |

## What we must beat

Track 3 scores only the final object trajectory and our reconstructed mesh, so object-pose accuracy against Vicon is the right local proxy.

| Item | Fact | Source |
| --- | --- | --- |
| Scored output | Object trajectories from the learned policy + reconstructed object meshes, for all 20 eval videos | [FAQ](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf) |
| Metrics | AUC, SP-SR, MP-SR, RPE, MPPE (pose) and CD-O (object mesh: surface Chamfer distance in cm) | [FAQ](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf), [submission kit](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip) `tracks/track_3/README.md` |
| Ranking | Sum of per-metric ranks; lowest wins | [FAQ](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf) |
| Intermediates | Not assessed | [FAQ](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf) |
| Meshes | Must be our own reconstructions, even where the public set has scans | [FAQ](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf) |
| External models/data | Allowed in all tracks | [FAQ](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf) |
| Eval inputs | `ego_cam_a` (2028×1520 colour) + `ego_cam_b/c` (1280×800 mono stereo) | [eval videos](https://huggingface.co/datasets/nvidia/video_to_data_challenge/tree/main/track_3/evaluation/videos/chunk-000) |
| Local GT | 20 episodes, 5,246 frames; `world_T_object` in scan frame; x,y centred per episode, z absolute | [README](https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/main/track_3/README.md) |
| Objects per episode | 1–2, occasionally more | [README](https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/main/track_3/README.md) |
| Calibration gap | No camera-to-mocap extrinsic is provided | [README](https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/main/track_3/README.md) |
| Deadline | Leaderboard freeze Nov 4, 2026, 5 p.m. EST; 5 submissions/week | [challenge page](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/) |

The policy can only track the trajectory we hand it, so reconstruction error caps every policy metric.

## Baseline error budget

Six design choices in v0.4.0 HaMeR/HaWoR mode cap object-pose accuracy; each row is a target Path B must fix. Causes are traced in our v0.4.0 walkthrough (project doc) against the [v0.4.0 tag](https://github.com/nvidia-isaac/video_to_data/tree/v0.4.0); step numbers are unchanged from 0.2.0.

| Error | Code-level cause (v0.4.0) | Path B fix |
| --- | --- | --- |
| Object scale off | Mesh size fitted to MoGe depth on one reference frame (steps 7–9) | Metric stereo depth on every frame; scale fitted over many frames |
| Orientation flips | FoundationPose re-registers from scratch when IoU < 0.3, no temporal prior, no symmetry info (step 10) | Re-register only among hypotheses near the last good pose; symmetry class supplied |
| Only one object | Only the top Grounding DINO box is tracked (step 5) | VLM lists every manipulated object; SAM 3 tracks each instance |
| Hallucinated mesh | SAM 3D from one image, unseeded in HaMeR mode (step 7) | Seeded SAM3D on several frames, scaled and ranked on stereo depth (reuse the HOI module's SAM3D mode) |
| No metric world frame | Identity or DROID first-camera world; gravity step only rotates (steps 3, 16, 18) | Camera trajectory with stereo scale; world = table plane + gravity, matching GT's absolute z |
| Wrong intrinsics | Learned 4-param fisheye estimate; principal point forced to image centre (steps 0, 2) | Use the shipped 14-coefficient factory calibration |

## Candidate A: feed-forward 4D reconstruction

Strong components exist, but no single feed-forward model outputs metric multi-object 6-DoF poses and meshes from ego video. A is a parts list, not a system.

| Model | What it gives us | Limit for Track 3 |
| --- | --- | --- |
| [FoundationStereo](https://arxiv.org/abs/2501.09898) (NVIDIA) | Zero-shot stereo depth; trained on \~1M synthetic pairs | Needs the rectified b/c pair; mono cameras, so no colour |
| [Depth Anything 3](https://arxiv.org/abs/2511.10647) | Depth + ray maps + camera pose from any number of views; beats VGGT by 35.7% pose, 23.6% geometry | Main model is not metric; metric depth is a separate model |
| [ViPE](https://arxiv.org/abs/2508.10934) (NVIDIA) | Intrinsics, camera motion, near-metric depth from raw video; 3–5 FPS on one GPU | "Near-metric" only; stereo is stronger for scale |
| [EgoMono4D](https://arxiv.org/abs/2411.09145) | Feed-forward intrinsics, poses, video depth for ego video | Scene geometry only; no object pose or mesh |
| [EgoGrasp](https://arxiv.org/abs/2601.01050) | World-space hands + objects; best object rotation error on its benchmarks | Itself a multi-stage pipeline (DA3, WiLoR, SAM 3D, diffusion); code release not stated |

**Verdict:** use DA3/ViPE and FoundationStereo as components, not as the framework.

## Candidate B: VLM agent in the loop

A VLM helps where the baseline makes brittle discrete choices, but evidence says it must not judge geometry or grade its own output.

**Where a VLM adds value (semantic, discrete):**

- Name every manipulated object from the video, replacing the single hand-typed prompt. [SAM 3](https://arxiv.org/abs/2511.16719) then segments and tracks all instances of each noun phrase, and roughly doubles prior concept-segmentation accuracy.
- Pick a reference frame per object where it is unoccluded.
- Declare a symmetry class (e.g. "cup: rotational about its axis"), which FoundationPose currently never receives.
- Mark interaction phases (grasp, carry, release) for the grounding stage.
- [Qwen3-VL](https://arxiv.org/abs/2511.21631) fits: open dense sizes from 2B to 32B, video timestamps, box/point grounding.

**Where it must not be trusted:**

- Metric geometry. VLMs are unreliable at distance and viewpoint reasoning, and naive tool use sometimes lowered scores ([spatial reasoning gap](https://www.alphaxiv.org/overview/2606.00963)).
- Self-verification. In grounding, a VLM's confidence correlated with real IoU at only r ≈ 0.22, and every deployable iterate-and-fix policy scored below single-shot ([self-correction mirage](https://arxiv.org/abs/2606.13156)).

**Verdict:** keep the VLM, but gate every result with geometric checks (render IoU, stereo-depth residual), never with the VLM's opinion.

## Candidate C: video to policy (VLA / WAM)

Rejected for this deadline: it cannot produce what Track 3 scores, and our data is too small to train it.

- **Missing outputs.** Track 3 needs a reconstructed mesh plus object trajectories ([FAQ](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf)). A VLA outputs actions; we would still need reconstruction for the mesh and the target trajectory.
- **Compute.** [GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T) fine-tuning is no longer the blocker: v0.4.0's own workflow fine-tunes GR00T N1.7-3B on one 48 GB A6000. But the official evaluator scores one FlashCHORD checkpoint per episode, not a GR00T policy, and any policy still learns from a reconstructed trajectory.
- **Data.** 20 public episodes over 10 tasks is a tiny fine-tuning set.
- **Still useful later.** N1.7 is pretrained on 20K hours of EgoScale human video with a Cosmos-Reason2-2B (Qwen3-VL architecture) backbone. Worth revisiting after the challenge for the Industry Innovation angle.

## Comparison and pick

The A+B hybrid is the only option that produces every scored output, fits 1–2 GPUs, and directly removes the baseline's known errors.

| Criterion | A: feed-forward | B: VLM agent | C: VLA / WAM | **A+B hybrid (pick)** |
| --- | --- | --- | --- | --- |
| Mesh + object trajectory | Partly (geometry only) | Only via tools | No | **Yes** |
| Multi-object | No | Yes | n/a | **Yes** |
| Metric scale | Depends on depth model | No | n/a | **Yes (stereo)** |
| Fixes flips | No | Partly (symmetry class) | n/a | **Yes** |
| Training needed | None | None | Heavy fine-tune | **None (inference only)** |
| Fits 1–2 GPUs | Yes | Yes (2B–8B VLM) | Yes (48 GB A6000), but needs reconstruction | **Yes** |
| Risk by Nov 4 | Medium | Medium | High | **Medium** |

Assessments are our judgement from the sources cited in the sections above.

## Proposed architecture

One entrypoint runs video to trajectory + mesh; only the gate in the middle decides what is kept.

*Architecture diagram (VLM planner, metric geometry, segmentation → object mesh + pose tracking → geometric checks → world frame + output) is in the [live doc](https://claude.ai/code/artifact/0873d10c-acc3-455c-936a-12baf370fabf).*

The VLM only feeds discrete choices into tracking; every pose passes the geometric gate before it reaches the world frame and the policy stage.

## Local evaluation vs Vicon

Score every change with the team's `tools/track3/score_local.py`, which runs the kit's unmodified metric code on the 20 public episodes; a component swap is accepted only if it improves those scores.

1. **Align frames.** As the official scorer does: one rigid world transform (rotation + shift, no scale) set from frame 0 of the anchor object, so no camera-to-mocap extrinsic is needed. Rollout translations are never rescaled, so trajectory scale error stays in the score.
2. **Pose metrics.** AUC, SP-SR, MP-SR, RPE and MPPE, with the kit's symmetries: cups match under any turn about their axis, the pot lid under a half-turn.
3. **Mesh metric.** CD-O: surface Chamfer distance (cm) between our mesh and the scan, after a separate Sim(3) registration of each mesh.
4. **Run-to-run variance.** Three seeds per episode; report mean and spread.
5. **Ablate one change at a time** against the baseline (NVIDIA `8c778f9`, whose reconstruction code matches v0.4.0) on the same episodes.

The scans are used only for scoring, never as submitted meshes, per the [FAQ](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf). The [submission kit](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip) is in the team repo at `assets/data/v2d_submission_kit/`.

## Plan, compute, risks

Four stages fit before the Nov 4 freeze, ordered so the biggest expected gain lands first and each stage is shippable alone.

1. **Oct 13–18: scorer + metric depth.** Local Vicon scorer; factory-calibration undistortion; `v2d_foundation_stereo` depth swapped in for MoGe inside the v0.4.0 ego flow. First leaderboard submission.
2. **Oct 19–25: multi-object + flips.** VLM object list → SAM 3 tracks; per-object FoundationPose with a symmetry group (adapted from `multiview_tracker.py`) and pose-gated re-registration.
3. **Oct 26–Nov 1: mesh + world frame.** Mesh from the HOI module's SAM3D mode (seeded candidates from several frames, `--sam3d_use_depth` with stereo); BundleSDF only if an episode shows enough sides. Camera trajectory; table-plane + gravity world.
4. **Nov 2–4: freeze.** Seed selection; v0.4.0 grounding (retarget to Sharpa Wave → RL expert or FlashCHORD); final submissions (unlimited in the last three days).

**Compute:** everything is inference on 1–2 GPUs; Path B's VM is a single L40S (48 GB), which NVIDIA verified for the whole v0.4.0 workflow alongside the A6000; one heavy job at a time. FlashCHORD training with Edmond's njmax 1024 settings needs at least 80 GB, so it stays on Babel.

**Risks:**

- Stereo is mono 1280×800 and offset from `ego_cam_a`; depth must be reprojected into the colour camera using the shipped extrinsics.
- The shipped FoundationStereo engine takes 576×960 input, so stereo depth is lower resolution than the colour frames.
- `ego_cam_a` calibration lists `rotation_deg: 180`; check frame orientation before detection runs.
- Hands still come from the baseline's trackers (HaMeR or the new HaWoR); Path B does not improve them, and grounding needs them.
- The kit publishes the alignment rule, but our local scorer estimates the hidden anchor points and symmetry axes from the scans, so local numbers can differ slightly from the leaderboard's.

## Open questions

- [x] Embodiment: Sharpa Wave (Track 3 default per the FAQ; v0.4.0 retargets to it as `floating`, `tools/e2e/cli.py`).
- [x] Stereo stream (`ego_cam_b/c`) is allowed: confirmed by NVIDIA, as reported by James on 2026-10-18 (the message itself is not on file).
- [ ] Can the public scans be used for training or shape retrieval, or only for local scoring? The FAQ only requires submitting our own meshes. The team uses scans in `score_local.py`, `bundles.py audit` and the opt-in `--rescale-by-scan`; the 2026-10-16 still submission did not use rescaling (`tools/track3/jobs/still_plan_2026-10-16.json`, `still_submit.sh`).
- [x] World alignment: one rigid transform (rotation + shift, no scale), set from frame 0 of the anchor object; each mesh is registered to the scan separately, scale included. Refs: submission kit `tracks/track_3/README.md` L72–75; `metric_code/track_3/AUC.py` `_align_registered_scene` (L476), `_score_registered_track3` (L824); project doc `track3_scorer_guide.md` Part 2 §2–3.
- [x] "Object Mesh" metric = CD-O: symmetric surface Chamfer distance in cm, after a separate Sim(3) registration of each mesh to the scan. Refs: kit `tracks/track_3/README.md` L100; `metric_code/track_3/CD-O.py`.
- [x] Learned policy = one FlashCHORD checkpoint per episode, run by the official evaluator; GR00T is not needed. Refs: kit `examples/run_policy_evaluation.py` L2–4, L35; `flash_chord/src/flash_chord/configs/evaluation/flash_sac.yaml` L1.

All kit paths are under `assets/data/v2d_submission_kit/` in the team repo (`main_buffer` @ `c83f9eb`). Because rollout translations are never rescaled, trajectory scale stays the lever Path B's stereo depth targets.

## Sources

Pages opened Oct 11, 2026; v0.4.0 code checked Oct 13, 2026.

- [V2D Challenge page](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/)
- [V2D Challenge FAQ (PDF)](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf)
- [Dataset card](https://huggingface.co/datasets/nvidia/video_to_data_challenge) · [Track 3 README](https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/main/track_3/README.md) · [Track 3 eval videos](https://huggingface.co/datasets/nvidia/video_to_data_challenge/tree/main/track_3/evaluation/videos/chunk-000)
- [video\_to\_data release/0.2.0](https://github.com/nvidia-isaac/video_to_data/tree/release/0.2.0/)
- [CHORD (arXiv 2607.00033)](https://arxiv.org/abs/2607.00033)
- [FoundationStereo (arXiv 2501.09898)](https://arxiv.org/abs/2501.09898)
- [FoundationPose (arXiv 2312.08344)](https://arxiv.org/abs/2312.08344)
- [BundleSDF (arXiv 2303.14158)](https://arxiv.org/abs/2303.14158)
- [SAM 3 (arXiv 2511.16719)](https://arxiv.org/abs/2511.16719)
- [Depth Anything 3 (arXiv 2511.10647)](https://arxiv.org/abs/2511.10647)
- [ViPE (arXiv 2508.10934)](https://arxiv.org/abs/2508.10934)
- [EgoMono4D (arXiv 2411.09145)](https://arxiv.org/abs/2411.09145)
- [EgoGrasp (arXiv 2601.01050)](https://arxiv.org/abs/2601.01050)
- [Qwen3-VL technical report (arXiv 2511.21631)](https://arxiv.org/abs/2511.21631)
- [Self-correction mirage in VLM grounding (arXiv 2606.13156)](https://arxiv.org/abs/2606.13156)
- [The spatial reasoning gap in VLMs](https://www.alphaxiv.org/overview/2606.00963)
- [Isaac GR00T N1.7 README](https://github.com/NVIDIA/Isaac-GR00T)

v0.4.0 code checked:

- [video\_to\_data v0.4.0 tag](https://github.com/nvidia-isaac/video_to_data/tree/v0.4.0)
- [run\_ego\_wilor.py](https://github.com/nvidia-isaac/video_to_data/blob/v0.4.0/reconstruction/modules/v2d_pipelines/run_ego_wilor.py)
- [v2d\_hoi\_object\_reconstruction run\_reconstruction.py](https://github.com/nvidia-isaac/video_to_data/blob/v0.4.0/reconstruction/modules/v2d_hoi_object_reconstruction/docker/run_reconstruction.py)
- [v2d\_foundation\_stereo](https://github.com/nvidia-isaac/video_to_data/tree/v0.4.0/reconstruction/modules/v2d_foundation_stereo)
- [multiview\_tracker.py](https://github.com/nvidia-isaac/video_to_data/blob/v0.4.0/reconstruction/modules/v2d_foundation_pose/lib/multiview_tracker.py)
- [e2e\_workflow.md](https://github.com/nvidia-isaac/video_to_data/blob/v0.4.0/docs/e2e_workflow.md)
