# Path B v3: spec conformance audit

`path_b/` at `0dd1d357` (branch `zjifan121_pathb_v3`, 2026-10-22) against `path_b/V3_BUILD_SPEC.md`. The spec
is not edited; where the code is right and the spec is stale, the row says so ("spec stale") for James.
"Decision" cites `BUILD_LOG.md` → Decisions by its date and title. Status: **same**, **different** (built,
not as written), **missing** (not built yet), **not run** (built, waiting for the VM).

| Spec item | What was built | Status | Why |
|---|---|---|---|
| **§0 Rules** | | | |
| 0.1 Nothing under `baseline/` or the kit changes; modules called as they are | No tracked file under either changed (`git status` clean). The Mac's stages 2–3 venv left git-ignored `*.egg-info` under `baseline/` (editable installs, as on the VM), and `git lfs pull` restored the committed Sharpa meshes and body models in place of their pointers | same | Decision 2026-10-22 "The Mac gets the stages 2–3 venv (CPU) and the robotic_grounding LFS assets" |
| 0.2 GT and scans only in `score_local` and public-guarded stages | `pathb/gt.py` is the only reader (`allowed()` raises on evaluation); `tests/test_gt_firewall.py` checks every file. The team runner's own scan read is blanked on evaluation episodes (`drivers/run_lr.py`, `PATHB_NO_SCANS`) | same | The runner gap was found in task 14 (S5, s23) |
| 0.3 `report.json` through `report.py`; `loss_report.md` names the first FAIL/ERROR | As written; stand-in stages write the same schema plus `stand_in` | same | Decision 2026-10-22 "Stand-in GPU outputs on the Mac" |
| 0.4 Rerun only with `--force` or a missing/ERROR report | As written, plus: once a stage reruns, every later requested stage reruns (`runner.run_episode`) | different | Keeps later stages from using outputs of an older input; the stand-in run hit it (rerunning P3/P4 rebuilt S3a) |
| 0.5 One heavy model at a time; `pathb` imports only numpy/scipy/cv2/trimesh | One model at a time, yes. `pathb` also imports pyarrow (S5 parquet), open3d + scikit-image (A2), rtree (trimesh queries) | different | S5's parquet rewrite and A2's TSDF/ICP/Poisson need them; all CPU and in `path_b/.venv` |
| **§1 Repository** | | | |
| Move v2 files to `cache/v2/`; drop `s4_scene` | Done (task 1, `ae1f0e08`) | same | |
| New `runner.STAGES` | As listed. Phase A2's `s3a2_fuse_scan`, `s3a2_register` sit in `runner.ORDER` (after S4a, before S4b) but not in the default chain | different | Decision 2026-10-22 "A2: stages run only when named" (§11 says A2 replaces S3 only if it wins) |
| `run.py` gains `pack`, `oracle` | Both built (`pack`: task 16). Also added: `standin` (Mac stand-ins), `canonical` (A2.2), `analysis`, `a2`, `p2` | different | The extra commands come from the stand-in decision, step 2 and §11/§8 |
| `waterfall` gains `keyframe`, `realised`, `rollout_<candidate>` | Columns `fused`, `stage2_clean`, `ref_{keyframe,tracking,still}`, `realised_{keyframe,tracking}`, `rollout_{…}`, every bundle column again as `<column>@scan`; baselines `0fa525bc stage2_stereo`, S0.2 recon/place and still | different | Decision 2026-10-22 "Waterfall columns" and "Waterfall: stage2_clean and <column>@scan" (spec stale: `keyframe`/`realised` are per candidate) |
| README §Stages rewritten | Rewritten in task 1; the State column is stale (P3/P4 still say "v3 in task 9") | different | Not updated after tasks 9–15 |
| **§2 Environments** | | | |
| `path_b/.venv` | As written, with the extra packages of §0.5 | same | |
| baseline host venv + GHCR images | `vm/setup.sh` builds them (12 images + FoundationPose TRT); not on the Mac | same | |
| `sam3` venv | `setup.sh` builds it (`v3_envs`); not yet built on a VM | not run | VM blocked (SSH key refused, then a new host key) |
| `mvsam3d` | Not installed; S3a uses the spec's fallback (best single view by stereo fit) | missing | Spec's own fallback rule; not installable without a VM |
| `daid` venv | `setup.sh` builds it (clone under `/root/v2d/third_party/`, `COMMITS.txt`) | not run | VM |
| `flash_chord` venv | Used by S0.1, S3a (CoACD), S6 | same | |
| `tools/.venv` runs `score_local`, `run_load_retarget`, packer | `score_local` is imported in `path_b/.venv` (`pathb/team.py`); `run_load_retarget` runs in the stages 2–3 venv (`$RETARGET_VENV`, README §5.12), on the Mac too (CPU) | different | Spec stale: the runner needs torch, pinocchio and USD, which `tools/.venv` does not have |
| **§3 Conventions** | | | |
| Frames, `%06d`, stereo offset via `calib.stereo_frame` | As written | same | |
| Camera OpenCV; world table z = 0, +z up, +y camera forward on the table | As written; the origin is the table point under the first rest camera | different | Decision 2026-10-22 "v3 world origin" (spec leaves the origin open; Kaggle's frame-0 alignment removes it) |
| Poses on disk `Transform3d`; arrays (N,4,4); parquet quaternions xyzw | As written (FlashCHORD's ManoSharpa parquet is wxyz, as its loader requires) | same | |
| Depth PNG inverse encoding | As written; Path B writes "no depth" as 0, which `DepthImage` decodes as infinity: DROID, SAM3D and the FP driver handle it, `align_hands` does not (P1d reports the share) | same | Code note in `pathb/geometry.py`, P1d metrics |
| Masks 8-bit, 255 = object, one folder per track | As written (`obj_<i>` in roster order, `hand_L`, `hand_R`) | same | |
| Hands via `tools/track3/read_mano.py` | The baseline loader's MANO (`EgoReconDatasetLoader.load_mano_data`, `drivers/mano_world.py`) | different | Decision 2026-10-22 "`tools/track3/read_mano.py` (spec §3) does not exist" (spec stale) |
| **S0 Oracles** | | | |
| S0.2 regime split | Done (Edmond's `oracles/oracle_2026-10-17.json`) | same | |
| S0.1 GT-reference policy on ep40, ep26 | `pathb/oracle.py` + `run.py oracle`, training/eval shared with S6 (`pathb/policy.py`); in `vm/dev_cache.sh` | not run | VM |
| **S1 Evidence** | | | |
| P0, P1, P2 unchanged | Unchanged; ep40/ep26 outputs from run `4c2eac31` (restored with frames) | same | |
| P1b: SAM 3.1 text prompts, SAM2 + Grounding DINO fallback, seed frame (conf ≥ 0.35, area ≥ 0.5 %), propagation both ways | As written (`drivers/sam3_masks.py`, `drivers/sam2_masks.py`), plus: SAM 3.1 without FlashAttention 3 off Hopper GPUs; `init_state` gets only the arguments it declares; the second hand's seed instance must differ from the first's (IoU ≤ 0.5); hand masks lose pieces farther than 4 % of the width from the hand's blob | different | Smoke test 2026-10-23 (decisions "ep40 smoke test, P1b/P1d fixes" and "hand jitter"): SAM 3.1 never ran on the L40S without the first two; both hand prompts found one hand; far pieces moved HaMeR's crop and the wrist ~15 cm |
| P1b outputs, bars (mask ≥ 95 % / 80 %; hand mask on ≥ 90 % of WiLoR frames, checked in P1d; 8 overlays) | As written, plus two bars: SAM 3.1 ran (fallback = WARN), left/right hand overlap (pass ≤ 0.5, warn ≤ 0.8) | different | The fallback and a shared hand mask passed silently on the first smoke run |
| P1c: DROID with `input_depth_folder` + `align_to_depth_folder` on masked stereo depth | As written; the prior stays at half resolution | different | Code reason (`p1c_camera.py` docstring): DROID resizes it to 384×512 anyway |
| P1c bars: speed, plane-height spread over the rest window, scale within 3 %, public camera-path comparison | As written; the rest window is the 2-D one (frames before a hand mask touches an object mask, ≤ 30); the scale check is DROID's own scale correction against stereo | different | Decision 2026-10-22 "The rest window of P1c (and of P3 v3)" (S4a's window needs the track, which comes later) |
| P1d: WiLoR → HaMeR → align → interpolate through `run_baseline_reconstruction.py --late-hands` | Each baseline step through its own wrapper; late hands come from P1b's both-way mask tracks | different | Code reason (`p1d_hands.py` docstring): P1b already gives each hand its own track, which is what `--late-hands` does |
| P1d outputs and bars | As written (`hand_scale.json`, `handedness.json`, `overlay.mp4`; 90 %, 0.8–1.2 / 0.6–1.5, 2 m/s on 99 %); records converted to the host layout (`camera.cam_t`); `align_hands` gets a depth copy with hand-mask holes at the valid median; non-finite and > 0.2 m spike fits rejected to `tracks_interpolate`; HaMeR / align folders cleared each run | different | The GHCR images write the older record layout, which the host bundle writer would read as hands at the camera; the baseline modules skip existing frames. ep40 after the fixes: wrist speed 1.00 / 1.00 |
| GPU smoke tests of P1b, P1c, P1d on ep40/ep26 | Coded; on the Mac their outputs are stand-ins | not run | VM; decision "Stand-in GPU outputs on the Mac" |
| **S2 One metric world** | | | |
| P3: factor = stereo / P1c camera height, bar 0.97–1.03; `scale.json` with factor, rest window, table plane, gravity | As written. Stand-in run: 1.00 on ep40 and ep26 | same | |
| P4: v3 bundle (P1c camera, P1d hands, table plane, gravity), no mesh | As written (baseline writer with a placeholder triangle, removed) | same | Decision "v3 world origin" |
| **S3 Object model and track** | | | |
| S3a view selection (≥ 2 %, hand overlap ≤ 5 %, ≤ 5 views, angular spread, a rest-before and a rest-after view from S4a's phases) | As written, with the 2-D rest windows instead of S4a's | different | S4a needs S3's track; same 2-D rule as P1c |
| S3a SAM 3D per view with the stereo pointmap, seed 0 | Through the module's depth route (`depth_path` + intrinsics + mask): its own `_depth_to_pointmap` makes the PyTorch3D pointmap | different | Decision 2026-10-22 "S3a calls run_image_to_mesh with depth_path" (spec stale: `pointmap_path` expects OpenCV axes) |
| S3a MV-SAM3D fusion | Not wired; best single view by stereo-depth fit (the spec's fallback) | missing | Not installed; spec's fallback rule |
| S3a metric size (uniform fit, axes 0.8–1.25), outputs, bars | As written (uniform 0.5–2×, axes 0.8–1.25; CoACD 0.05; `mass.json`; `sha256.txt`) | same | |
| S3b FoundationPose driver in the container, no scale search, registration on the largest-mask view, forward + backward | As written (`drivers/fp_track.py`; renders the centred mesh for its IoU), with `--dev` as the spec says: the GHCR image predates the TensorRT backend, so the host's modules run inside it (compiled extensions kept) | same | Code note: the baseline's `_mask_iou` renders the centred mesh at the uncentred pose; the driver renders both centred. ep40: WARN (largest rotation step), 79–88 s |
| S3b recovery rule (symmetry candidates + register top-5, 2 iterations, score with λ 2.0, μ 50; particles when hand overlap > 40 %) | As written | same | |
| S3b outputs and bars | As written | same | |
| S3c: anchors (median, chordal mean, residual ≤ 2 cm), ESKF/RTS on world poses, anchor and recovered noise, optional Point2Pose stream; completes the bundle | As written | same | |
| S3c with no anchor frame | No anchor written; S4b then takes the window's rest anchor over the fused track | different | Decision 2026-10-22 "S4b: a missing S3c anchor falls back…" (spec silent) |
| GPU smoke tests of S3a, S3b | Coded; stand-ins on the Mac | not run | VM |
| **S4 Reference construction** | | | |
| S4a phases (10 cm, ±5 dilation, carry ≥ 10 unless static) | As written: first reach / last release are the raw near frames, the ±5 dilation sets the windows | same | James 2026-10-21 ("raw reach") |
| S4b outputs at `s4b_reference/<candidate>/` | `s4b_reference/<object>/<candidate>/`, plus `reference_report.json` per object with the per-frame provenance | different | James 2026-10-21 ("subfolders"); provenance added for S6's evaluation rule |
| S4b keyframe / tracking / still, hands cleaned with the runner's hand cleaning | As written (`pathb/reference.py`, `hold_until` both ends, Hermite + slerp, `cleaned_bundle(hand_clean only)`) | same | |
| S4c adapter, DaiD stages 2–5, `--max-sim-steps 0`, knobs in `s4c.yaml` | As written; the reference resampled to 50 Hz, the table as the MuJoCo floor, warm-up = rest-before, reference padded by the horizon; MuJoCo Warp buffers 512 / 2048 per world; MPC 256 samples, 1.6 s horizon, 0.4 s control step, ≤ 16 iterations | different | Decisions 2026-10-22 "S4c settings", 2026-10-23 "S4c MPC settings cut": DaiD's buffers overflowed on ep40 and its override took > 5 h per object (James may restore it) |
| S4c embodiment | Bimanual when both hands are valid on ≥ 50 % of frames, else the better hand | different | Decision 2026-10-22 "S4c embodiment" (spec silent) |
| S4c out: `trajectory_mjwp.npz` with contacts | Contacts come from a MuJoCo replay of DaiD's result at the video frames, in ManoSharpa's 16 MANO-link slots, with wrists and link poses | different | Decision "task 14: Contacts" (DaiD writes no contacts) |
| S4c gate and checks; evaluation episodes take the public decision per task type | As written (`oracles/s4c_gate.json`) | same | |
| `sharpa_map.json` | 22 joints per hand match by name, order and limits; kinematics agree to 0.2 mm | same | |
| S4c smoke test | ep40 on the L40S (2026-10-23): runs with the cut settings, ~20 min per candidate; DaiD's own settings ~2.7 h per candidate | same | |
| **S5 Task file** | | | |
| Mode B columns, FK of `robot_*_frames` with FlashCHORD's URDF | FK by `pathb/fk.py` from the retargeter's MJCF (the 67-frame list every Sharpa parquet carries), checked against FlashCHORD's URDF to 1e-7 m | different | Decision "task 14: robot_{side}_frames from the retargeter's MJCF" |
| Mode B contacts "from DaiD's contact output", part ids by nearest MANO part | Replayed contacts; slot = the MANO link of the touching Sharpa link; deepest per slot; normals into the object; part id 1 | different | Decision "task 14: Contacts" |
| Mode B runner flags | `--no-rest-hold --no-object-clean --no-level` (mode A keeps the spec's `--no-rest-hold`) | different | Decision "task 14: Mode B runner flags" |
| Mode A fallback; still with `--still-object` | As written; two-object episodes are mode A (merge) in phase 1 | same | Decision "task 14: Two-object episodes stay mode A" (agrees with spec §8) |
| Scene files via the runner | As written; on the Mac too (`$RETARGET_DEVICE=cpu`) | same | |
| S5 acceptance: FlashCHORD kinematic replay to the end; fingertip–object distance matches S4c contacts within 1 cm on 90 %; runner checks | Runner checks (`lr_check.py`: links > 2 cm inside, seating gap, support disk) and the contact hand-over check (DaiD's robot-side contact points carried by the task file's robot, ≤ 1 cm from the object on ≥ 90 % of contact frames) built; `scripts/debug/replay.py` not wired | missing | The replay needs FlashCHORD's GPU simulator; the hand-over check is its CPU proxy (S5 docstring) |
| **S6 Policy** | | | |
| s6_train: FlashSAC, `still_submit.sh` EXTRA, njmax/nconmax 512, 60,006,400 steps, checkpoint recorded | As written (`pathb/policy.py`, resumable) | same | |
| s6_eval: kit evaluator, one world, explicit reset, trajectories | As written | same | |
| s6_select public: best AUC, ties → MPPE; acceptance beats still | As written (tie within 0.001 AUC) | same | |
| s6_select evaluation: lowest rollout-vs-reference error from the evaluator JSON, termination frames penalised | Rollouts against one shared observation (S4b keyframe reference on held/trusted frames, frame-0 anchor alignment, symmetry-aware, frames > 10 cm / 0.5 rad cost 10 cm); also run on public episodes to check it against the score | different | Decision "task 15: evaluation episodes select by the rollout vs one shared observation" (the JSON's own-reference error always favours still) |
| S6 on ep40, ep26 | Coded | not run | VM |
| **S7 Pack and submit** | | | |
| Preflight, pack, upload, tag, HF | `run.py pack` (`pathb/preflight.py`): preflight (selection + rollout, mesh geometry S3a = URDF = packer, GT firewall on evaluation episodes, git clean + pushed), then the team packer (`tools/track3/pack_submission.py`, which runs the kit's packer) and the kit's `tools/preflight.py --skip-kaggle`; `MESH WARNING` lines collected. No upload, tag or HF step in `pack` | different | Team packer reused (rule 4): it relabels bodies to roster names and calls the kit's packer, so the kit command is not called by hand. Upload and tag are on HOLD (James); HF upload is `hf_outputs.py` after the run. Dry pack on the public split waits for phase B's public half |
| **§6 Run plan** | Day plan not followed: no VM job has run since task 2 (the instance refused the Mac's key, then showed a new host key) | different | VM provisioning wait |
| **§7 Compliance** | | | |
| GT firewall: `pathb/paths.py` gains `gt_allowed(ctx)`; `files_read` per stage | `pathb/gt.py` (`allowed`, every reader), `files_read` in every report, test over all files | different | Placement only (spec stale: `gt.py`, not `paths.py`) |
| Mesh hash S3a → S5 → S7 | Preflight compares a geometry hash (vertices rounded to 1 µm, sorted, SHA-256) of S3a's mesh, the URDF's mesh and the packer's mesh | different | The three are different files (S3a OBJ, the loader's OBJ, the runner's score OBJ); equal file hashes cannot happen, equal geometry can (decision, task 16) |
| `manual_inputs.json` per episode | Not written | missing | No hand-chosen parameter exists yet; nothing writes the (empty) file |
| `THIRD_PARTY.md` | Not written | missing | Not done; licences known (DaiD MIT, SAM 3.1 SAM licence, FoundationStereo/FoundationPose NVIDIA, Open3D MIT) |
| **§8 Phase 2** | Not started | missing | After A2 (James, 2026-10-22 follow-up: A2 first) |
| **§9 Workflow** | | | |
| Local-first; CPU stages run on the Mac on the real ep40/ep26 data | CPU stages ran on the Mac on ep40/ep26 with real P0–P2 and stand-in GPU outputs (run `standin_d6e79acb`) | different | Decisions "Build locally first" and "Stand-in GPU outputs on the Mac" (the dev cache has not run) |
| Dev cache = v3's own GPU stages + S0.1, uploaded and restored | `vm/dev_cache.sh` (one tmux command; P0–P2 from `4c2eac31`, chain to S4b, S0.1, upload with frames) | not run | Decision "Supersedes the dev-cache decision above"; VM |
| Per task: local commit, then cloud smoke test, `BUILD_LOG.md` | Local commits and log rows for every task; no cloud step since task 2 | different | VM |
| GitHub pushes only at phase boundaries | Nothing pushed since PR #16 (task 1, opened under the 2026-10-21 per-task workflow) | same | |
| Three waits | Kept (VM, push, Kaggle) | same | |
| **§11 Phase A2** | | | |
| A2.1 partial scan (TSDF 4 mm / 16 mm, rest frames, bars) | Built; the rest-after window is fused at the rest-before placement; P4's camera; mask erosion 2 px; held-out RMS from a second fusion | different | Decisions "A2.1: rest-after frames are fused where the object rested before" and "A2.1: P4's camera…" |
| A2.2 canonical mesh (ICP, symmetry fit and completion, Poisson, ≤ 20k faces; fallbacks SymmCompletion → AdaPoinTr → ForeHOI) | Built: clip registration (normal-checked ICP, point-to-point then point-to-plane), vertical-start revolution fit with the revolved generatrix, vertical mirror planes, table cap, Poisson or a voxel solid; learned fallbacks not wired | different | Decisions "A2.2: what completion does where the spec is silent" and "A2: ICP pairs only normal-compatible points" |
| A2.3 registration | Built; phase A's track re-expressed for the canonical mesh; bar on the share of the partial explained | same | Decision "A2: stages run only when named" (bar) |
| A2.4 mesh-free tracking (Point2Pose / TAPIP3D) | Not built | missing | Next (James's step 3a: code now, run on a VM) |
| A2.5 comparison and decision | Not built | missing | Needs A2.1–A2.3 on ep40/ep26 (step 3a) |

## Differences that change what Kaggle sees

Kaggle scores the rollout poses and the submitted mesh. These differences change one of them relative to
the spec as written:

1. **S6 evaluation-split selection** (shared observation instead of each candidate's own reference): decides
   which rollout each evaluation episode submits.
2. **S5 mode B runner flags** (`--no-object-clean --no-level`): the object trajectory FlashCHORD trains on
   when S4c's realised trajectory is used.
3. **S5 mode B contacts** (MuJoCo replay, MANO-link slots): the contact supports FlashCHORD's wrench terms
   train against.
4. **S3a without MV-SAM3D** (best single view): the submitted mesh (CD-O), FoundationPose's template and
   the collision body.
5. **S3a views and P1c's rest window from the 2-D rule**: which views make the mesh, and the window P1c's
   checks use.
6. **P1c prior at half resolution**: the camera path, hence every world pose.
7. **S4b's anchor fallback** when S3c has no anchor frame: the rest poses of every reference.
8. **S4c settings and embodiment rule** (spec silent): the realised trajectory, when the gate takes it.
9. **A2** (built, not adopted): would replace the mesh and the track for everything after A2.5 if it wins.
10. **S7 mesh check by geometry, not file hash**, and the team packer instead of the kit command (task 16): a mesh that changed between S3a and the packer still stops the pack.

## Differences that are internal only

World origin (removed by Kaggle's frame-0 alignment); `robot_*_frames` from the MJCF (FlashCHORD's Sharpa
embodiment does not read them); `read_mano.py` replaced by the same loader code; GT firewall in `gt.py`
and the runner's scan blanking; per-object S4b folders and provenance; waterfall columns (`stage2_clean`,
`@scan`); A2 stages outside the default chain; the rerun cascade; extra `pathb` imports and the
`run_load_retarget` environment; P1d's direct wrapper calls; S3a's SAM 3D depth route (same pointmap);
S5's hand-over check in place of the replay; stand-ins, the Mac venv and LFS assets (dev episodes only);
`hf_outputs.py` archiving per-frame folders; README's stale State column; missing `THIRD_PARTY.md` and
`manual_inputs.json` (compliance paperwork, no effect on scores).
