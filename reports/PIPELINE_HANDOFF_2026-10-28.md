# V2D Track 3: the whole pipeline, from the top (handoff, 2026-10-28)

Written so you can pick the work up from a different directory without this session. Sources:
the repo README (§5–6), `tools/track3/{BUNDLES,NIGHT_RUN_2026-10-24}.md`, `tools/track3/jobs/README.md`,
`path_b/README.md`, git history up to `0d9ec0e9`, and the live notes in
`reports/submission_candidate_2026-10-28/{README,ACTIVE_WORK}.md` (last entry 04:05 EDT today).

---

## 0. Read this first: where things live, and what is not in git

| What | Where | In git? |
|---|---|---|
| Code (tools, FlashCHORD fixes, job scripts) | branch **`edm_training`**, pushed, HEAD `0d9ec0e9` | yes |
| Current submission work: selection, training launcher, checkpoint monitor, repaired scenes, packed parquets, Kaggle receipts | `~/NVIDIA_V2D_Egocentric/reports/submission_candidate_2026-10-28/` | **no** (`reports/` is in `.git/info/exclude`) |
| Older reports (bundle audit, ep41 review, oracle, stereo checks) | `~/NVIDIA_V2D_Egocentric/reports/` | no |
| Heavy data: Apptainer images, weights, bundles, training runs, venvs, Slurm logs | `/data/user_data/edmondwa/v2d/` (compute nodes only, not visible from the login node) | no |
| Shared bundles + night-run outputs | private HF dataset `rose-tea-cafe/v2d-t3-recon` | no |
| Left untracked on purpose | `Track 3 Pipeline Status.html` (old status page source), `best_meshes_20261024.tar.gz` (37 MB mesh tarball) | no |

**If your new directory is a fresh clone or a worktree, `reports/` will not be there.** The scripts that
run the current submission loop (`refresh_tracking.py`, `train_tracking.sbatch`, `checkpoint_monitor.py`,
`evaluate_saved.py`, `checkpoint_pairs.py`, and the rest) exist only in `reports/submission_candidate_2026-10-28/`.
Either keep working from the original checkout for that part, symlink the folder in, or copy it over
(`rsync -a ~/NVIDIA_V2D_Egocentric/reports/ <new>/reports/`). Promote the scripts that proved useful into
`tools/track3/` and commit them when you get a chance.

Worktree recipe (what the night-run sheet used):
```bash
cd ~/NVIDIA_V2D_Egocentric && git fetch origin
git worktree add ../v2d_work origin/edm_training   # or: -b <new-branch>
ln -s ~/NVIDIA_V2D_Egocentric/baseline/reconstruction/data ../v2d_work/baseline/reconstruction/data
mkdir -p ../v2d_work/assets/data && ln -s ~/NVIDIA_V2D_Egocentric/assets/data/track_3 ../v2d_work/assets/data/track_3
ln -s ~/NVIDIA_V2D_Egocentric/reports ../v2d_work/reports
```

**Pushing from Babel:** `git push` fails because the Git LFS pre-push hook tries to authenticate to
NVIDIA's upstream LFS store (`.lfsconfig` points there). If your commits add no LFS files, use
`git push --no-verify origin <branch>`. `gh` is not installed on Babel, so open PRs from the web
(compare URL: `https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/compare/main_buffer...<branch>`).
Babel's home filesystem was very slow today. Even `git status` took more than 90 s, so run git in the background or be patient.

---

## 1. The challenge in one page

- **NVIDIA Isaac Video-to-Data (V2D) Challenge, Track 3: Egocentric Video → Policy.** For each scored
  episode, a head-mounted video of human hands manipulating 1–2 objects on a table. We must output
  **per-frame 6-DoF object poses produced by a robot policy rollout in simulation** (Sharpa robot hands,
  FlashCHORD simulator), plus a mesh for every object.
- **Submission** = one parquet (kit's `pack_submission.py`), scored on Kaggle across 6 metrics/comps:
  AUC (ADD-AUC, main), SP-SR, MP-SR (3 cm / 30°, strictest), RPE, MPPE, CD-O (mesh Chamfer).
- **Data** (`assets/data/track_3/`, HF `nvidia/video_to_data_challenge`): 20 **public** episodes with
  ground truth + scans, 20 **evaluation/scored** episodes with video only. ~262 frames per clip at
  **20.254 fps**. 16/20 scored episodes have two objects.
- **Leaderboard freeze: 2026-11-04, 5 pm EST.** Submission quota: the README says 5 per week. Today's notes
  track "AUC slots", with 3 left on 10-28 at the last check. Confirm the quota on Kaggle before uploading.
- **Baseline:** NVIDIA `video_to_data` `main@8c778f9`, imported unmodified as commit `d9264a4` into `baseline/`.
  The kit's evaluator runs FlashCHORD at exactly this revision.

### Team / repo layout
- `baseline/`: NVIDIA's code plus marked in-place fixes ("Path A"). Each edited file carries
  `# Modified for V2D Track 3 by …` headers.
- `path_a/`: Path A's new code (mostly empty; Path A work went into `tools/` + `baseline/`).
- `path_b/`: James's (GitHub `zjifan121`) independent "v3" pipeline (see §9).
- `tools/`: shared tooling with its own uv env (`uv run --project tools …`). Almost all our code is in `tools/track3/`.
- `assets/`: the official challenge material, unmodified (kit, docs, dataset).
- Branch flow: personal branch → PR → `main_buffer` (default) → PR → `main`. Tag Kaggle submissions `submit/a-YYYY-MM-DD-N`.

---

## 2. The pipeline: five stages

```
 video ──► 1 RECONSTRUCT ──► bundle ──► 2 LOAD ──► 3 RETARGET ──► scene ──► 4 TRAIN ──► policy ──► 5 EVALUATE + SUBMIT
          (per object)     result.npz   (MANO →    (Sharpa IK,      parquet   (FlashSAC,           (kit evaluator rollout
                           + mesh.obj   parquet;    URDF, table;    + URDF    100M steps)           → pack → Kaggle)
                                        merge 2     repairs)        + table
                                        objects)
```

Terms:
- **Bundle**: one reconstruction run's final output for one object: `result_slam_gravity_aligned/{result.npz, mesh.obj, material.*, manifest.json}`. Per-frame camera, object and both-hand poses.
- **Scene**: what FlashCHORD trains on: the ManoSharpa parquet under
  `assets/human_motion_data/ego_recon/processed/sequence_id=<seq>/robot_name=sharpa_wave/`, plus object URDF(s) and the support (table) surface.
- **Reference**: the object/hand motion the policy is asked to track (= the reconstruction after stage 2–3 repairs).
- **Own-reference AUC / "tracking"**: the policy's rollout scored against *its own reconstructed reference*. It is what we can measure on scored episodes. It is **not** the hidden Kaggle score.
- **Namespace (NS)**: a folder under `baseline/reconstruction/data/outputs/<NS>/episode_XXXXXX/<object>/`, one per reconstruction recipe (`fs_sam3d`, `fs_hands`, `fs_noref`, `ken_c_scored`, `bm_scored`, …).

### Stage 1: Reconstruct (video → bundle)
NVIDIA's `run_ego_reconstruction.py` (HaMeR mode): undistort → Grounding DINO/SAM masks → MoGe depth → DROID-SLAM
camera → gravity alignment → SAM3D mesh (or provided mesh) → FoundationPose object tracking → HaMeR hands
aligned to depth → (optional) gsplat refinement. Each step runs in its own container; on Babel we use
**Apptainer `.sif`** images (no Docker; podman build fails; `apptainer pull docker://…` works).

What the team changed or added for Track 3 (all on branches merged into `main_buffer` or on `edm_training`):

| Change | Why | Where |
|---|---|---|
| **FoundationStereo metric depth** from the real stereo pair (75 mm baseline, factory calibration) as the seed, instead of MoGe | Metric scale is the #1 score lever (×1.2 scale drops AUC 1.0 → 0.78) | `tools/track3/stereo_seed.py`, `jobs/seed.sbatch`, `fs.sbatch`, `run.sbatch <ep> <obj> fs_sam3d` |
| **Late hands** (`--late_hand_prompts` / `--late-hands`) | Hands entering after the reference frame were never reconstructed, and they often carry the object | `run_baseline_reconstruction.py`, `jobs/hands.sbatch`, `hands_next.sh` |
| **SAM 3 text-prompted object masks** (`--object-segmenter sam3`, Haisen's PR #19) | Grounding DINO grabbed wrong objects (e.g. "mini sweeper" tracked the dust pan) | `jobs/sam3_run.sbatch` |
| **No gsplat refinement** (unrefined bundles) | Refinement jittered the camera path (sometimes >6× the SLAM path) | `write_unrefined_bundle.py`, `fs_noref` NS |
| **Donor meshes**: track with the best public-episode mesh of the same object (`--object_mesh … --skip_object_scale_estimation`) | SAM3D meshes vary run to run and are often the wrong size | `jobs/reuse_mesh.sbatch`, `swap_mesh.py`, night run §3 |
| **Multi-key-frame mesh pick for `wooden_spoon`** (12 SAM3D meshes = 6 frames × 2 seeds, pick by mask overlap and depth on held-out frames) | Spoon has no public scan but is in 10/20 scored episodes | `mesh_candidates.py`, `select_mesh.py`, `jobs/mesh_cand.sbatch`, `tools/track3/r78/` |
| Mesh proportion fitting to masks+depth | Better CD-O | `fit_shape.py`, `eval_mesh.py` |

Traps (each one cost real score):
- **Seed folder:** `stereo_test/seed_fs/ep<N>` holds FoundationStereo depth (`median_scale` 1.0). `stereo_test/seed/ep<N>` holds MoGe rescaled to stereo (~0.7) and scored 0.15 on ep41 vs 0.84.
- **Never rotate frames 180°.** Calibration says `rotation_deg: 180`, but the released videos are already upright.
- **Two-object episodes:** one run per object, **same reference frame** for both, merged in stage 2.
- Episodes 11 and 18: per-frame object order ≠ metadata order. **Always match objects by name** (`tools/track3/episodes.json` uses per-frame order).
- `visible == False` → pose is all zeros; filter on it.

Quality bar per bundle (`bundles.py audit`; full table in `tools/track3/BUNDLES.md`): frame count = video;
object valid ≥ 95%; mesh/scan size 0.8–1.25×; hand scale 0.7–1.5 (hands are the metric ruler);
`run.json` + an overlay mp4 present. Check identity visually: `check_identity.py` projects the mesh onto frame 0.

### Stage 2–3: Load + Retarget (bundle → scene)
One command: **`tools/track3/run_load_retarget.py`** (runs NVIDIA's `onboard_ego_clip` recipe natively, no Docker/Isaac).
Env: `/data/user_data/edmondwa/v2d/venvs/retarget` (Py 3.11; recipe in README §5.12; needs MANO pkls).

- **Load**: MANO forward → finger joints; hands + object into one world; writes the human-motion parquet. The loader used to hardcode `EGO_FPS = 30`; it now reads `V2D_EGO_FPS` (the runner sets 20.254).
- **Retarget**: Sharpa hand IK, object URDF (CoACD collision pieces), table/support surface, optional QA video.
- **Repairs (default on)**, recorded under `cleaning` in `report.json`: hand cleaning (refill wrists beyond reach or >2 m/s), object cleaning (interpolate >1.5 m/s or >10 rad/s), rest hold (object still until a wrist is within 10 cm), level (rotate so the object starts resting on a face; `--no-level` turns it off).
- **`--merge`** (two-object episodes): merges both objects' bundles into one scene, keeping the steadier run's SLAM world and placing the other object by its camera-frame pose.
- **`--attach-hands --smooth-hands 2`**: required for the training recipe. It attaches each hand to one object at a time (commit `dd22f373`). **Known limitation found today:** attach preserves the best wrist/object offset rigidly even when it is 20–30 cm, so fingers may never reach the object (see §6).
- Never train with `--still-object`, `--rescale-by-scan` or `--rescale-by-hands` (GT-derived or safety-net only).
- Fix pushed today (`620b3636`): when `--bundle` is an already-merged multi-object bundle, body names now come from its `object_names`.

Typical call (as in the night run):
```bash
V2D_BUNDLE_NS=$NS $RV/bin/python tools/track3/run_load_retarget.py --episode $EP --merge \
  --attach-hands --smooth-hands 2 --skip-video --device cpu \
  --sequence-id ep${EP}_${NS}_attach --out $OUT        # one object: --object <name> instead of --merge
```
Scene helpers: `scene_variant.py` (object physics / table probes), `relocate_scene.py` (scene built on another machine),
`support_stacked.py`, and the WIP repair tools in §6.

### Stage 4: Train (FlashCHORD / FlashSAC)
Env: `/data/user_data/edmondwa/v2d/venvs/flash_chord` (or `baseline/robotic_grounding/flash_chord/.venv`; `jax[cuda13]==0.10.2`).

**Team default recipe** (commit `36d1a430`; curriculum `tracking_100m_object2x`):
- 100,003,840 env steps, all curriculum thresholds ×0.4 of NVIDIA's 250M schedule; helper force gone by ~49M; frame-0 starts in the last ~20M
- object keypoint reward ×2 (`object_keypoints: 40`)
- `task.motion_speed=1.0 sim.fps=20.254119873046875` **(mandatory, see the frame-numbering trap below)**
- `sim.njmax=1024 sim.nconmax=512` (2048 for scored eps 1, 10, 25); `termination.wrist_position.threshold=0.3`
- Current submission loop adds: object termination 0.3 m / 1.5 rad ("OT"), replay 1,048,576 with **float16 observations**, full replay+learner state saving for preemption resume

**Frame-numbering trap (confirmed):** the evaluator numbers rows by control step of the *resampled* reference,
and the kit packer treats that number as a video frame. With NVIDIA's `motion_speed: 0.5`, the packer silently
submits the first half of the motion. With speed 1.0 and `sim.fps` = the parquet's float32 fps, step *k* = video frame *k*.

Throughput: ~21.6k steps/s on A6000, ~29–31k on faster cards. A 100M run takes ~40–55 min on L40S/A100/H100.
`njmax 1024` with 1M float16 replay fits in ~22 GB (A6000 measured), so L40S is fine.

Launchers:
- `tools/track3/jobs/scene_train.sh`: stage 2–3 + `train_diag.sbatch` (self-requeuing on `preempt`). E.g.
  `NS=fs_noref TAG=obj2x_s42 LR_CPU=1 SCORE=1 tools/track3/jobs/scene_train.sh 12`
- `tools/track3/jobs/gpu_queue.sh`: no-Slurm queue, one worker per GPU (used on the H100 night run)
- `reports/submission_candidate_2026-10-28/train_tracking.sbatch`: **current** launcher (full state resume, per-run flock, TERM exits 143; only USR1 requeues). `checkpoint_pairs.py` keeps two matching learner+replay pairs.
- `tools/track3/train_object_radius.py` (new, WIP): training-only override of the object keypoint radius (env `V2D_OBJECT_KEYPOINT_RADIUS_M`; upstream 1 m). Evaluator unchanged.

FlashCHORD library fixes on `edm_training` (evaluator code hash must stay unchanged; these are training-side):
- `0f466fb5`: non-finite transitions are kept out of FlashSAC replay (zeroed + terminated)
- `6bc05388` (today): finite float32 obs that overflow float16 storage are rejected too; 25 replay tests pass on CPU

### Stage 5: Evaluate + submit
- Kit evaluator (`evaluate_policy.py`) rolls out the policy deterministically, without helper forces, from frame 0. Our wrappers: `evaluate_saved.py` (saved checkpoints at 50/75/90/95M…), `score_rollout.py`, `select_policies.py`.
- **Do not edit `baseline/robotic_grounding/flash_chord/` evaluator files or the kit.** The evaluator hashes its own code (`eval_code_sha256`, expected `b1b3…`, script hash `160d…`).
- Selection: `refresh_tracking.py` scans `flash_runs/diag` + imported night-run runs (including `early_*` checkpoint evals), validates coverage, fps, hashes and finite poses, excludes still/half/oracle/scan references, and **requires ep25 actual mesh-bottom lift ≥ 5 cm on both boards**. It writes `plan.json`, `tracking_inventory.json`, `packed/track3_submission.parquet`.
- Pack and preflight with the kit, run through the tools env (the kit's requirements miss pillow and scipy):
  `uv run --project tools python tools/pack_submission.py …` from the kit folder, then `preflight.py --skip-kaggle`. Expect 20 episodes / 9120 rows.
- Uploaded candidates are **immutable** snapshots: `packed/candidate_<mean>.{parquet,plan.json,receipt.json}`. Never overwrite them.
- `tools/track3/pack_submission.py` accepts single-object scenes (commit `8e124e61`).

Local scoring on public episodes (no simulation): `score_local.py` (kit's official metric code, `--self-test`),
`score_playback.sh` (reconstruction played back), `dryrun_public.py`. Self-test reference points: GT 1.0; still
object 0.447; ×1.2 scale 0.775. These public scores use scans only for scoring and never for choices.

---

## 3. Compute: Babel

- GPU jobs → `preempt` partition, `--qos=preempt_qos` **always explicit** (sbatch inside a maxlab-cpu allocation inherits the wrong QOS). Use `--gres=gpu:1` with a VRAM constraint. Current pool is narrowed to `L40S|6000Ada|H100|H200` (A100 80GB used too). Limits: 100 queued / 24 running per user.
- `maxlab` Blackwell (sm_120) GPUs: unusable for reconstruction images, fine for FlashCHORD, but need Max's approval. Not used.
- CPU work → `maxlab-cpu`. Scene builds: ~8 cores, ≤32 GB each.
- **Cancelling self-requeuing jobs:** a plain `scancel` can come back. Use `scancel --signal=KILL --full <id>`, then plain `scancel`, then verify with `sacct` (or hold before cancelling). Older scripts have the requeue trap. New `train_tracking.sbatch` doesn't.
- **Don't kill `herdr` jobs:** their purpose and approval aren't established.
- Disk: `/data/user_data/edmondwa` at ~445/500 GB (56 GB free at 04:05). Each run keeps 2 learner+replay pairs (~6.6 GB/run). Never use `/data/group_data/maxlab/common_datasets`.
- Reading W&B histories: copy the binary files to `/tmp` first (`datastore.open_for_scan` opens `r+b` and fails on read-only `/data`).
- The VS Code login node can't see `/data`. Put anything you want to browse under `~/NVIDIA_V2D_Egocentric/reports/`.

Data roots under `/data/user_data/edmondwa/v2d/`: `sif/` images, `recon_data/{outputs,weights}` (symlinked
into `baseline/reconstruction/data`), `stereo_test/{seed,seed_fs,logs}`, `flash_runs/diag/<run>`, `venvs/`,
`jobs/logs/`, `ken_night/…` (imported H100 night-run results).

---

## 4. History: how we got here

| Date | Milestone |
|---|---|
| 10-12/13 | Baseline synced to `8c778f9`. Apptainer route found. Manifest, local scorer, episode runner built. Frame-numbering trap found and confirmed. |
| 10-14 | Stage 2–3 runner reproduces NVIDIA's tissue_box retarget exactly. First real bundle (ep41) shows reconstruction is the bottleneck: AUC 0.28 < frozen object 0.36. |
| 10-15/16 | Bundle-sharing hub + audit. "Still" safety-net submission plan. |
| 10-17 | GT-reference oracle runs: how far the training stage can go with a perfect reference. |
| 10-18–23 | Stereo (FoundationStereo) seed, late hands, SAM 3 masks, unrefined bundles, mesh candidates/reuse. Path B v3 built by James. Short 100M training schedule. |
| 10-24 | 100M object×2 recipe made default; attach-hands-per-object; **H100 night run** (Ken's machine): scored + public reconstruction with donor meshes, 3 seeds per scored episode. |
| 10-25–27 | Relocate night-run scenes; scene variants (object physics, edge-aware table); non-finite replay guard. |
| 10-28 (night) | Submission loop: selection, repaired grasps for ep3/ep25, ep25 slot fit + finer collision pieces, wider-exploration and early-frame-0 trials, automatic checkpoint evaluation. Two Kaggle uploads. |

### Kaggle scores (hidden AUC)
| Submission | Candidate (own-ref mean) | Hidden AUC |
|---|---|---|
| v2 (earlier) | (n/a) | 0.42455 |
| 57726860 | `candidate_0792945` (0.7929) | **0.64008** |
| 57729024 | `candidate_0807696` (0.8077) | **0.65379** (best) |

Current selection (`plan.json`, not uploaded): **own-reference mean 0.811621** over the 20 scored episodes.
The live goal from last night: keep iterating until the own-reference mean is ≥ 0.90 with **real** ep25 manipulation,
with no stationary references and no edits to object motion to inflate tracking.

---

## 5. Current state per episode (04:05 EDT, 10-28)

Weakest episodes by own-reference AUC (before the latest gains): **33** (.463, now WIDE/early75M .511), **19** (.491 → .584),
**25** (now .827), **5** (.583 → .602), **30** (.645), **3** (.669 → .740), **10** (.690), **20** (.707), **1** (.773).

- **ep25 (wooden_piece_1 + wooden_piece_2, assembly):** the big win. Selected `ep25_assembly2_OT1M_s42/early_90005504` = **0.827** (boards .859/.795). Both boards really lift (mesh bottoms 12.4/10.8 cm vs ref 14.5/11.6). Assembly is still imperfect: board 2 ends 5.8 cm above its reference. Scene chain: `ep25_grip2_scene` (right-hand hold repaired; keep this first-board behaviour) → `ep25_pinch2_scene` (robot wrist+fingers rigid to board 2) → `ep25_assembly2_scene` (slot pose fit 9.6 mm + 4.5° after frame 190, plus CoACD threshold 0.02 / no preprocessing = 104 collision pieces, so the slots aren't filled). **Training regresses after 90M** (100M final .632). Keep intermediate checkpoints. `grip3` (both-hands repair) failed badly (.29); don't build on it.
- **ep3 (white_pot + wooden_spoon):** left hand was 24–28 cm from the pot while it really lifts ~6 cm. `ep3_grip_scene` puts the left hand on the solid handle (hold 45..265). Best `ep3_grip_s43/early85004288` .740. Don't freeze the pot.
- **ep33 (blue_cup + wooden_spoon):** still the worst. Radius 5 cm (R005) was worse (.248). Rigid grasp / lead-grasp variants failed nominal tests: the approach knocks the cup aside, and the wrist lags 17–18 cm (Sharpa wrist kp 25 / kd 6). Pad-only IK doesn't certify a grasp, because palm and phalanges collide too. Running: `ep33_WIDE_EARLY_NOUNINT_s42` (10965436, unintended-contact penalty zeroed via the new `tracking_100m_object2x_no_unintended.yaml`), with WIDE_EARLY as its matched control.
- **ep19 (mini_sweeper + mini_dust_pan):** sweeper falls ~1 m off a small support. Full-table + object-termination run (`ep19_ST_OT_s42`) is being tested; 75M = .323, not better. Don't apply the ep30 table alignment here (the pan needs its local levelling).
- **ep30 (mini_sweeper + plastic_dish_rack):** `align_table_reference.py` levels the scene to the measured stereo table plane (11 fits, 9.9° tilt). The full-table trial didn't beat the selected .645.
- **ep5:** radius pair: R100 .550; R005 still being evaluated. Best .602 (kv_ep5_OT_s42/early60M).

### Jobs that were live at 04:05 (check `squeue -u edmondwa`; IDs in the JSONs next to ACTIVE_WORK.md)
- 10965496 `ep25_assembly2_LOWLR_s42`: resumes the 95M state at constant LR 5e-5, to 125M (tests the post-90M regression)
- 10965461 `ep25_assembly2_OT1M_s43`: second seed of the winning ep25 scene
- 10965436 `ep33_WIDE_EARLY_NOUNINT_s42`: unintended-contact ablation
- 10964736/10964737 WIDE, 10964959/10964960 WIDE_EARLY (ep25/ep33), 10964131 ep19 full table: older trials finishing
- **10965501 checkpoint monitor** (maxlab-cpu, 3 h, started ~04:00): auto-submits official evals of 50/75/90/95M (LOWLR: 100–125M) for trials listed in `checkpoint_monitor_trials.json`, at most 2 outstanding. Don't duplicate its evals by hand. It doesn't refresh the plan or upload anything.

"WIDE" = wider exploration: linear action mapping, initial std 0.3, log_std_max log(0.6), target sigma 0.3.
"EARLY" = also `reset.reset_to_first_frame_probability=0.5`, `reset.first_frame_voc_threshold=1.1` (train the pickup from frame 0 at every stage).
So far neither beats the defaults (ep25 WIDE 90M .626, EARLY 75M .602; ep33 EARLY 75M .272).

---

## 6. WIP tools pushed today (`0d9ec0e9`), all unproven as policy fixes

| Tool | What it does | Status |
|---|---|---|
| `tools/track3/repair_hand_contact.py` | During an inspected hold, translates one hand by one object-local offset so the thumb/index pinch midpoint sits on a thin object's midplane edge. Object/camera arrays unchanged. | Made `ep25_grip2` (→ .827 chain) and `ep3_grip` (→ .740) |
| `tools/track3/stabilize_robot_grasp.py` | IK on the actual Sharpa collision-pad vertices at an inspected anchor, then holds that robot/object transform rigidly through the hold (`--grip SIDE:OBJ:START:END:ANCHOR`, `--compression-mm`, `--digits`, `--max-wrist-shift-m`) | ep33 variants failed nominal tests. Needs full-hand + dynamic validation |
| `tools/track3/align_table_reference.py` | Rotates the whole scene to the stereo-measured table normal, undoing stage 2's per-object tilts | ep30 trial, no gain |
| `tools/track3/train_object_radius.py` | Training-only keypoint-radius override | ep33 R005 worse. ep5 pair pending |
| `configs/curriculum/tracking_100m_object2x_no_unintended.yaml` | Default 100M recipe with `unintended_contact: 0` in all 11 stages | ep33 trial running |

Experimental scripts that are still only in `reports/submission_candidate_2026-10-28/` (not in git):
`refine_pinch{,2}.py`, `install_board_collision{,2}.py`, `decompose_boards*.py`, `build_board_fit_scene*.py`,
`optimize_full_hand_{,surface_}grasp.py` (offline, never applied; best fit still 2.46 mm bone penetration),
`audit_donor_spoon_grasp.py`, `record_successful_spoon_grasp.py`, `register_spoon_grasp.py` (transfer ep37's
successful spoon grasp to ep33, not applied yet), `debug_nominal_contacts.py`, `evaluate_nominal.py`
(zero-residual "nominal" test of a reference), `transfer_*_checkpoint.py`, `verify_reset_sampling.py`.

---

## 7. Rules we hold ourselves to

1. **No ground truth or scans in any choice** (meshes, seeds, runs, references). Public scores are for reporting only.
2. **No hold-still / stationary references** and no edits to object motion to raise tracking. Selection excludes still/half/oracle/scan refs. ep25 must actually lift both boards.
3. **Evaluator and kit untouched** (hash check). Training-side changes only, each recorded in checkpoint metadata.
4. Partial, reset-tail or warm-start diagnostic rollouts are **never** selected.
5. Don't add seeds when the reference hand can't reach the object. Fix the reference first and verify it with a nominal test.
6. A "nominal test" (zero residual actions) or a static pad audit is a diagnostic, not proof that a grasp works.
7. Don't overwrite uploaded candidate files. Repack into a new `candidate_<mean>` before any upload.

---

## 8. Resume checklist (in order)

1. `squeue -u edmondwa` and `sacct -S 2026-10-28`: what finished or died since 04:05.
2. `tail reports/submission_candidate_2026-10-28/checkpoint_monitor.log`. The monitor job ends ~07:00, so resubmit `checkpoint_monitor.sbatch` if trials are still producing checkpoints (add new trials to `checkpoint_monitor_trials.json`).
3. `tools/.venv/bin/python reports/submission_candidate_2026-10-28/refresh_tracking.py` → new mean and `plan.json`.
4. Compare: ep25 LOWLR (100–125M) vs 90M; ep25 seed 43; ep33 NOUNINT vs WIDE_EARLY; ep5 R005 final.
5. If the mean improved meaningfully: repack → kit preflight (20 eps / 9120 rows) → new immutable `candidate_<mean>` → upload (check quota first) → tag `submit/a-2026-10-2X-N`.
6. Biggest open problems: the ep33 grasp approach (cup knocked aside; try the ep37 donor spoon grasp with dynamic validation); ep25 final assembly (board 2 high) and post-90M regression; ep19 sweeper falling.
7. Disk: keep > 30 GB free on `/data/user_data/edmondwa`.
8. Promote proven scripts from `reports/` into `tools/track3/`, commit, push (`--no-verify`), and PR `edm_training` → `main_buffer`.

---

## 9. Path B (James), for context

`path_b/` is an independent end-to-end pipeline ("v3", spec `path_b/V3_BUILD_SPEC.md`, log `BUILD_LOG.md`), rebuilt
from the three head-camera videos: rectified stereo → FoundationStereo depth → SAM 3.1 masks → DROID-SLAM with stereo
prior → WiLoR/HaMeR hands → SAM 3D multi-view mesh + metric size → FoundationPose → ESKF/RTS fusion → phases →
"Do as I Do" retarget in MuJoCo Warp → FlashCHORD task → train/eval/select → pack. Every stage writes a `report.json`
with PASS/WARN/FAIL measurements, plus a score waterfall (raw → stage 2 → stereo scale → scan scale). Many GPU stages
were "smoke test pending" as of 10-23. It calls `baseline/` read-only and reuses `run_load_retarget.py` as its fallback
for stages 2–3. Run: `uv run --project path_b python path_b/run.py run --episodes 41`.

---

## 10. File index (most useful first)

| File | Purpose |
|---|---|
| `README.md` §5–6 | Setup, reconstruction, stages 2–3, data facts, traps |
| `tools/track3/jobs/README.md` | Babel job order (seed → fs → run → hands → scene_train → score) |
| `tools/track3/BUNDLES.md` | Bundle standards, HF hub upload/download/audit |
| `tools/track3/NIGHT_RUN_2026-10-24.md` | Full worked run: env setup, donor table, queue, upload |
| `tools/track3/episodes.json` | Every episode: frames, objects (scorer order), prompts |
| `tools/track3/run_load_retarget.py` | Stages 2–3 |
| `tools/track3/score_local.py`, `score_playback.py`, `score_rollout.py`, `select_policies.py`, `pack_submission.py` | Scoring, selection, packing |
| `reports/submission_candidate_2026-10-28/README.md` + `ACTIVE_WORK.md` | Current numbers + detailed experiment log (dense; newest sections at the bottom) |
| `reports/submission_candidate_2026-10-28/{plan.json,tracking_inventory.json,packed/}` | Current selection and uploaded candidates |
| `assets/data/track_3/README.md` | Dataset card |
| `baseline/.claude/skills/onboard_ego_clip/` | NVIDIA's own stage 2–3 recipe |
