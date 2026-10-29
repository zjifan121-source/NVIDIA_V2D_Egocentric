# Path B task queue (2026-10-29; mirrors V4_BUILD_SPEC.md §14)

Numbers in two columns: playback / submitted (rollout ≈ playback − 0.07…0.10). Today 0.598 / ≈ 0.50; this queue 0.70–0.74 / 0.60–0.66. Public first. Evaluation runs (L10, G4, G5) stay paused until James lifts the pause (2026-10-28). Rules unchanged: push `zjifan121_pathb_v3` after every passing test; no merge to `main_buffer`, no Kaggle upload without James; no GT or scans on evaluation.

## Local (Mac)
| # | Task | Effort | Check | Expected (playback) | Status |
|---|---|---|---|---|---|
| L1 | `run.py analysis` on the 8 mid episodes (12 2 42 45 13 22 27 31): rests vs carries | 0.5 h | timelines | scopes L5 | done: AUC loss carries 49 %, rests after a carry 28 %, still second objects 15 %, initial rests 9 % ⇒ L5 in scope |
| L2 | ep21 depth check: masked depth vs table depth on 20 rest frames | 0.5 h | behind-table ⇒ L3 | gate | done: not see-through (88-90 % in front, heights match the scans); cause = S3c's TSDF target without the below-table filter (cup fitted to the floor, 1.06 m off); fix on all 20 running (`v4_j1`) |
| L3 | Silhouette term (static fit when the held-out depth check fails; pass 2 at low weight) | 1 day | ep21 up; 22/12/13 not down; silhouette IoU | ep21 → ≈ 0.5 | todo |
| L4 | Upright prior + concentric stacking for revolution objects at rest; log overrides | 0.5 day | ep11 up, none down | ep11 → ≈ 0.6 | todo |
| L5 | Carry-only joint refinement (if L1 says carries) | 0–1.5 days | mid MPPE, MP-SR | +0.03–0.05 | todo |
| L6 | Hand-mesh non-penetration (hand scale ≥ 0.85 or inflated) | 0.5 day | no regression 41/44 | with L5 | todo |
| L7 | Tracked bridge for in-hand turns (clip 5 mm / 3°, end terms, no wrist prior); ep0 lid grasp by hand mesh | 0.5 day | ep43 up, 41/45 not down | ep43, ep0 → ≈ 0.5 | todo |
| L8 | Boards / sweeper at 1–2 mm voxels; scorer registration 10×; stop if flips remain | 0.5 day | flip rate | ≤ +0.02 | stopped: boards flip 1-8 of 10 at 2 mm, sweepers and every other object 0/10 |
| L9 | Six metrics in every checkpoint compare | 1 h | — | — | done `982c3f6e` |
| L10 | Evaluation CPU chain S3b → S5 on 20 episodes (method frozen at the lift), GT-free signals | 3–4 h | firewall | submission possible | **paused** |
| L11 | Kit packer for S7; dry-pack public today | 2 h | preflight | — | done `a7219a86`: dry pack of 40/26/18 on the VM, kit preflight "Ready to submit." |

## GPU (L40S)
| # | Task | GPU-h | Check | Status |
|---|---|---|---|---|
| G1 | F2 on public continues; `njmax`/`nconmax` ≥ 1024 before two-object episodes | — | no overflow lines | done `24793d99`: njmax 1024 / nconmax 512 (1024 / 1024 out of memory); F2 resumed at ep41 06:15 UTC |
| G2 | Selector decision after the first four rollouts | 0 | accuracy on four | todo |
| G3 | Swap improved public references per episode; retrain those | ≈ 1 h/ep | rollout vs playback | todo |
| G4 | Evaluation: 60M × 2 × 20, then 250M on the chosen reference | ≈ 48 h | GT-free gate | **paused** |
| G5 | Pack (kit packer), preflight, first evaluation submission (James's go); resubmit by Nov 3 | 0.5 h | preflight | **paused** |

Schedule: Oct 29 L1 L2 G1 L11 L9 L3-start · Oct 30 L3 L4 L5/L6 · Oct 31 L7 L8 · L10 → G4 → G5 take ≈ 3 days once the pause is lifted (lifted Oct 30: submission Nov 2, resubmission Nov 3; lifted Nov 1: 60M only).

## Research additions (2026-10-29, verified; see V4_BUILD_SPEC.md §14.4)
| # | Task | Where | Check | Expected (playback) | Status |
|---|---|---|---|---|---|
| R0 | ep21 cause corrected: depth is fine (median 0.44 m in the pitcher mask); the pitcher has no rest-before window because the hand track is the same-hand failure, and the cup anchor registers ambiguously. Re-seed hands on ep21/22/44 (E1), rerun ep21; B2 margin for the cup | L40S 15 min/ep + Mac | ep21 up; 22/44 not down | ep21 → ≈ 0.5 | todo |
| L3' | Silhouette term stays as a low-weight pass-2 term and as the static residual where the held-out depth check fails; no longer the ep21 fix | Mac | no episode down | small | todo |
| R1 | FoundationPose refinement from the inverted trajectory (baseline container, no new licence): per frame `track_one` iterations from the inverted pose, canonical mesh, RGB-D, mask; keep only where depth residual and silhouette IoU improve | L40S ≈ 3 min/ep + Mac driver | mid group MPPE −1 cm; MP-SR flips on 2 12 13 22 27 45 | +0.03–0.05; MP-SR 0.40 → 0.55+ | todo |
| R2 | GoTrack refiner (RGB flow render-and-compare, HOT3D ADD-S ≈ 90) — CC BY-NC 4.0, James decides; only if R1 underdelivers; needs R3 | L40S | as R1 | as R1 | hold |
| R3 | Textured canonical mesh (colour TSDF) for RGB refiners | Mac | — | — | hold |

## History (tasks 0–15 of the 2026-10-26 queue, as the coding session left them)

| # | Task | Where | Check | Status |
|---|---|---|---|---|
| 0 | Upload the Mac-only results to HF `path_b/runs/v4` (ep38 S1, ep2 rerun, 19 public CPU chains, canonicals; results only, no frame archives) with the refreshed token; commit and push `BUILD_LOG.md` | Mac | `hf_outputs.py list` shows 40 episodes | partial: 0.32 GB of results blocked by HF's storage limit (`usedStorage` 100.0 GB); the refreshed read token is not on the Mac yet |
| 1 | A1 pooling gates: drop partials failing the held-out depth check; axis prior = table normal, reject > 20°; pot clips aligned by handle azimuth, reject > 30°; reject clips > 0.3 cm from the pool | Mac | explain-share ≥ 0.8 per clip; public CD-O; dimensions vs the 9 scans | done 2026-10-26: CD-O and dimensions fixed for the cups, pot, pitcher; explain-share fails where the voxel solid closes (pot, rack, cups): to A2/A5 |
| 2 | A2 neighbour-free scans; 2 mm voxels for rack, sweeper, boards | Mac | public Chamfer ≤ 0.3 on cup/pot pairs | done 2026-10-26: + below-table filter (the table patches' cause); Chamfer ≤ 0.3 on 48 of 55 partials (was 43); misses: ep11/ep12 cup one window each, pitcher |
| 3 | Download evaluation S1 results (~30 GB); A3 pool over all 40 clips; keep 40-clip vs 20-clip pool per object by explain-share | Mac | wooden_spoon has a mesh | done 2026-10-26: spoon 4.2 × 17.3 × 32.5 cm (width suspect, no scan); 40-clip pool kept for 10 objects, 20-clip for blue_cup |
| 4 | A4 three mesh candidates per object (pooled / own clip / team SAM 3D); A5 completion keeps real asymmetric geometry | Mac | explain-share, B1 margin | done 2026-10-26: pooled 23 / own 6 / team 2; chosen CD-O 0.507 vs all-pooled 0.479 (explain-share favours the own mesh). 2026-10-28 (`6fbe65ed`): the team SAM 3D candidate removed (James: no team work in the pipeline); it had been chosen for ep22's cup only |
| 5 | B1 registration-ambiguity test with `score_local.py`'s registration; B3 frame-0 bar (< 5 mm / 1° vs the settled static pose) | Mac | public: flagged ⇔ scorer flipped (13, 31, 0) | done 2026-10-26: flips now 0, 21, 23, 24, 31, 43 (13 fixed); B1 ≤ 0.5 mm flags 4 of 6, 1 false; B3 fails 15 of 21 (6 settles with 0 contacts) |
| 6 | Rerun S3a2 → S4b + waterfall on all 19 public episodes on one mesh set | Mac | mean vs Edmond 0.401; 12/21/22 > 0 | done 2026-10-26: 0.358 vs 0.401 (expected 0.47–0.55); 12/21/22 > 0 (0.23 / 0.18 / 0.33); single-object carries lost in S3c on the new meshes (ep40: old mesh 0.731, new 0.427) |
| 6b | S3c on the new meshes (James, 2026-10-27): ep40 first, then 45, 41, 26 | Mac | per-segment errors vs GT; waterfall vs 0.401 | done 2026-10-27: **0.571** (single 0.707, two-object 0.492); ep40 0.938, ep41 0.787, ep45 0.632, ep26 0.871, ep42 0.593, ep31 0.749. Open voxel solid, pass 2 ICP start in a 5 mm / 3° box, backward pass, release trim, static tie-break (rays, then overhang) with turns from the best fit, failed grasps bridged. Edmond still ahead on 21, 22, 23, 24, 42 |
| 7 | GPU window 1 (new L40S): S0.1 on ep41/ep13 (first S5/S6 run); S5/S6 on ep26 (two-object smoke); E1/E2 re-runs on the public 2 21 22 44 and 2 22 24; dry `run.py pack` on public | L40S | S0.1 vs still; overlap < 0.5; spread < 1 cm; pack passes | in progress 2026-10-29: L40S 86.38.238.34 (the 217.177.40.89 disk). ep2's S1 done (public 20 of 20). S0.1 with our own recipe (no team settings, James): **ep41 rollout 0.935**, **ep13 0.899** on the GT reference (still 0.356 / 0.416; ≥ 0.7: the reference is the whole problem; ep13 with njmax 512 overflows). Then F1b/F1c fold into task 11's run. E1/E2 re-runs not built |
| 8 | C1 per-frame refinement inside a grasp (contact set fixed, temporal prior); C2 rotation with the wrist; C3 thin-object weights; C4 turned-over rest | Mac | inverted ≥ keyframe on 41, 18, 43; MPPE | done 2026-10-28 (`6ae4fc91`): public 20 (ep2 = 0) best of three 0.536 → **0.554** on repeatable runs (checkpoint 0.543; 0.571 on 19), inverted 0.511 → 0.530, MP-SR 0.263 → 0.421; 8 up, 1 down (ep45 −0.008), no high scorer fell. Grasp end terms across short S3b gaps; wrist-carried bridges; S3c made repeatable (Open3D on one thread). C3 no gain, C4 no target left. Check holds: inverted ≥ keyframe on 41, 18, 43 |
| 9 | D1 bottom-up supports and settle; D2 per-object candidate mix in the reference; ~~D3 `edmond_place` candidate~~ (dropped, James 2026-10-28: no part of Edmond's work in the pipeline) | Mac | ep0 > 0.424; public selection accuracy | done 2026-10-28: D1 (`d1275092`) leaves every score as it was (ep0's supports corrected); with it a new S3c rule, a rest within the hands' reach: best of three 0.554 → 0.554, inverted 0.530 → 0.531, MPPE 13.2 → 9.3 cm, no episode fell. D2 measured: +0.0003, not built |
| 10 | G2 GT-free signals per object → score fit on public | Mac | — | in progress: the rest rule (worst rest's missed rays) does not separate on 20 episodes: inverted wins on ep2 / ep22 where it flags (0.630 / 0.701 vs still 0.438 / 0.341); agreement with the observation picks inverted everywhere (scratch `obs_agree.py`) |
| 11 | F2 policies: two candidates × 60M on the 20 public episodes | L40S | rollout vs playback on public | in progress (since 2026-10-29 03:16 UTC, L40S, tmux `f2`, `phase_b.sh` with RUN=v4_h1): two candidates × 60M on the 20 public episodes, on Path B's own stages 2–3 (`pathb/stage23.py`: NVIDIA's code as imported, no team runner, James) and own recipe; references from run `v4_h1` (ep0's from `v4_i1`), ~1 GPU-h per episode, ep40 first |
| 12 | F3 selection experiment on public (rollout-vs-reference gate, then signals): selection accuracy vs the GT-best candidate | Mac | selection accuracy | in progress (`efdf2082`): GT-free picks now cost 0.007 against always taking inverted (0.569 vs 0.576 on `v4_g1`): ep2 and ep22 go to still; until a better signal, inverted is the safer default. Rollout gate waits for task 11 |
| 13 | E3 bars: wrist speed → WARN < 0.95; hand-scale bar 0.7 | Mac | reports | done 2026-10-28 (`d1275092`) |
| 14 | Path B's own stages 2–3 (James, 2026-10-29: no team code): NVIDIA's loader, retarget, URDF and support from a mirror of the import `d9264a4e`; our multi-object loader, hand scale, hand repair, still scene, output check; S4a/S4b/S5 off `run_load_retarget.py`; S6 recipe our own | Mac | S5 on ep41 / ep13; S0.1 | done 2026-10-29 (`668f552f`): S5 on ep41 and ep13 passes its checks but ep13's keyframe seating (2.33 cm, NVIDIA's merged disk); still on team code outside the evaluation chain: public scoring via `score_local.py`, S0.1, the S7 packer |
| 15 | S3a/S3c robustness from the full rerun: 12 global registration starts; slip retry from the end pose (guarded by the 3 cm jump bar) | Mac | public 20, per-episode | 12 starts done (`6081b5af`): inverted 0.560 → **0.576** (ep43 0.115 → 0.435); guarded slip retry done (`ced525c3`): ep41 0.654 → 0.831, nothing fell; duplicate-mask rule (ep0's pot track on the lid) 0.080 → 0.353. Public inverted now **0.598** (`v4_i1`) |

Paused (James, 2026-10-28: no runs on evaluation episodes, Mac or cloud GPU, until he lifts it): E1 re-run on 10, 28, 33 and E2 on 20, 32 (task 7); G1 CPU chain on the 20 evaluation episodes and applying G2's fit to them (task 10); F2 on the 20 evaluation episodes, two candidates × 60M and 250M on the chosen reference (task 11); F3 selection on evaluation, G3 pack + preflight, HOLD for James (task 12).

Dropped: scans as meshes; "frame 0 is free"; slip absorption to 3 cm; invented mesh markers; rollout splicing; the ep41/ep13 go/no-go gate; S4c; full HF re-upload of frame archives.

Schedule (set 2026-10-26, before the pause; the submission steps wait for the evaluation runs): Oct 27 tasks 0–7; Oct 28 tasks 8–10; Oct 29–31 tasks 11–12 → safe submission Oct 31 (James's go). Nov 1–3 reruns and resubmission on Nov 3 if James lifts the Nov 1 stop (freeze Nov 4, 5 pm EST).
