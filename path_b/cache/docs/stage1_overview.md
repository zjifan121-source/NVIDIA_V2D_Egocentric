# Path B Stage 1: metric scale from the head stereo pair

V2D Challenge · Track 3 · Path B · as of 2026-10-19 · run `4c2eac31` · code on GitHub branch `zjifan121` (`428ad8c`); the Stage 1 base is merged into `main_buffer` (PR #13, `974d1d7`). The HTML version of this overview (charts, one frame image) is `path_b/docs/stage1_overview.html`.

Path B Stage 1 keeps the team's reconstruction bundles and fixes their length scale with stereo depth. On the 19 scored public episodes, reconstruction-only AUC rises from 0.314 to **0.418** with the official metric code, above what rescaling by the public scans gives (0.388).

| Headline | Value |
|---|---|
| Mean AUC, after stage 2 repairs → with Path B's scale (19 episodes) | 0.314 → **0.418** |
| Mean MPPE, keypoint error (lower is better) | 11.2 → **9.1 cm** |
| Mean MP-SR, the strictest success rate | 0.05 → **0.11** |
| Episodes better · tied · worse (±0.005 AUC) | 14 · 4 · 1 |

## 1. What Stage 1 does

The team's bundles come from NVIDIA's ego pipeline (the `v0.4.0` code at `8c778f9`): an object track, a mesh, two hands and a camera path. Their size comes from a monocular depth guess (MoGe) fitted on one frame. Stage 1 does not replace any of those parts yet. It measures each episode's true length scale with the head's stereo pair, rescales the bundle by that one factor, and hands it to the team's stages 2–3 unchanged.

Scale is worth fixing first because the official scorer never rescales trajectories. In the local self-test, a perfect trajectory scaled ×1.2 drops AUC from 1.000 to 0.775 and MP-SR from 1.00 to 0.45. The stereo factors put the team's bundles at about 23% too large in a typical episode (median factor 0.81).

## 2. Pipeline components

`P0 ingest → P1 calibrate → P2 stereo depth → P3 scale bridge → P4 export → stages 2–3 → official scoring`. "New" means written for Path B; "reused" means team tools or a baseline module, called read-only.

| Component | What it does | Source | What it showed on the 20 public episodes |
|---|---|---|---|
| **P0 ingest** | Decodes `ego_cam_a` (colour, 2028×1520) and `ego_cam_b/c` (mono stereo, 1280×800); frame counts and resolutions must match the manifest and calibration | new | All 20 pass |
| **P1 calibrate** | Undistorts `ego_cam_a` with the 14-coefficient factory model and rectifies the stereo pair. Tries every reading of the calibration file on the episode's own frames, then fits and removes the remaining row drift | new | Rows off by 0.94–2.12 px with the factory rectification, 0.34–0.43 px after the correction |
| **P2 stereo depth** | FoundationStereo on every rectified pair; up to 3 planes per frame; finds the frame offset and pose between the stereo pair and `ego_cam_a`, then reprojects depth into `ego_cam_a` | new, around the reused `v2d_foundation_stereo` | Flat surfaces fit to 0.41 cm (median RMS). Stereo leads `ego_cam_a` by 1 frame in 18 episodes, 2 in ep21 and ep22. 57–76% of matched features land within 5 px; the factory extrinsics agree with the data to ≤0.35° and ≤13 mm. About 5 min and 16 GB of GPU memory per episode |
| **P3 scale bridge** | Camera height above the table from stereo (the largest horizontal plane ≤35 cm under the object) ÷ the same height in the bundle (its lowest mesh point), median over the frames before a hand reaches an object. A second ruler (distance to the object's surface along its ray) cross-checks it | new | Factors 0.36–1.00, median 0.81. Ep24 stopped here (section 4) |
| **P4 export** | Writes each object's rescaled bundle in the team layout (`tools/track3/BUNDLES.md`), with `run.json`, `numeric_qa.json` and `status.json` beside it | new | 19 episodes exported; 14 object meshes no longer match their scan's size (section 6) |
| **Stages 2–3** | Load, retarget to the Sharpa Wave hands, object URDF, support surface, QA video: `tools/track3/run_load_retarget.py` with its default repairs | reused | Run on the 7 one-object episodes; fingertip error 1.75–2.73 cm mean |
| **Scoring** | The kit's unmodified metric code on public ground truth (`tools/track3/score_local.py`), as a waterfall: raw bundle → stage 2 repairs → Path B scale → scan-oracle scale | reused + new waterfall | Section 5 |
| **Loss reports and pages** | Every stage writes each measurement with its unit, bar and verdict; each episode gets a loss report naming the first failing stage, and `run.py visualize` draws an HTML page per episode | new | Pages for ep13 and ep26 in `~/V2D/bundles_public/` |

## 3. What changed compared with the baseline

| Part | Baseline (the team's bundles) | Path B Stage 1 | Why |
|---|---|---|---|
| **Camera intrinsics** | Learned per episode: focal length 950–1061 px, principal point pinned to the image centre (1014, 760) | Factory calibration: focal 1149.8 px, measured principal point, 14-coefficient distortion; the reading confirmed on each episode's frames | One physical camera; the estimates disagree by 11% across episodes and sit 8–17% below the factory value |
| **Depth** | MoGe, monocular, scale fitted on one reference frame | FoundationStereo on the 75 mm stereo pair, every frame | Metric by triangulation instead of a learned guess (below) |
| **Length scale** | Whatever the monocular depth implies | One factor per episode: stereo camera-to-table height ÷ the bundle's | The scorer works in metres and never rescales trajectories |
| **Stereo pair** | Not used by the ego flow | Rectified; 1–2 px row drift removed; the −1 or −2 frame offset to `ego_cam_a` found per episode | Each would otherwise bias the depth or its alignment to the colour camera |
| **Object track, mesh, hands, camera path** | From the bundle | Unchanged, only rescaled | Isolates the effect of scale; Stage 2 replaces these parts |
| **Stages 2–3, training, submission** | Team runner, FlashCHORD, the kit | Unchanged | Kit-ready output from the first day |

### Why FoundationStereo

- **Metric by geometry.** With the known 75 mm baseline, depth = focal length × baseline ÷ disparity. At the table (about 0.5 m, rectified focal 477 px) the disparity is 72 px, so a 1 px error moves depth by 0.7 cm, 1.4% of the distance. Measured: stereo planes fit flat surfaces to 0.41 cm RMS, median over the 20 episodes.
- **Every frame, not one.** The baseline fits its scale on a single reference frame, so one bad frame sets the whole episode. Stereo gives a fresh metric measurement on every frame, and P3 takes the median.
- **Zero-shot.** NVIDIA's FoundationStereo ([arXiv 2501.09898](https://arxiv.org/abs/2501.09898)) is trained on about 1M synthetic pairs and needs no fine-tuning; we have no stereo ground truth to train with anyway.
- **Already in the repo.** The baseline ships it as `v2d_foundation_stereo` (TensorRT, 576×960 input). Path B calls it read-only; on the L40S the engine compiles once (562 s) and a full episode of depth plus planes takes about 5 minutes.
- **Allowed.** NVIDIA confirmed the stereo stream may be used (reported by James, 2026-10-18).
- **Alternatives.** Depth Anything 3's main model is not metric, ViPE is only near-metric, and MoGe is what the baseline already uses.

## 4. Episodes: what ran, what was skipped

- **All 20 public episodes** ran P0–P2 (calibration and stereo depth).
- **19 are scored.** Ep24 stopped in P3: its bundle's wooden-piece mesh is so oversized that a hand already counts as touching it at frame 0, which leaves no rest frames to measure on.
- **Stages 2–3 ran on the 7 one-object episodes** (39–45). The 12 two-object episodes were skipped there: the runner's `--merge` builds a two-object scene only from the shared hub, not from Path B's rescaled bundles. The waterfall still scores them, using the same stage 2 repairs in-process.
- **The 20 evaluation episodes have not been run.** Stage 1 was validated on the public episodes first, which have ground truth; the evaluation run and the FlashCHORD training on Babel come next.

| Ep | Task · objects | P0–P2 | P3 (factor) | P4 | Stages 2–3 | AUC raw | stage 2 | stereo | scan oracle | Main issue |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Pot with lid · white pot + lid | PASS | WARN (×0.89) | FAIL | SKIP | 0.063 | 0.088 | **0.091** | 0.090 | AUC under 0.1 in every variant; tracking and placement dominate |
| 2 | Pot with lid · white pot + lid | PASS | FAIL (×0.91) | FAIL | SKIP | 0.044 | 0.065 | **0.065** | 0.065 | Stereo support distance jumps 11 cm between frames; pot mesh 0.68× scan |
| 11 | Cup stack · blue cup + beige cup | WARN | WARN (×0.90) | FAIL | SKIP | 0.212 | 0.280 | **0.327** | 0.287 | Blue cup mesh 0.78× scan after the rescale |
| 12 | Cup in dish rack · blue cup + dish rack | PASS | FAIL (×0.76) | PASS | SKIP | 0.248 | 0.342 | **0.767** | 0.777 | Largest gain; factor spread 13% from the bundle side |
| 13 | Cup in dish rack · blue cup + dish rack | PASS | FAIL (×0.83) | FAIL | SKIP | 0.269 | 0.437 | **0.739** | 0.561 | Bundle cup pose wobbles (spread 11%); cup mesh 0.75× scan |
| 18 | Dust brush · sweeper + dust pan | PASS | WARN (×0.76) | FAIL | SKIP | 0.245 | 0.254 | **0.159** | 0.263 | Bundle tracks the wrong object (sweeper run follows the dust pan) |
| 21 | Water can, blue cup · blue cup + pitcher | PASS | WARN (×0.81) | PASS | SKIP | 0.207 | 0.405 | **0.562** | 0.604 | Clean; stereo runs 2 frames ahead of ego_cam_a |
| 22 | Water can, blue cup · blue cup + pitcher | PASS | WARN (×0.88) | PASS | SKIP | 0.375 | 0.421 | **0.646** | 0.643 | Clean; stereo runs 2 frames ahead of ego_cam_a |
| 23 | Planter stand · two wooden pieces | PASS | FAIL (×0.36) | FAIL | SKIP | 0.016 | 0.038 | **0.035** | 0.040 | Blob meshes 3–4× scan; the factor is meaningless |
| 24 | Planter stand · two wooden pieces | PASS | ERROR | not run | not run | – | – | – | – | P3 stopped: a hand already touches the oversized mesh at frame 0, no rest frames |
| 26 | Pan in dish rack · dish rack + pot | PASS | WARN (×0.80) | FAIL | SKIP | 0.241 | 0.293 | **0.333** | 0.306 | Pot mesh 0.67× scan; the 127° flip is untouched |
| 27 | Pan in dish rack · dish rack + pot | PASS | FAIL (×0.75) | PASS | SKIP | 0.447 | 0.447 | **0.566** | 0.549 | Only 3 rest frames with a support plane; the object-ray ruler was used |
| 31 | Brush, dish rack · sweeper + dish rack | PASS | FAIL (×0.53) | FAIL | SKIP | 0.038 | 0.070 | **0.088** | 0.094 | Sweeper mesh 2.25× scan; hands 0.55× after the rescale |
| 39 | Dust pan · dust pan | PASS | FAIL (×0.74) | FAIL | WARN | 0.200 | 0.168 | **0.319** | 0.168 | Mesh 0.50× scan after the rescale; P3 fails only the oracle bar |
| 40 | Dust pan · dust pan | PASS | PASS (×0.75) | PASS | WARN | 0.339 | 0.338 | **0.491** | 0.481 | Clean |
| 41 | Pan · white pot | PASS | WARN (×0.83) | PASS | WARN | 0.284 | 0.287 | **0.470** | 0.432 | Clean; the pot sits on a small lap board, with the floor 0.6 m below |
| 42 | Pan · white pot | PASS | PASS (×0.88) | PASS | WARN | 0.708 | 0.693 | **0.735** | 0.742 | Clean |
| 43 | Pan · white pot | PASS | FAIL (×0.79) | FAIL | FAIL | 0.174 | 0.176 | **0.262** | 0.151 | Stages 2–3 seat the pot 15.5 cm above its table at frame 0 |
| 44 | Pan · white pot | PASS | PASS (×0.87) | PASS | FAIL | 0.777 | 0.747 | **0.865** | 0.855 | Best score; frame-0 seating gap 2.6 cm |
| 45 | Pan · white pot | PASS | FAIL (×1.00) | FAIL | FAIL | 0.252 | 0.416 | **0.418** | 0.258 | Factor 1.00: the bundle was already near metric |

Statuses come from each stage's loss report (PASS, WARN, FAIL, ERROR, SKIP). Many P3 FAILs come from the "scale error vs scan oracle" bar, which measures mesh size rather than world scale (section 6); they do not mean the factor is wrong.

## 5. Results on the official metrics

All numbers are reconstruction-only: the object trajectory a FlashCHORD policy is trained to follow, scored with the kit's metric code against public ground truth. That is the ceiling for a policy trained on it; no policy rollouts are scored yet.

| Mean over 19 episodes | AUC ↑ | SP-SR ↑ | MP-SR ↑ | MPPE cm ↓ | RPE cm ↓ | CD-O cm ↓ |
|---|---|---|---|---|---|---|
| Team bundle as uploaded | 0.270 | 0.368 | 0.053 | 11.90 | 8.89 | 0.739 |
| + stage 2 repairs | 0.314 | 0.474 | 0.053 | 11.15 | 8.85 | 0.739 |
| **+ Path B stereo scale** | **0.418** | 0.474 | **0.105** | **9.06** | **6.73** | 0.740 |
| Scan-oracle scale (comparison) | 0.388 | 0.474 | 0.105 | 10.63 | 6.60 | 0.738 |

**Where the mean AUC comes from:** team bundle 0.270, + 0.044 from stage 2 repairs, + 0.104 from Path B's scale = 0.418. Rescaling by the public scans instead reaches 0.388. The remaining 0.582 is what no scale fixes: tracking, mesh and the placement of two objects.

| Episodes | raw | stage 2 | stereo | scan |
|---|---|---|---|---|
| One object (7) | 0.391 | 0.404 | **0.508** | 0.441 |
| Two objects (12) | 0.200 | 0.262 | **0.365** | 0.357 |

Two-object scenes start lower: the second object is placed from its own camera-frame pose, and any error in that placement costs points that no scale recovers. SP-SR and CD-O do not move with scale: SP-SR is already 1 wherever it is reachable, and the scorer registers each mesh with its own scale for CD-O.

## 6. Error analysis

### Where it works

The largest gains come where the bundle tracks the right object cleanly and scale was its main error: ep12 (0.342 → 0.767), ep13 (0.437 → 0.739), ep22 (0.421 → 0.646), ep41 (0.287 → 0.470), ep21 (0.405 → 0.562), ep40 (0.338 → 0.491), ep39 (0.168 → 0.319) and ep44 (0.747 → 0.865, the best score). In these episodes the stereo table distance holds within 0.1–0.4 cm across the rest frames, and the factor shrinks an oversized world by 12–26%.

### Where it does not help

- **Wrong object (ep18, −0.095).** The bundle's sweeper run follows the dust pan. Rescaling a wrong track cannot help, and here it hurts. Stage 2's masks address identity.
- **Blob meshes (ep23, ep24).** The wooden-piece meshes are 3–4× their scans, so the "lowest mesh point" ruler gives a meaningless factor (0.36) and ep24 has no rest frames at all. AUC stays near 0.04.
- **Pot with lid (ep0, ep2).** Under 0.1 AUC in every variant: tracking and placement dominate, with the lid resting on the pot. Ep2's stereo support distance still jumps by 11 cm between frames.
- **Brush with dish rack (ep31).** 0.070 → 0.088: the sweeper mesh is 2.25× its scan, and hands shrink to 0.55× true size after the rescale.
- **Already near metric (ep45).** The factor is 1.00, so scale changes nothing (0.416 → 0.418).

### Systematic issues the loss reports found

- **Mesh size and world scale disagree.** One factor rescales the whole bundle, but the team's meshes are often sized differently from their own world. After the rescale, 14 object meshes fall outside 0.8–1.25× their scan (ep0 pot 0.75× and lid 1.26×, ep2 pot 0.68×, ep11 and ep13 blue cup 0.78× and 0.75×, ep18 sweeper 0.71×, ep23 boards 3.0× and 4.2×, ep26 pot 0.67×, ep31 sweeper 2.25× and rack 0.77×, ep39 dust pan 0.50×, ep43 pot 0.71×, ep45 pot 0.69×). The score barely notices (the scorer registers meshes with their own scale), but a simulated object that small changes how the robot grasps it. Stage 3 rebuilds meshes.
- **The bundle side of the ruler wobbles.** In ep13 the stereo distance stays at 0.53 m while the bundle's reading drifts from 0.70 to 0.57 and jumps to 0.83 with the head still: the bundle's cup pose is moving. Factor spread stays above 10% in ep12, ep13 and ep45. Stage 2 replaces this ruler with each object's stereo position.
- **The scan "oracle" measures mesh size, not world scale.** On ep39 stereo is 50% below it yet gains 0.15 AUC, while the oracle gains nothing. P3's "scale error vs oracle" bar therefore fails on good episodes and should become a check against the ground-truth trajectory.
- **Plane choice, fixed.** P3 first took the nearest plane under the object, which was often a dish rack's base or flat palms (12 of 18 rest frames in ep13). It now takes the largest one. Stereo support jumps fell from 6 episodes to 1; mean AUC moved 0.420 → 0.418, because the wrong planes had partly cancelled the bundle-side bias. (The HTML version shows ep13's frame 0 with the floor, the table and the plane P3 used before the fix.)
- **Not touched in Stage 1.** Rotation flips (ep26: a 127° jump in one frame) and hand-size disagreement (ep26 hands 0.78 and 1.04 after the rescale) remain. In stages 2–3, fingertip error is 1.75–2.73 cm mean and up to 9.0 cm at worst, inherited from the bundle hands; ep43's pot is seated 15.5 cm above its table at frame 0. The MANO-scale fix (`717f43d`) left reconstruction-only AUC unchanged on all seven episodes.

### Findings about the data, useful beyond Path B

- The stereo streams run one frame ahead of `ego_cam_a` in 18 public episodes and two frames ahead in ep21 and ep22, so the offset must be estimated per episode.
- The factory stereo rectification leaves rows 0.9–2.1 px apart; a linear fit per episode brings them to 0.34–0.43 px.
- The calibration file's intrinsics are meant as listed (not turned upright), and `ego_cam_a`'s extrinsics are relative to `ego_cam_b`; a feature-based pose estimate agrees to within 0.35°.

## 7. What comes next

Stage 2 replaces the bundle-side ruler with each object's metric position from stereo inside its mask: Qwen3-VL boxes each roster object and separates look-alikes, SAM 3 tracks the masks once access to the gated model is granted, and P3 compares stereo and bundle object positions. Still open from Stage 1: a fallback for ep24's missing rest frames, stages 2–3 for two-object scenes, a trajectory-based scale check, and the 20 evaluation episodes.

## 8. Sources

- Code: GitHub `zjifan121-source/NVIDIA_V2D_Egocentric`, branch `zjifan121` at `428ad8c` (plane rule `ddbab90`); `main_buffer` at `974d1d7` (PR #13). Entry point `path_b/run.py`, design in `path_b/README.md`.
- Runs on Hugging Face `rose-tea-cafe/v2d-t3-recon`: `path_b/runs/4c2eac31` (this overview), `0fa525bc` (before the plane fix), `59bc521b` (MANO-fix re-check). Logs: `~/V2D/vm_logs/l40s_2026-10-19/`.
- Episode pages: `~/V2D/bundles_public/ep13_path_b_scale.html`, `ep26_path_b_scale.html`; baseline comparison `ep26_problems.html`.
- Metric definitions: submission kit `tracks/track_3/README.md`; local scorer self-test: README §5.11.
- Plan: [Path B: End-to-End VLM Framework for Track 3](https://claude.ai/code/artifact/0873d10c-acc3-455c-936a-12baf370fabf).
