# Path B task queue (2026-10-26; mirrors V4_BUILD_SPEC.md §13)

Rules: push `zjifan121_pathb_v3` after every passing test or smoke test; no merge to `main_buffer` and no Kaggle upload without James. Scans are never read on evaluation. Pooling over evaluation clips is allowed. Every change ships with a GT-free signal (explain-share, B1 margin, B3 frame-0 consistency, P3 scale, segment agreement, settle penetration, grasps failed; primary gate for the submission = rollout error vs own reference).

| # | Task | Where | Check | Status |
|---|---|---|---|---|
| 0 | Upload the Mac-only results to HF `path_b/runs/v4` (ep38 S1, ep2 rerun, 19 public CPU chains, canonicals; results only, no frame archives) with the refreshed token; commit and push `BUILD_LOG.md` | Mac | `hf_outputs.py list` shows 40 episodes | partial: 0.32 GB of results blocked by HF's storage limit (`usedStorage` 100.0 GB); the refreshed read token is not on the Mac yet |
| 1 | A1 pooling gates: drop partials failing the held-out depth check; axis prior = table normal, reject > 20°; pot clips aligned by handle azimuth, reject > 30°; reject clips > 0.3 cm from the pool | Mac | explain-share ≥ 0.8 per clip; public CD-O; dimensions vs the 9 scans | todo |
| 2 | A2 neighbour-free scans; 2 mm voxels for rack, sweeper, boards | Mac | public Chamfer ≤ 0.3 on cup/pot pairs | todo |
| 3 | Download evaluation S1 results (~30 GB); A3 pool over all 40 clips; keep 40-clip vs 20-clip pool per object by explain-share | Mac | wooden_spoon has a mesh | todo |
| 4 | A4 three mesh candidates per object (pooled / own clip / team SAM 3D); A5 completion keeps real asymmetric geometry | Mac | explain-share, B1 margin | todo |
| 5 | B1 registration-ambiguity test with `score_local.py`'s registration; B3 frame-0 bar (< 5 mm / 1° vs the settled static pose) | Mac | public: flagged ⇔ scorer flipped (13, 31, 0) | todo |
| 6 | Rerun S3a2 → S4b + waterfall on all 19 public episodes on one mesh set | Mac | mean vs Edmond 0.401; 12/21/22 > 0 | todo |
| 7 | GPU window 1 (new L40S): S0.1 on ep41/ep13 (first S5/S6 run); S5/S6 on ep26 (two-object smoke); E1/E2 re-runs on 2 10 21 22 28 33 44 and 2 20 22 24 32; dry `run.py pack` on public | L40S | S0.1 vs still; overlap < 0.5; spread < 1 cm; pack passes | todo |
| 8 | C1 per-frame refinement inside a grasp (contact set fixed, temporal prior); C2 rotation with the wrist; C3 thin-object weights; C4 turned-over rest | Mac | inverted ≥ keyframe on 41, 18, 43; MPPE | todo |
| 9 | D1 bottom-up supports and settle; D2 per-object candidate mix in the reference; D3 `edmond_place` candidate | Mac | ep0 > 0.424; public selection accuracy | todo |
| 10 | G1 CPU chain on the 20 evaluation episodes + GT-free signals per object; G2 signals → score fit on public | Mac | — | todo |
| 11 | F2 policies: two candidates × 60M on all 40; 250M on the chosen reference for the 20 evaluation episodes | L40S | rollout vs playback on public | todo |
| 12 | F3 selection (rollout-vs-reference gate, then signals); G3 pack + preflight; HOLD for James | Mac | preflight clean | todo |
| 13 | E3 bars: wrist speed → WARN < 0.95; hand-scale bar 0.7 | Mac | reports | todo |

Dropped: scans as meshes; "frame 0 is free"; slip absorption to 3 cm; invented mesh markers; rollout splicing; the ep41/ep13 go/no-go gate; S4c; full HF re-upload of frame archives.

Schedule: Oct 27 tasks 0–7; Oct 28 tasks 8–10; Oct 29–31 tasks 11–12 → safe submission Oct 31 (James's go). Nov 1–3 reruns and resubmission on Nov 3 if James lifts the Nov 1 stop (freeze Nov 4, 5 pm EST).
