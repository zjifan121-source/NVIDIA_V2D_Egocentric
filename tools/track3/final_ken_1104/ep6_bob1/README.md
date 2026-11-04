# Episode 6 = bob1 (pointer only; no new code from us)

On every pose board, ep6 keeps the team baseline bob1 unchanged. Its scene and policy were built and released by Edmond.

| | |
|---|---|
| Scene | Edmond's episode-6 scene `bm_scored_attach` (`/data/user_data/edmondwa/v2d/recon_data/outputs/load_retarget/episode_000006/bm_scored_attach`). Source: Ken's reconstruction. Bodies: `ep6_bm_scored_attach_white_pot`, `ep6_bm_scored_attach_wooden_spoon`. Our relocated copy on Babel is `/data/user_data/haisenk/v2d/edmond_bob1/relocated/eb1_ep6`. |
| Policy | `ep6_bm_s42` @ 100003840 (dashboard run `ep6_bm_s42_RS/early_100003840_r13`), `policy_100003840.safetensors`, sha256 `837fb7d025aab257e2ff784cafc68600e39b308565e86b40bccbc8642daec321`. Own-ref add_auc 0.9685 [white_pot 0.9828, wooden_spoon 0.9542]. |
| Release | Hugging Face `rose-tea-cafe/v2d-t3-recon`, `experiments/edmond/bob1_submission_20261103/episode_000006/` (`scene/` and `policy/` with `results/`). The README there lists row 6 as source "ken", tracking 0.9685. To train on it, Edmond points to `reports/policy_targeted_2026-10-28/campaign.py` and its TRAINING_GUIDE.md. |
| Kaggle | bob1 = `candidate_bob1`, submission 57934736, AUC 0.75166. The team's best entries keep this ep6 (AUC 0.76526, SP-SR 0.95, MP-SR 0.55, RPE 4.10520, MPPE 3.42999 for base4, and MPPE 3.33289 for N1). bob1's ep6 passes both SP-SR and MP-SR. The bob1 rows carry `code_commit_url` `https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/commit/1689cf6396bee0f6a3bf4cced610f2fe0f71ebf6`. |

## Why we changed nothing (ledger `C:\v2d_demo\tests\ledger.md`, 4 Nov)

- **T7 (`m_ep6_K_s42` @ 90M), rejected.** It swapped in a spoon mesh from an ep15 SAM 3D draw, picked by an ep6 video+depth fit, and carried the track over.
  - AUC-board test (4 Nov 02:26): 0.74704, i.e. ep6 -0.092. When a teammate re-sent the same file to the other boards, SP-SR and MP-SR went from pass to fail.
  - A later audit found the swapped spoon rolled about 165 deg about its long axis, relative to the kit's own shape registration.
- **S2 (spoon in-pot centroid re-fit + pot burst de-smoothing)** failed its pre-registered GT-free gates:
  - held-out IoU_occ 0.43-0.45, below the 0.50 bar;
  - rotation steps of 10.4 deg/frame, above the 6 deg limit.
- **The refine-round designs failed their gates too.** F3a made rotation steps of 16.6 deg/frame against a 15 deg limit. D2's attach moved the grip by 0.6 cm against a 0.5 cm limit.

## One exception, on the CD-O board only

Entry C2 (submission 57984124, CD-O 0.36781) replaced the `white_pot` mesh rows of ep6, and of ep1, ep3 and ep8, with ep5's T8 pot mesh (`rl2_ep5`: Wentao's v2 pot, rim-fixed -1.0 cm). This relies on "same object -> same hidden scan", which was verified on Kaggle with C1.

That code and mesh belong to the ep5 / CD-O-transfer part of this release. ep6's poses and policy are unchanged.
