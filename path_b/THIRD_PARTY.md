# Third-party code and models in Path B (V4_BUILD_SPEC.md §7)

Path B's own code is in `path_b/`; it calls the organiser's baseline (`baseline/`, NVIDIA's
`video_to_data` at `8c778f9`) and the team's `tools/track3/` without changing them. Third-party
code that is not a Python package lives outside the repo on the GPU instance
(`/root/v2d/third_party/`, commits pinned in `COMMITS.txt` by `vm/setup.sh`).

| Component | Version | Licence | Used by | Source of the licence |
|---|---|---|---|---|
| SAM 3.1 (facebookresearch/sam3; weights `facebook/sam3.1`, gated) | pinned in `COMMITS.txt` | SAM License | P1b masks | spec §7; the repository's licence file |
| Do as I Do (malik-group/do-as-i-do, `retargeting/`) | pinned in `COMMITS.txt` | MIT | S4c (off by default) | read on GitHub 2026-10-22 (BUILD_LOG task 13) |
| Open3D | 0.19.0 | MIT | S3a scans, S3c ray casting, ICP | package metadata |
| MuJoCo | 3.4.0 | Apache-2.0 | S3d settle (CPU), S4c | package metadata |
| CoACD | 1.0.11 | MIT | S3a convex pieces | package metadata |
| trimesh | 5.x | MIT | meshes throughout | package metadata |
| NumPy | 2.x | BSD-3-Clause (and bundled MIT/0BSD/Zlib parts) | throughout | package metadata |
| SciPy | 1.x | BSD | throughout | package metadata |
| scikit-image | 0.26 | BSD | S3a voxel solid (marching cubes) | package metadata |
| OpenCV (headless) | 5.0 | Apache-2.0 | masks, images | package metadata |
| PyArrow | 26 | Apache-2.0 | task and rollout parquets | package metadata |

Through the baseline, as the organiser ships it (each module's own licence in `baseline/`):
FoundationStereo (P2 depth), DROID-SLAM (P1c camera), WiLoR and HaMeR (P1d hands), FoundationPose
(S3b_track seed; NVIDIA Open Model License, accepted by James for each instance), SAM2 + Grounding
DINO (P1b's fallback, flagged), FlashCHORD (S5 tasks, S6 training and evaluation), the kit's
evaluator and packer. MANO (hand model): the MPI licence of each team member, files never in git.

Not used: EgoFound3R (CC BY-NC-SA 4.0: its ShareAlike and non-commercial terms conflict with the
challenge's release of winning code, spec §7); SPIDER (v2 only, `cache/v2/`); SAM 3D Objects (v3's
S3a, `cache/v3/`).
