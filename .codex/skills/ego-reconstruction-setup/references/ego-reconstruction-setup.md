<!-- Modified for V2D Track 3 by zjifan121, 2026-10: paths updated for baseline/ layout -->
# Egocentric Reconstruction Setup Reference

## Important Files

- `baseline/reconstruction/scripts/install_ego_reconstruction_packages.sh`: installs host-side wrapper packages needed by `run_ego_reconstruction.py`.
- `baseline/reconstruction/scripts/build_ego_reconstruction_packages.sh`: builds Docker images for AnyCalib, MoGe, Grounding DINO, SAM2, SAM3D, FoundationPose, HaMeR, WiLoR, GeoCalib, DROID-SLAM, gsplat refinement, ego hand reconstruction, and hand alignment.
- `baseline/reconstruction/scripts/download_ego_reconstruction_weights.sh`: downloads model weights by mode.
- `baseline/reconstruction/modules/v2d_pipelines/run_ego_reconstruction.py`: consolidated public entrypoint for egocentric hand-object reconstruction.
- `baseline/reconstruction/modules/v2d_pipelines/run_ego_wilor.py`: WiLoR/SAM2/HaMeR implementation used by `--hand_tracking hamer`.
- `baseline/reconstruction/modules/v2d_pipelines/run_v2d_ego_e2e.py`: legacy ViPE + DynHaMR implementation used by `--hand_tracking dynhamr`.
- `baseline/reconstruction/modules/v2d_ego_hand_reconstruction/README.md`: manual MANO/BMC setup notes for DynHaMR.

## Setup Command Selection

Use these commands from the repo root:

```bash
bash baseline/reconstruction/scripts/install_ego_reconstruction_packages.sh
bash baseline/reconstruction/scripts/build_ego_reconstruction_packages.sh
bash baseline/reconstruction/scripts/download_ego_reconstruction_weights.sh --mode all
```

Use a narrower download mode when the intended run is known:

```bash
# HaMeR object-prompt path, including SAM3D mesh generation
bash baseline/reconstruction/scripts/download_ego_reconstruction_weights.sh --mode hamer_prompt

# HaMeR path with a caller-provided object mesh
bash baseline/reconstruction/scripts/download_ego_reconstruction_weights.sh --mode hamer_mesh

# DynHaMR prompt path
bash baseline/reconstruction/scripts/download_ego_reconstruction_weights.sh --mode dynhamr_prompt
```

The script always downloads common weights: MoGe, SAM2, FoundationPose, and AnyCalib. Prompt modes add Grounding DINO and SAM3D. HaMeR modes add WiLoR/HaMeR and optional postprocess weights for DROID-SLAM, GeoCalib, and gsplat refinement.

## Manual Assets

For DynHaMR, confirm these manual assets exist before a run:

```text
baseline/reconstruction/data/weights/hand/models/MANO_RIGHT.pkl
baseline/reconstruction/data/weights/hand/BMC/*.npy
```

For HaMeR, the download script populates `data/weights/hamer`, but MANO-style assets may still matter for rendering/export. If a renderer reports missing MANO files, inspect the expected path in the error and compare it to `data/weights/hamer/_DATA/data`.

## Validation Commands

Run from `baseline/reconstruction/` after setup:

```bash
python modules/v2d_pipelines/run_ego_reconstruction.py --help
bash -n scripts/install_ego_reconstruction_packages.sh
bash -n scripts/build_ego_reconstruction_packages.sh
bash -n scripts/download_ego_reconstruction_weights.sh
```

If the import/help check fails, first re-run the install script in the active environment. If Docker image commands fail, re-run the build script after checking Docker daemon and NVIDIA Container Toolkit availability.
