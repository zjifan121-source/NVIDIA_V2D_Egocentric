# SAM 3 object masks (`--object-segmenter sam3`)

Opt-in replacement for the object part of stage 1's masks. The default (`dino_sam2`) is unchanged.

| | Default `dino_sam2` | `sam3` |
|---|---|---|
| Object | Grounding DINO box from the prompt at the reference frame, then SAM2 tracks it | SAM 3 finds the object from the prompt itself and tracks it (no box) |
| Hands | SAM2 from the WiLoR boxes | the same (SAM2 tracks the hands only) |
| Outputs | `masks/1` object, `masks/2,3` hands, `dino_detections.json`, `object_track.json` | the same files and formats, plus `sam3_meta.json` |

Everything after the masks (SAM3D, FoundationPose, hands, refinement, export) is untouched.

## Turn it on

```bash
export V2D_SAM3_PYTHON=/path/to/sam3-venv/bin/python
python tools/track3/run_baseline_reconstruction.py --episode 41 --object-segmenter sam3 --branch my_sam3_runs ...
```

Use a fresh `--branch` (output namespace): the pipeline refuses to resume a folder made with the other segmenter.
The pipeline entrypoints take `--object_segmenter sam3` and `--sam3_weights` (folder with `sam3.pt`, default
`data/weights/sam3`). Not available with `--hand_tracking dynhamr`.

## Setup (once)

SAM 3 runs on the host in its own Python environment, called by the pipeline as a subprocess. Tested with:

```bash
uv venv sam3-venv --python 3.12
uv pip install --python sam3-venv/bin/python torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/facebookresearch/sam3 && git -C sam3 checkout 96914d2425f90a64f45ca977c2b5165418099543
uv pip install --python sam3-venv/bin/python -e sam3 opencv-python-headless einops tqdm pycocotools
```

Some of SAM 3's imports are not declared; install whatever `python -c "from sam3.model.sam3_video_predictor import
Sam3VideoPredictor"` reports missing (we needed scikit-image, scikit-learn, decord, hydra-core, omegaconf, pandas).

Weights: request access to `facebook/sam3` on Hugging Face (gated), log in with your own token, then
`hf download facebook/sam3 --local-dir baseline/reconstruction/data/weights/sam3` (`sam3.pt`, 3.4 GB). About 10 GB of
GPU memory per video.

## What it does (`sam3_object_masks.py`)

- Text prompt at the reference frame; if several objects match, the most probable one (as Grounding DINO's top box).
- If nothing passes SAM 3's default cut-offs, it retries at 0.3, then 0.15. Needed on public episodes 39 (dust pan,
  found at 0.42) and 22 (watering pitcher, found at 0.19); without it those runs fail. At a lowered cut-off it takes
  only a clear winner (twice the next candidate's probability) and otherwise stops with the candidate list.
- Propagates through the whole video in both directions; removes the hand pixels of SAM2's hand tracks.
- A frame where SAM 3 loses the object takes the nearest frame's mask, with a warning: an empty mask would blank
  FoundationPose's depth on that frame. (On 18 public objects this never happened.)
- Writes the reference-frame mask's box as `dino_detections.json`, so later steps read the same file as before;
  `object_track.json` records `segmenter`, the cut-off used and the number of candidates.
- Checks `V2D_SAM3_PYTHON` and `sam3.pt` before any step runs, and refuses to resume a run folder made with the
  other segmenter.

Same limitation as the default: two objects with the same prompt (episode 23, "wooden board") both get the most
probable instance.

## Evidence (public episodes, Babel, Oct 2026)

Mask accuracy against the true outline (released scan projected at the true pose; median over frames of IoU on the
hand-free part):

| Object | SAM 3 | Grounding DINO + SAM2 |
|---|---|---|
| ep41 white pot | 0.912 | 0.786 (0.917 with the hand-written prompt "white saucepan with handle") |
| ep31 mini sweeper | 0.640 | 0.605 |
| ep31 dish rack | 0.821 | 0.821 |

Grounding DINO's "white pot" box cuts off the pot's handle, so the SAM3D mesh has none and its spin is unobservable.
With SAM 3 and the same plain prompt, ep41 scores 0.971, 0.948, 0.971 (3 runs; local official metric, unrefined
bundles), as good as the hand-written prompt. Where Grounding DINO's box is already right, the masks are about the
same and so is the score (ep31: 0.38 vs 0.39 averaged over 7 meshes).

These numbers come from an earlier two-pass version (a full default run, then SAM 3's object masks swapped in).
This in-pipeline version was checked against it: see the commit message for the test runs.
