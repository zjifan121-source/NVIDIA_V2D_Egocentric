# track3/ — shared code for both paths

Baseline: nvidia-isaac/video_to_data release/0.2.0 @ 43a3f5b (tag: baseline-0.2.0)

Rule: shared code only. Nothing specific to Path A or Path B goes here.
Changes reach this folder only through a pull request into main.

- common/   result schema, Track 3 data loader, calibration helpers
- eval/     ground-truth scorer (public episodes) + GT-free proxy checks
- configs/  episode list, object prompts, local train/val split
- scripts/  data download, submission packaging
