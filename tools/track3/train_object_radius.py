"""Training-only object-keypoint radius experiment; evaluator files stay unchanged.

V2D_OBJECT_KEYPOINT_RADIUS_M chooses the reward keypoint radius (upstream: 1 m).
All remaining CLI arguments are passed to NVIDIA's train_flash_sac.py. Checkpoints
record the radius and launcher SHA in their metadata. Run in the FlashCHORD env.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import runpy
import sys


def main() -> None:
    radius = float(os.environ.get('V2D_OBJECT_KEYPOINT_RADIUS_M', '1.0'))
    if not math.isfinite(radius) or not 0 < radius <= 1:
        raise ValueError('V2D_OBJECT_KEYPOINT_RADIUS_M must be finite and in (0, 1]')
    from flash_chord.objectives import keypoints, tracking
    from flash_chord.training import checkpoint_metadata

    # Replace the training factory's vectors without editing its on-disk code.
    tracking.KEYPOINT_VECS_NP = keypoints.KEYPOINT_VECS_NP * radius
    original_metadata = checkpoint_metadata.build_checkpoint_metadata
    launcher_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def build_metadata(*args, **kwargs):
        result = original_metadata(*args, **kwargs)
        result['training_object_keypoint_radius_m'] = radius
        result['training_radius_launcher_sha256'] = launcher_sha
        return result

    checkpoint_metadata.build_checkpoint_metadata = build_metadata
    script = Path(__file__).resolve().parents[2] / 'baseline/robotic_grounding/flash_chord/scripts/train_flash_sac.py'
    sys.argv[0] = str(script)
    print(f'Training object-keypoint radius: {radius} m; launcher SHA256 {launcher_sha}', flush=True)
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
