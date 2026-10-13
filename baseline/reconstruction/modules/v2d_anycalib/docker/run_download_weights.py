# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for V2D Track 3 by Edmond Wang, 2026-10: route docker run through v2d.docker.runtime (opt-in Apptainer).
import argparse
import os

from v2d.anycalib.docker._config import IMAGE_NAME, MODULES_DIR
from v2d.docker.runtime import run_container_cmd

DEFAULT_MODEL_ID = "anycalib_gen"


def run_download(output_dir: str, model_id: str = DEFAULT_MODEL_ID, dev: bool = False) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.abspath(output_dir)

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "TORCH_HOME=/data/weights",
        "-v", f"{output_dir}:/data/weights",
    ]
    if dev:
        cmd += ["-v", f"{MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.anycalib.lib.download_weights",
        "--output_dir", "/data/weights",
        "--model_id", model_id,
    ]
    run_container_cmd(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AnyCalib checkpoint")
    parser.add_argument("--output_dir", required=True, help="Output directory for checkpoint")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID,
                        help="AnyCalib variant: anycalib_pinhole|gen|dist|edit")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_download(output_dir=args.output_dir, model_id=args.model_id, dev=args.dev)
