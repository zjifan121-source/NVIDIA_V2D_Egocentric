# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for V2D Track 3 by Edmond Wang, 2026-10: route docker run through v2d.docker.runtime (opt-in Apptainer).
import os

from v2d.droid_slam.docker._config import IMAGE_NAME, MODULES_DIR
from v2d.docker.runtime import run_container_cmd


def run_download(output_dir: str, dev: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)

    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{output_dir}:/data/weights",
    ]
    if dev:
        cmd += ["-v", f"{MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.droid_slam.lib.download_weights",
        "--output_dir", "/data/weights",
    ]
    run_container_cmd(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download DROID-SLAM checkpoint")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_download(output_dir=args.output_dir, dev=args.dev)
