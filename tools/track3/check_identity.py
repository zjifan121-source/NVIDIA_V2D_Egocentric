"""Did each reconstruction track the right object? Project every bundle's mesh onto frame 0 of its
video and tile the results into contact sheets.

The reconstruction finds its object from a text prompt, and a prompt can lock onto the wrong
object: 'mini sweeper' tracked the dust pan on ep18 and ep19 and the dish rack on ep30. The
numeric checks (NaNs, validity, mesh-to-scan size) pass on such runs, so look at the sheets. Each
tile shows the mesh points in magenta, the camera distance and the mesh size; a mesh tens of cm
across smeared over the table (ep23-25 wooden pieces) is a failed reconstruction.

The frame is the raw ego_cam_a frame while the bundle intrinsics are for the undistorted image, so
points can sit a little off near the image edges.

Usage (repo root):
    uv run --project tools python tools/track3/check_identity.py --out <dir> --episodes 19 30
    (no --episodes: every episode under data/outputs/hub)
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
HUB = REPO / "baseline/reconstruction/data/outputs/hub"
VIDEOS = "assets/data/track_3/{split}/videos/chunk-000/observation.images.ego_cam_a/episode_{ep:06d}.mp4"
TILE_W = 480


def first_frame(ep: int, out: Path) -> Path:
    video = next(REPO / VIDEOS.format(split=s, ep=ep) for s in ("public", "evaluation")
                 if (REPO / VIDEOS.format(split=s, ep=ep)).is_file())
    frame = out / f"frame0_{ep:06d}.png"
    if not frame.is_file():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vframes", "1", str(frame)], check=True)
    return frame


def tile(ep: int, run: Path, frame: Path) -> Image.Image:
    bundle = next(b for b in (run / "result_slam_gravity_aligned", run) if (b / "result.npz").is_file())
    d = np.load(bundle / "result.npz")
    fx, fy, cx, cy = d["camera_intrinsics"]
    mesh = trimesh.load(bundle / "mesh.obj", force="mesh", process=False)
    scale = float(d["object_scale"])
    T = d["object_to_camera_transform"][0].astype(float)
    p = np.asarray(mesh.sample(4000)) * scale @ T[:3, :3].T + T[:3, 3]
    p = p[p[:, 2] > 0.05]
    im = Image.open(frame).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)
    for x, y in zip(fx * p[:, 0] / p[:, 2] + cx, fy * p[:, 1] / p[:, 2] + cy):
        if 0 <= x < W and 0 <= y < H:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 255))
    size = np.round(np.sort(mesh.extents * scale)[::-1] * 100).astype(int)
    draw.rectangle((0, 0, W, 60), fill="black")
    draw.text((6, 6), f"ep{ep} {run.name}  {np.linalg.norm(T[:3, 3]):.2f} m away  mesh {size} cm",
              fill="yellow", font_size=40)
    return im.resize((TILE_W, int(TILE_W * H / W)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="*")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    episodes = args.episodes or sorted(int(p.name.split("_")[1]) for p in HUB.glob("episode_*"))
    tiles = []
    for ep in episodes:
        frame = first_frame(ep, args.out)
        for run in sorted(p for p in (HUB / f"episode_{ep:06d}").iterdir() if p.is_dir()):
            tiles.append(tile(ep, run, frame))
    th = tiles[0].size[1]
    for k in range(0, len(tiles), 9):
        sheet = Image.new("RGB", (3 * TILE_W, 3 * th), "white")
        for i, t in enumerate(tiles[k:k + 9]):
            sheet.paste(t, ((i % 3) * TILE_W, (i // 3) * th))
        sheet.save(args.out / f"identity_{k // 9}.jpg", quality=85)
    print(f"{len(tiles)} runs -> {args.out}/identity_*.jpg")


if __name__ == "__main__":
    main()
