"""Seed a baseline ego-reconstruction run with the headset's calibration and stereo-metric depth.

Two stages, both writing into one seed folder that run_baseline_reconstruction.py --seed links
into each run, so the pipeline skips the steps these files replace:

1. (CPU) calibrated cameras + stereo depth:
     anycalib/undistorted.mp4, anycalib/undistorted_intrinsics.json   (replaces AnyCalib)
     intrinsics/XXXXXX.json, intrinsics_stable.json
     stereo_depth/XXXXXX.png    sparse metric depth from the b/c stereo pair, in the colour camera
   fs_depth/XXXXXX.png        (--foundation-stereo, any frame range, CPU) dense metric depth from
                              NVIDIA's FoundationStereo on the same rectified pair: replaces MoGe
                              outright (shape and scale) instead of only rescaling it
2. (--rescale, after MoGe has written moge_depth/ from the undistorted video on a GPU):
     depth/XXXXXX.png           MoGe depth times a per-frame scale from stereo (replaces MoGe)
     depth_scale.json           the scales: how far MoGe's own scale is off, frame by frame
   MoGe keeps the dense shape the baseline relies on; stereo fixes only its scale, which is what
   the oracle study found wrong (the reconstructed world is ~1.3x too big).

Stereo: OpenCV SGBM with a left-right check (sparse, exact where it matches), and NVIDIA's
FoundationStereo ONNX (small, 576x960; dense). With the geometry below they agree to 1.3-1.9% median
on the 0.2-1.5 m pixels SGBM matches (89-92% within 10%). Scaled MoGe still errs by 14-18% there:
it compresses depth, pushing near surfaces too far, which one scale per frame cannot fix.

Geometry (checked photometrically: stereo points projected into the colour camera correlate 0.91
with its image this way, 0.45 otherwise; rectification leaves <2 px vertical error): the release's
calibration applies to the frames as stored, with no 180-degree turn. The stereo left camera is
ego_cam_c and right ego_cam_b; each camera's extrinsics_to_stereo_left maps points from that camera
into ego_cam_b's frame, so c->b is ego_cam_c's matrix and b->a is the inverse of ego_cam_a's. The
remaining 1-2 px vertical misalignment after rectification is removed per frame with the median
ORB-match row offset before SGBM.

Usage (stereo venv):
    python tools/track3/stereo_seed.py --episode 41 --out <seed dir> [--frames 0 50] [--preview <png>]
    python tools/track3/stereo_seed.py --episode 41 --out <seed dir> --rescale
    python tools/track3/stereo_seed.py --episode 41 --out <seed dir> --foundation-stereo <onnx> --frames 0 40
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("episodes.json")
CALIBRATION = REPO / "assets/data/track_3/public/meta/camera_calibration.json"
NEAR, FAR = 0.2, 1.5  # metres: the table and objects, where stereo is reliable and MoGe matters
MIN_PIXELS = 2000     # stereo pixels needed to trust one frame's scale
SMOOTH = 11           # frames in the running median over per-frame scales


def camera(cal: dict, name: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    c = cal["cameras"][name]
    return np.array(c["intrinsic_matrix"], float), np.array(c["distortion_coefficients"], float), tuple(c["resolution"])


def extrinsics(cal: dict, name: str) -> np.ndarray:
    """4x4 map from `name`'s coordinates into ego_cam_b's."""
    return np.array(cal["cameras"][name]["extrinsics_to_stereo_left"], float)


class Rig:
    def __init__(self, cal: dict):
        self.Ka, Da, self.size_a = camera(cal, "ego_cam_a")
        Kc, Dc, size_s = camera(cal, "ego_cam_c")
        Kb, Db, _ = camera(cal, "ego_cam_b")
        self.size_s = size_s
        # Colour camera: undistort onto a pinhole with the same matrix.
        self.map_a = cv2.initUndistortRectifyMap(self.Ka, Da, None, self.Ka, self.size_a, cv2.CV_32FC1)
        c_to_b = extrinsics(cal, "ego_cam_c")
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(Kc, Dc, Kb, Db, size_s, c_to_b[:3, :3], c_to_b[:3, 3:4],
                                                    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        self.map_l = cv2.initUndistortRectifyMap(Kc, Dc, R1, P1, size_s, cv2.CV_32FC1)
        self.map_r = cv2.initUndistortRectifyMap(Kb, Db, R2, P2, size_s, cv2.CV_32FC1)
        self.f, self.cx, self.cy = P1[0, 0], P1[0, 2], P1[1, 2]
        self.baseline = float(abs(P2[0, 3] / P2[0, 0]))
        # rectified-left -> c (R1 rotates c into the rectified frame) -> b -> colour camera
        c_to_a = np.linalg.inv(extrinsics(cal, "ego_cam_a")) @ c_to_b
        self.R_rect_a = c_to_a[:3, :3] @ R1.T
        self.t_a = c_to_a[:3, 3]
        self.orb = cv2.ORB_create(3000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def intrinsics(self) -> dict:
        W, H = self.size_a
        K = self.Ka
        return {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                "width": int(W), "height": int(H)}

    def undistort_a(self, bgr: np.ndarray) -> np.ndarray:
        return cv2.remap(bgr, *self.map_a, cv2.INTER_LINEAR)

    def rectify(self, c: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Rectified (left = c, right = b), right shifted by the median ORB row offset; returns the offset."""
        left, right = cv2.remap(c, *self.map_l, cv2.INTER_LINEAR), cv2.remap(b, *self.map_r, cv2.INTER_LINEAR)
        k1, d1 = self.orb.detectAndCompute(left, None)
        k2, d2 = self.orb.detectAndCompute(right, None)
        dy = 0.0
        if d1 is not None and d2 is not None:
            ms = self.matcher.match(d1, d2)
            if len(ms) >= 30:
                dy = float(np.median([k1[m.queryIdx].pt[1] - k2[m.trainIdx].pt[1] for m in ms]))
                right = cv2.warpAffine(right, np.float32([[1, 0, 0], [0, 1, dy]]), self.size_s)
        return left, right, dy

    def depth_in_a(self, disparity: np.ndarray, max_depth: float = 2.5, fill: bool = False) -> np.ndarray:
        """Rectified-left disparity -> metric depth in the undistorted colour camera,
        z-buffered at half resolution (the stereo cameras' angular resolution), then upsampled."""
        H, W = disparity.shape
        v, u = np.mgrid[0:H, 0:W]
        ok = disparity > 0.5
        z = np.where(ok, self.f * self.baseline / np.maximum(disparity, 1e-6), 0)
        ok &= (z > 0.1) & (z < max_depth)
        pts = np.stack([(u[ok] - self.cx) * z[ok] / self.f, (v[ok] - self.cy) * z[ok] / self.f, z[ok]], 1)
        pa = pts @ self.R_rect_a.T + self.t_a
        pa = pa[pa[:, 2] > 0.05]
        Wa, Ha = self.size_a
        w2, h2 = Wa // 2, Ha // 2
        x = np.round((pa[:, 0] / pa[:, 2] * self.Ka[0, 0] + self.Ka[0, 2]) / 2).astype(int)
        y = np.round((pa[:, 1] / pa[:, 2] * self.Ka[1, 1] + self.Ka[1, 2]) / 2).astype(int)
        inside = (x >= 0) & (x < w2) & (y >= 0) & (y < h2)
        half = np.full(w2 * h2, np.inf)
        np.minimum.at(half, y[inside] * w2 + x[inside], pa[inside, 2])
        half = half.reshape(h2, w2).astype(np.float32)
        half[~np.isfinite(half)] = 0
        if fill:  # dense input: close one-pixel splat gaps only (never overwrite a measured pixel)
            closed = cv2.morphologyEx(half, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            half = np.where(half == 0, closed, half)
        return cv2.resize(half, (Wa, Ha), interpolation=cv2.INTER_NEAREST)


class FoundationStereo:
    """NVIDIA's FoundationStereo ONNX on CPU (~20-30 s/frame on 16 cores) or GPU (STEREO_DEVICE=cuda); same pre/post-processing
    as baseline/reconstruction/modules/v2d_foundation_stereo/lib/trt_inference.py."""
    H, W = 576, 960
    MEAN = np.array([123.675, 116.28, 103.53], np.float32)
    STD = np.array([58.395, 57.12, 57.375], np.float32)

    def __init__(self, model: str):
        import os
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = len(os.sched_getaffinity(0))  # Slurm cgroups reject ORT's default pinning
        opts.enable_cpu_mem_arena = False  # the arena kept growing past 32 GB between frames
        gpu = os.environ.get("STEREO_DEVICE") == "cuda"  # GPU: <1 s/frame; needs the pip CUDA libs on LD_LIBRARY_PATH
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model, opts, providers=providers)
        if gpu and self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise SystemExit("FoundationStereo fell back to CPU (~30 s/frame); check the CUDA libraries")
        self.inputs = [i.name for i in self.session.get_inputs()]

    def prep(self, gray: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = gray.shape
        s = min(self.W / w, self.H / h)
        nw, nh = int(w * s), int(h * s)
        img = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        img = cv2.copyMakeBorder(img, 0, self.H - nh, 0, self.W - nw, cv2.BORDER_REPLICATE)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32)
        return ((rgb - self.MEAN) / self.STD).transpose(2, 0, 1)[None].copy(), s, nw, nh

    def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        L, s, nw, nh = self.prep(left)
        R, _, _, _ = self.prep(right)
        d = np.squeeze(self.session.run(None, dict(zip(self.inputs, (L, R))))[0])[:nh, :nw] / s
        return cv2.resize(d, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)


class SGBM:
    def __init__(self):
        self.m = cv2.StereoSGBM_create(minDisparity=0, numDisparities=256, blockSize=5, P1=8 * 25, P2=32 * 25,
                                       disp12MaxDiff=1, uniquenessRatio=15, speckleWindowSize=100,
                                       speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return self.m.compute(left, right).astype(np.float32) / 16.0


def encode_depth(depth_m: np.ndarray) -> np.ndarray:
    """The pipeline's DepthImage PNG encoding: 65535/(d+1); 0 m (missing) -> 65535 -> decodes to 0."""
    return np.clip(65535.0 / (depth_m + 1.0), 0, 65535).astype(np.uint16)


def decode_depth(path: Path) -> np.ndarray:
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float64)
    return np.where(a >= 65535, 0.0, 65535.0 / np.maximum(a, 1.0) - 1.0)


def frames(path: Path, gray: bool):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if gray else f
    cap.release()


def seed(args: argparse.Namespace) -> int:
    ep = {e["episode"]: e for e in json.loads(MANIFEST.read_text())["episodes"]}[args.episode]
    video_a = REPO / ep["video"]
    video_b, video_c = (Path(str(video_a).replace("ego_cam_a", f"ego_cam_{c}")) for c in "bc")
    rig, stereo, out = Rig(json.loads(CALIBRATION.read_text())), SGBM(), args.out
    for d in ("anycalib", "stereo_depth", "intrinsics"):
        (out / d).mkdir(parents=True, exist_ok=True)
    intr = rig.intrinsics()
    fps = cv2.VideoCapture(str(video_a)).get(cv2.CAP_PROP_FPS) or 20.0
    W, H = rig.size_a
    start, stop = args.frames or (0, 10 ** 9)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W}x{H}", "-r", str(fps), "-i", "pipe:", "-c:v", "libx264", "-crf", "17",
                           "-pix_fmt", "yuv420p", str(out / "anycalib/undistorted.part.mp4")], stdin=subprocess.PIPE)
    n, coverage, shifts = 0, [], []
    for i, (fa, fb, fc) in enumerate(zip(frames(video_a, False), frames(video_b, True), frames(video_c, True))):
        if i < start:
            continue
        if i >= stop:
            break
        ua = rig.undistort_a(fa)
        ff.stdin.write(ua.tobytes())
        left, right, dy = rig.rectify(fc, fb)
        shifts.append(dy)
        depth = rig.depth_in_a(stereo(left, right))
        cv2.imwrite(str(out / "stereo_depth" / f"{i:06d}.png"), encode_depth(depth))
        (out / "intrinsics" / f"{i:06d}.json").write_text(json.dumps(intr, indent=4))
        coverage.append(float(((depth > NEAR) & (depth < FAR)).mean()))
        if args.preview and n == 0:
            col = cv2.applyColorMap(np.clip(depth / FAR * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            col[depth == 0] = 0
            args.preview.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.preview), cv2.addWeighted(ua, 0.5, col, 0.5, 0)[::2, ::2])
        n += 1
        if n % 50 == 0:
            print(f"  {n} frames, near-range stereo coverage {np.mean(coverage):.2f}", flush=True)
    ff.stdin.close()
    ff.wait()
    (out / "anycalib/undistorted.part.mp4").rename(out / "anycalib/undistorted.mp4")
    (out / "anycalib/undistorted_intrinsics.json").write_text(json.dumps(intr, indent=4))
    (out / "intrinsics_stable.json").write_text(json.dumps(intr, indent=4))
    (out / "stereo_seed.json").write_text(json.dumps({
        "episode": args.episode, "stereo": "sgbm", "frames": n, "near_coverage": float(np.mean(coverage)),
        "rectified_focal_px": float(rig.f), "baseline_m": rig.baseline, "row_shift_px_median": float(np.median(shifts)),
        "intrinsics": intr}, indent=1))
    print(f"seeded {n} frames into {out} (near-range stereo coverage {np.mean(coverage):.2f})")
    return 0


def foundation_stereo(args: argparse.Namespace) -> int:
    """fs_depth/ for frames [START, STOP): FoundationStereo on the rectified pair, in the colour camera."""
    ep = {e["episode"]: e for e in json.loads(MANIFEST.read_text())["episodes"]}[args.episode]
    video_a = REPO / ep["video"]
    video_b, video_c = (Path(str(video_a).replace("ego_cam_a", f"ego_cam_{c}")) for c in "bc")
    rig, fs, out = Rig(json.loads(CALIBRATION.read_text())), FoundationStereo(args.foundation_stereo), args.out / "fs_depth"
    out.mkdir(parents=True, exist_ok=True)
    start, stop = args.frames or (0, 10 ** 9)
    done = 0
    for i, (fb, fc) in enumerate(zip(frames(video_b, True), frames(video_c, True))):
        if i < start or (out / f"{i:06d}.png").is_file():
            continue
        if i >= stop:
            break
        left, right, _ = rig.rectify(fc, fb)
        cv2.imwrite(str(out / f"{i:06d}.png"), encode_depth(rig.depth_in_a(fs(left, right), fill=True)))
        done += 1
        if done % 10 == 0:
            print(f"  frame {i}: {done} done", flush=True)
    print(f"FoundationStereo depth for frames {start}-{min(stop, i + 1) - 1} in {out} ({done} new)")
    return 0


def rescale(args: argparse.Namespace) -> int:
    """depth/ = MoGe depth x per-frame (robust, then temporally smoothed) stereo/MoGe scale."""
    out = args.out
    names = sorted(p.name for p in (out / "moge_depth").glob("*.png"))
    if not names:
        raise SystemExit(f"no MoGe depth in {out / 'moge_depth'}")
    raw, counts = [], []
    for name in names:
        m, s = decode_depth(out / "moge_depth" / name), decode_depth(out / "stereo_depth" / name)
        ok = (s > NEAR) & (s < FAR) & (m > 0)
        counts.append(int(ok.sum()))
        raw.append(float(np.median(s[ok] / m[ok])) if ok.sum() >= MIN_PIXELS else np.nan)
    raw = np.array(raw)
    good = np.isfinite(raw)
    if good.sum() < 0.2 * len(raw):
        raise SystemExit(f"stereo scale available on only {good.sum()}/{len(raw)} frames")
    filled = np.interp(np.arange(len(raw)), np.flatnonzero(good), raw[good])
    pad = SMOOTH // 2
    padded = np.pad(filled, pad, mode="edge")
    scale = np.array([np.median(padded[i:i + SMOOTH]) for i in range(len(raw))])
    (out / "depth").mkdir(exist_ok=True)
    for name, k in zip(names, scale):
        cv2.imwrite(str(out / "depth" / name), encode_depth(decode_depth(out / "moge_depth" / name) * k))
    summary = {"median_scale": float(np.median(scale)), "p10_p90": np.percentile(scale, [10, 90]).tolist(),
               "frames_with_stereo": int(good.sum()), "frames": len(raw),
               "per_frame_raw": [None if np.isnan(r) else round(r, 4) for r in raw],
               "per_frame_smoothed": scale.round(4).tolist(), "stereo_pixels": counts}
    (out / "depth_scale.json").write_text(json.dumps(summary))
    print(f"rescaled {len(names)} MoGe frames: median stereo/MoGe scale {summary['median_scale']:.3f} "
          f"(p10-p90 {summary['p10_p90'][0]:.3f}-{summary['p10_p90'][1]:.3f}), stereo on {good.sum()}/{len(raw)} frames")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--out", type=Path, required=True, help="seed folder")
    p.add_argument("--rescale", action="store_true", help="stage 2: scale moge_depth/ into depth/")
    p.add_argument("--foundation-stereo", metavar="ONNX", help="write fs_depth/ for --frames with this model")
    p.add_argument("--frames", type=int, nargs=2, metavar=("START", "STOP"), help="only these frames (checks)")
    p.add_argument("--preview", type=Path, help="write a depth-over-colour check image of the first frame")
    args = p.parse_args()
    if args.foundation_stereo:
        return foundation_stereo(args)
    if args.rescale:
        return 0 if (args.out / "depth_scale.json").is_file() else rescale(args)
    if (args.out / "stereo_seed.json").is_file() and not args.frames:
        print(f"{args.out} is already seeded")
        return 0
    return seed(args)


if __name__ == "__main__":
    raise SystemExit(main())
