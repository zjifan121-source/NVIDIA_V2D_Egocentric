"""hand7 ep5 (3 Nov): copy of hand5w/wprep.py pointed at Wentao scored20_lead_sprint_v2_20261103 (only W changed). Prepare a bundle on that release for a scene build.
  control : exact copy of Wentao's bundle dir (his objects, camera, hands = our cleaned hands).
  rawhands: same, but every hand_* array is replaced by our RAW pre-cleanup hands (ken_c_scored_attach/bundle_merged),
            re-expressed in Wentao's world: wrist_to_world = W.camera_to_world @ raw wrist_to_camera (camera-frame
            arrays are world-independent). --left-npz takes the hand_left_* arrays from that npz instead (ep3: the
            left-label HaMeR re-run). Objects, camera and everything else stay Wentao's, byte for byte.
usage: wprep.py <ep> <mode control|rawhands> <out bundle dir> [--left-npz path]"""
import shutil, sys, json
from pathlib import Path
import numpy as np
W = Path("/scratch/haisenk-v2d/wentao_v2/experiments/wentacc/scored20_lead_sprint_v2_20261103/scenes")
K = Path("/data/user_data/haisenk/v2d/v2d_train/baseline/reconstruction/data/outputs/load_retarget")
ep, mode, out = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
left_npz = sys.argv[sys.argv.index("--left-npz") + 1] if "--left-npz" in sys.argv else None
src = W / f"episode_{ep:06d}" / "bundle"
if out.exists(): sys.exit(f"exists: {out}")
shutil.copytree(src, out)
info = {"episode": ep, "mode": mode, "wentao_bundle": str(src), "left_npz": left_npz}
if mode == "rawhands":
    w = dict(np.load(out / "result.npz", allow_pickle=True))
    import os
    raw_path = os.environ.get("RAW_NPZ") or str(K / f"episode_{ep:06d}/ken_c_scored_attach/bundle_merged/result.npz")
    raw = dict(np.load(raw_path, allow_pickle=True)); info["raw_npz"] = raw_path
    left = dict(np.load(left_npz, allow_pickle=True)) if left_npz else None
    assert w["camera_to_world_transform"].shape == raw["camera_to_world_transform"].shape
    c2w = w["camera_to_world_transform"].astype(np.float64)
    for side in ("left", "right"):
        srcd = left if (side == "left" and left is not None) else raw
        keys = [k for k in raw if k.startswith(f"hand_{side}_")]
        for k in keys:
            assert k in w and w[k].shape == srcd[k].shape and w[k].dtype == srcd[k].dtype, k
            w[k] = srcd[k]
        w2c = srcd[f"hand_{side}_wrist_to_camera_transform"].astype(np.float64)
        w[f"hand_{side}_wrist_to_world_transform"] = (c2w @ w2c).astype(raw[f"hand_{side}_wrist_to_world_transform"].dtype)
        # sanity: the raw arrays are self-consistent in their own world
        rc = raw["camera_to_world_transform"].astype(np.float64) @ raw[f"hand_{side}_wrist_to_camera_transform"].astype(np.float64)
        info[f"raw_{side}_selfconsistency_max"] = float(np.abs(rc - raw[f"hand_{side}_wrist_to_world_transform"]).max())
        info[f"{side}_keys"] = keys
    np.savez(out / "result.npz", **w)
json.dump(info, open(out / "HANDFIX_PREP.json", "w"), indent=1)
print("PREP_OK", json.dumps(info)[:400])
