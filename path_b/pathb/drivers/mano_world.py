"""MANO hands of a v3 bundle in the v3 world, by the baseline loader's own re-posing, run in the stages
2-3 venv ($RETARGET_VENV: torch, manotorch, v2d.task_library_loader).

    $RETARGET_VENV/bin/python path_b/pathb/drivers/mano_world.py <bundle dir> <out.npz> <mano dir>

EgoReconDatasetLoader.load_mano_data turns the bundle's camera-frame MANO parameters into world ones
(left hands converted from HaMeR's right-hand parameter space, per-frame hand scale about the
bundle's centroid, camera->world about MANO's rest wrist); its MANO model gives joints (21, OpenPose
order: tips 4, 8, 12, 16, 20) and vertices. With no_ground_align the loader only shifts the world
(object centred in xy, its bottom at z = 0); the shift is undone, so everything is in the bundle's
own (v3) world. Writes per side: joints (N, 21, 3), vertices (N, 778, 3), faces, global_orient
(N, 3, world rotvec of the native-side MANO), finger_pose (N, 45), betas (10,), valid (N,).
"""
import sys
from argparse import Namespace

import numpy as np
import torch

from v2d.task_library_loader.lib.ego_recon_loader import EgoReconDatasetLoader


def main(bundle: str, out: str, mano_dir: str) -> int:
    loader = EgoReconDatasetLoader()
    loader._args = Namespace(mano_model_dir=mano_dir, no_ground_align=True, ground_z_offset=0.0,
                             ground_yaw_offset_deg=0.0, dataset_root=bundle, result_subpath=".",
                             object_name="object", sequence_name="mano_world", device="cpu")
    seq = loader.list_sequences(loader._args)[0]
    data = loader.load_mano_data(seq, torch.device("cpu"))
    _, t_align = loader._scene_align(seq)  # world' = world + t_align (no rotation without ground alignment)
    mano = loader._ensure_mano()
    npz = np.load(seq.source.npz_path)
    result = {"t_align": t_align}
    with torch.no_grad():
        for side in ("right", "left"):
            o = mano.forward(side=side, betas=data[f"{side}_betas"], global_orient=data[f"{side}_global_orient"],
                             transl=data[f"{side}_trans"], finger_pose=data[f"{side}_finger_pose"],
                             scale=data[f"{side}_scale"], scale_center=data[f"{side}_scale_center"])
            result[f"{side}_joints"] = o["joints"].cpu().numpy() - t_align
            result[f"{side}_vertices"] = (o["vertices"].cpu().numpy() - t_align).astype(np.float32)
            result[f"{side}_faces"] = np.asarray(o["faces"].cpu() if torch.is_tensor(o["faces"]) else o["faces"])
            result[f"{side}_global_orient"] = data[f"{side}_global_orient"].cpu().numpy()
            result[f"{side}_finger_pose"] = data[f"{side}_finger_pose"].cpu().numpy()
            result[f"{side}_betas"] = data[f"{side}_betas"].cpu().numpy()
            result[f"{side}_valid"] = npz[f"hand_{side}_is_valid"].astype(bool)
    np.savez(out, **result)
    print(f"[mano_world] {out}: {len(result['right_joints'])} frames", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
