# ep6_fixpatch.py - H5_PATCH for refine1104 ep6 round 2 (D2 spoon de-smoothing).  exec'd by run_load_retarget.py
# (md5 a88f12fb) into its globals right before cleaned_bundle(); it wraps attach_hands only:
#   1. asserts that the pot (slot 0) and spoon (slot 1) tracks reaching attach_hands equal bob1's bundle_cleaned
#      (EP6_BOB1_CLEANED) within 1e-6, i.e. every earlier cleaning step (rest hold, levelling 1.4 / 15.4 deg) is bob1's;
#   2. writes the D2 spoon poses (EP6_TARGETS: object_1_to_world_transform in the bundle frame, object_1_mask) on the
#      masked frames only (f36-171), so bob1's levelled spoon is kept everywhere else;
#   3. calls the unchanged team attach rule (the right hand re-attaches to the corrected spoon);
#   4. with EP6_NO_LEFT_ATTACH=1 restores every hand_left_* array and drops left holds from the info, so the left
#      hand and the pot stay bob1's (bob1 had no left attach).
import os as _ep6_os

import numpy as _ep6_np

_ep6_orig_attach_hands = attach_hands  # noqa: F821  (run_load_retarget.attach_hands)


def attach_hands(data, bundle, *args, **kwargs):  # noqa: F811
    tg = _ep6_np.load(_ep6_os.environ["EP6_TARGETS"])
    ref = _ep6_np.load(_ep6_os.environ["EP6_BOB1_CLEANED"])
    for key in ("object_to_world_transform", "object_1_to_world_transform"):
        dmax = float(_ep6_np.abs(data[key].astype(_ep6_np.float64) - ref[key].astype(_ep6_np.float64)).max())
        print(f"== ep6 D2 patch: pre-attach {key} vs bob1 bundle_cleaned max |diff| {dmax:.3g}", flush=True)
        assert dmax <= 1e-6, f"{key} differs from bob1 bundle_cleaned by {dmax}"
    m = tg["object_1_mask"].astype(bool)
    T = data["object_1_to_world_transform"].astype(_ep6_np.float64)
    T[m] = tg["object_1_to_world_transform"][m]
    data["object_1_to_world_transform"] = T.astype(data["object_1_to_world_transform"].dtype)
    if "object_1_to_camera_transform" in data:
        ci = _ep6_np.linalg.inv(data["camera_to_world_transform"].astype(_ep6_np.float64))
        data["object_1_to_camera_transform"] = (ci @ data["object_1_to_world_transform"].astype(_ep6_np.float64)).astype(
            data["object_1_to_camera_transform"].dtype)
    fr = _ep6_np.flatnonzero(m)
    print(f"== ep6 D2 patch: spoon frames replaced: {int(m.sum())} [{int(fr.min())}..{int(fr.max())}]", flush=True)
    keep_left = _ep6_os.environ.get("EP6_NO_LEFT_ATTACH") == "1"
    saved = {k: data[k].copy() for k in list(data) if k.startswith("hand_left_")} if keep_left else {}
    info = _ep6_orig_attach_hands(data, bundle, *args, **kwargs)
    if keep_left:
        for k, v in saved.items():
            data[k] = v
        for k in list(info):
            if k.startswith("attached_") and isinstance(info[k], list):
                info[k] = [h for h in info[k] if h[0] != "left"]
                if not info[k]:
                    del info[k]
        info.pop("hand_left_attached_frames", None)
        info["ep6_no_left_attach"] = True
        print(f"== ep6 D2 patch: {len(saved)} hand_left_* arrays restored", flush=True)
    info["ep6_d2_spoon_frames"] = int(m.sum())
    print(f"== ep6 D2 patch: attach info {info}", flush=True)
    return info
