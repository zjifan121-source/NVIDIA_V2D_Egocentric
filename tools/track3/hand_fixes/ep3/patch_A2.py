# hand5 ep3 arm A2: use with H5_MERGED_NPZ=armA_result.npz (correct-handedness left hand). The rest hold (reach_frame,
# the only place hands change the object reference) is pinned to the control's frames (ctl_ep3 report:
# white_pot 122, wooden_spoon 21), so the object reference, levelling and world are identical to ctl_ep3 and only the
# hands change (own-ref add_auc stays comparable). Without this the correct left hand reaches the pot at frame 49,
# the pot keeps its raw track from 49 and levelling moves the whole world by ~3.6 cm / 4.4 deg (arm A1).
_H5A2_FRAMES = {"object_to_world_transform": 122, "object_1_to_world_transform": 21}


def reach_frame(data, vertices, cm=10.0, pose_key="object_to_world_transform"):
    return _H5A2_FRAMES[pose_key]
