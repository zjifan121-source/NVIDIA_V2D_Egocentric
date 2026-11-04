# k10 F1-b PROPOSAL (not used for any k10_* build): F1 (p10seat.py) + hold-preserving attach.
# Hold ownership is decided on the tracked trajectory BEFORE the seat correction (the object's z minus the applied
# dz), while each grip is applied on the corrected trajectory. Realises F1 rule step 4 ("the holding hand follows
# the corrected object"): without it the unchanged attach sees the seated blue as never lifted (>2 cm) and
# re-picks the beige for the left hand. Automatic, no per-episode numbers. exec'd by rlr_h5.py via H5_PATCH.
_P10A = "/home/haisenk/v2d_ops/kenc/diag/ep10_fix/p10seat.py"
exec(compile(open(_P10A).read(), _P10A, "exec"))  # F1: respinf + depth seat (defines level_scene wrapper)

_p10b_level_a = level_scene
_p10b_lifted_orig = lifted_segments
_P10B_Z = {}  # corrected z series of the seated object -> its dz (m)


def level_scene(data, bundle, *a, **k):
    info = _p10b_level_a(data, bundle, *a, **k)
    names = [str(x) for x in data["object_names"]] if "object_names" in data else [""]
    i = names.index(_P10_OBJECT)
    pose_key = object_slots(data)[i][0]
    dz = np.array(info["p10_dz_cm"], float) / 100.0
    _P10B_Z["z"] = data[pose_key][:, 2, 3].astype(np.float64).copy()
    _P10B_Z["dz"] = dz
    info["p10b_hold_test"] = "hold ownership on the pre-seat trajectory (z - dz) of the seated object"
    return info


def lifted_segments(z, lift):
    z = np.asarray(z, np.float64)
    if "z" in _P10B_Z and z.shape == _P10B_Z["z"].shape and float(np.abs(z - _P10B_Z["z"]).max()) < 0.005:
        z = z - _P10B_Z["dz"]  # the seated object, possibly re-spun (origin moves < 1 mm): test lift before the seat
    return _p10b_lifted_orig(z, lift)
