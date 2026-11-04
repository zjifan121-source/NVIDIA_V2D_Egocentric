# k10 F1 (ep10, 4 Nov): seat the blue cup with a depth-measured visible-top rule, then the UNCHANGED respinf
# attach (bob1's ep10 recipe). exec'd by rlr_h5.py via H5_PATCH, inside its globals.
# Plan: C:/v2d_demo/diag/ep10_fix/ep10_diagnosis_plan.json, plan.fixes[0] (F1). Rule (GT-free):
#  1) rise(f) = p95 height of the visible blue top from FS depth (blue_toprise.json, made by make_toprise.py from
#     the A2 skeptic's sk10a2.json) minus its rest median; 5-frame median.
#  2) dz(f) = min(0, z_support + max(rise, 0) - z_lowest_vertex(f)); applied only where >= 2000 valid depth points
#     and |dz| >= 3 mm (else 0). z_support = median lowest vertex over the rest-hold frames (the leading frames the
#     rest hold made identical to frame 0; 13 on ep10).
#  3) never raise; 5-frame box smoothing; only translation z changes (x, y, rotation unchanged).
#  4) the respinf attach_hands then runs as in bob1, so the holding hand follows the corrected object.
# Acts after levelling (wraps level_scene: levelled world, z up) and before attach_hands.
_P10_BASE = "/home/haisenk/v2d_ops/kenc/hand5/ep10/patch_respinf.py"
exec(compile(open(_P10_BASE).read(), _P10_BASE, "exec"))  # bob1's respinf (defines attach_hands)

_P10_TOPRISE = "/home/haisenk/v2d_ops/kenc/diag/ep10_fix/blue_toprise.json"
_P10_OBJECT = "blue_cup"
_P10_MIN_PTS = 2000
_P10_MIN_DZ = 0.003
_p10_level_orig = level_scene


def _p10_rule(z_low, rise_cm, npts, hold_n):
    """Pure rule: per-frame dz (m, <= 0) and the intermediate series. z_low (m, levelled world), rise_cm and npts
    per frame (NaN / 0 where unmeasured), hold_n = number of rest-hold frames."""
    n = len(z_low)
    rise = np.asarray(rise_cm, float) / 100.0
    rise5 = np.array([np.nanmedian(rise[max(0, f - 2):f + 3]) if np.isfinite(rise[max(0, f - 2):f + 3]).any()
                      else np.nan for f in range(n)])
    z_support = float(np.median(z_low[:hold_n]))
    dz_raw = np.minimum(0.0, z_support + np.maximum(np.nan_to_num(rise5, nan=0.0), 0.0) - z_low)
    ok = (np.asarray(npts) >= _P10_MIN_PTS) & np.isfinite(rise5) & (np.abs(dz_raw) >= _P10_MIN_DZ)
    dz_gated = np.where(ok, dz_raw, 0.0)
    dz = np.convolve(np.pad(dz_gated, 2, mode="edge"), np.ones(5) / 5.0, mode="valid")
    dz = np.minimum(dz, 0.0)  # never raise
    return dz, dict(z_support=z_support, rise5=rise5, dz_raw=dz_raw, gate=ok, dz_gated=dz_gated)


def level_scene(data, bundle, *a, **k):
    info = _p10_level_orig(data, bundle, *a, **k)
    names = [str(x) for x in data["object_names"]] if "object_names" in data else [""]
    i = names.index(_P10_OBJECT)  # slot 0 on ep10 (roster: blue_cup, beige_cup)
    pose_key, scale_key, mesh_file = object_slots(data)[i]
    T = data[pose_key].astype(np.float64)
    n = len(T)
    same = np.all(np.abs(T - T[0]) <= 1e-12, axis=(1, 2))
    hold_n = int(np.argmin(same)) if not same.all() else n
    if hold_n < 3:
        raise SystemExit(f"p10seat: no rest hold found for {_P10_OBJECT} ({hold_n} identical leading frames)")
    v = object_vertices(bundle, data, scale_key, mesh_file)
    z_low = np.array([(v @ T[f, :3, :3].T + T[f, :3, 3])[:, 2].min() for f in range(n)])
    tr = json.loads(open(_P10_TOPRISE).read())
    rise_cm = np.array([np.nan if tr["rise_cm"].get(str(f)) is None else float(tr["rise_cm"][str(f)])
                        for f in range(n)])
    npts = np.array([int(tr["n"].get(str(f), 0)) for f in range(n)])
    dz, d = _p10_rule(z_low, rise_cm, npts, hold_n)
    T[:, 2, 3] += dz
    data[pose_key] = T.astype(data[pose_key].dtype)
    nz = np.flatnonzero(dz != 0.0)
    info.update({
        "p10_rule": "F1 depth visible-top seat (k10, plan.fixes[0])",
        "p10_object": f"{_P10_OBJECT} (slot {i})",
        "p10_toprise": _P10_TOPRISE,
        "p10_rest_top_cm": tr.get("rest_top_cm"),
        "p10_rest_hold_frames": hold_n,
        "p10_z_support_m_levelled": round(d["z_support"], 5),
        "p10_frames_changed": [int(nz[0]), int(nz[-1]), int(len(nz))] if len(nz) else [],
        "p10_gate_frames": int(d["gate"].sum()),
        "p10_dz_cm": [round(float(x) * 100, 3) for x in dz],
        "p10_dz_raw_cm": [round(float(x) * 100, 3) for x in d["dz_raw"]],
        "p10_rise5_cm": [None if not np.isfinite(x) else round(float(x) * 100, 3) for x in d["rise5"]],
        "p10_lowest_before_cm": [round(float(x - d["z_support"]) * 100, 3) for x in z_low],
        "p10_dz_min_cm": round(float(dz.min()) * 100, 3),
    })
    print(f"== p10seat: {_P10_OBJECT} rest hold {hold_n}, z_support {d['z_support']:.4f}, "
          f"dz min {dz.min() * 100:.2f} cm, {len(nz)} frames changed", flush=True)
    return info
