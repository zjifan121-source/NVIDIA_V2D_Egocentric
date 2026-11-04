#!/usr/bin/env python3
"""fc_hulls.py -- run flash_chord's own CoACD path on test URDFs (orchard, CPU only).

For every URDF: newton.ModelBuilder + flash_chord.scene.objects._object_builder + add_urdf exactly as the training's
_spawn_component does, then for every colliding mesh shape: the training's cache key, whether it was already cached,
and flash_chord.assets.mesh.convex_decompose(vertices, faces) with DEFAULT params (threshold 0.05, max_convex_hull -1,
preprocess auto) into the DEFAULT cache dir (FLASH_CHORD_CACHE_DIR unset -> <flash_chord>/assets/.cache/convex), i.e.
the cache the training reads -> warm.  Optional diagnostics: --off_compare re-runs CoACD with preprocess 'off'
(uncached) and reports whether 'auto' gave the identical hulls (=> no manifold preprocessing happened);
--info_log NAME re-runs that piece with CoACD info logging (uncached) so the manifold check is visible.
usage: fc_hulls.py OUT_PREFIX [--off_compare] [--info_log piece] [--no_decompose_uncached] URDF...
writes OUT_PREFIX.npz (names, p{i}_n, p{i}_v{k}, p{i}_f{k}) and OUT_PREFIX.json
"""
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import newton  # noqa: E402
from flash_chord.assets.mesh import _cache_key, convex_decompose, default_cache_dir  # noqa: E402
from flash_chord.scene.objects import _object_builder  # noqa: E402

args = [a for a in sys.argv[1:]]
out = args.pop(0)
off_compare = "--off_compare" in args
no_uncached = "--no_decompose_uncached" in args
info_log = None
if "--info_log" in args:
    info_log = args[args.index("--info_log") + 1]
    args.remove("--info_log"); args.remove(info_log)
urdfs = [a for a in args if not a.startswith("--")]
PARAMS = dict(threshold=0.05, max_convex_hull=-1, preprocess_mode="auto")
cache_dir = default_cache_dir()
res = dict(cache_dir=str(cache_dir), env_FLASH_CHORD_CACHE_DIR=os.environ.get("FLASH_CHORD_CACHE_DIR"),
           newton=getattr(newton, "__version__", "?"), pieces=[], urdfs=urdfs)
try:
    import coacd
    res["coacd"] = getattr(coacd, "__version__", "?")
except Exception as e:  # pragma: no cover
    res["coacd"] = f"import failed: {e}"
arrs = {}
names = []
for u in urdfs:
    files = [m.get("filename") for c in ET.parse(u).getroot().iter("collision") for m in c.iter("mesh")]
    dest = newton.ModelBuilder()
    comp = _object_builder(dest)
    comp.add_urdf(source=u, floating=True, enable_self_collisions=False, collapse_fixed_joints=True, ignore_inertial_definitions=False)
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES); mesh_type = int(newton.GeoType.MESH)
    shapes = [s for s in range(comp.shape_count) if int(comp.shape_type[s]) == mesh_type and (int(comp.shape_flags[s]) & collide)]
    if len(shapes) != len(files):
        raise SystemExit(f"{u}: {len(shapes)} colliding mesh shapes but {len(files)} <collision> meshes")
    res.setdefault("body_mass", {})[os.path.basename(u)] = [float(x) for x in comp.body_mass]
    for s, fn in zip(shapes, files):
        src = comp.shape_source[s]
        V = np.asarray(src.vertices, dtype=np.float64); F = np.asarray(src.indices, dtype=np.int64).reshape(-1, 3)
        key = _cache_key(V, F, PARAMS)
        cf = cache_dir / f"{key}.npz"
        hit = cf.exists()
        rec = dict(urdf=os.path.basename(u), file=os.path.basename(fn), key=key, cache_hit_before=bool(hit), n_vin=int(len(V)), n_fin=int(len(F)),
                   bbox_in=[V.min(0).round(7).tolist(), V.max(0).round(7).tolist()], scale=[float(x) for x in comp.shape_scale[s]])
        if not hit and no_uncached:
            rec["skipped"] = True; res["pieces"].append(rec); continue
        t0 = time.time()
        hulls = convex_decompose(V, F)
        rec["t_s"] = round(time.time() - t0, 2)
        rec["cache_file_after"] = str(cf); rec["cache_exists_after"] = bool(cf.exists())
        rec["n_hulls"] = len(hulls)
        HV = np.concatenate([h.vertices for h in hulls]).astype(np.float64)
        rec["bbox_hulls"] = [HV.min(0).round(7).tolist(), HV.max(0).round(7).tolist()]
        rec["bbox_diff_mm"] = round(float(max(np.abs(HV.min(0) - V.min(0)).max(), np.abs(HV.max(0) - V.max(0)).max())) * 1000, 4)
        if info_log and os.path.basename(fn) == info_log:
            import coacd
            sys.stdout.flush()
            coacd.set_log_level("info")
            parts = coacd.run_coacd(coacd.Mesh(V, F), **PARAMS)
            coacd.set_log_level("error")
            sys.stdout.flush()
            rec["info_run_n_hulls"] = len(parts)
            rec["info_equals_cached"] = bool(len(parts) == len(hulls) and all(np.asarray(p[0]).shape == h.vertices.shape and np.allclose(np.asarray(p[0], float), h.vertices.astype(float), atol=1e-6) for p, h in zip(parts, hulls)))
        if off_compare:
            import coacd
            coacd.set_log_level("error")
            parts = coacd.run_coacd(coacd.Mesh(V, F), threshold=0.05, max_convex_hull=-1, preprocess_mode="off")
            same = len(parts) == len(hulls) and all(np.asarray(p[0]).shape == h.vertices.shape and np.allclose(np.asarray(p[0], float), h.vertices.astype(float), atol=1e-6) for p, h in zip(parts, hulls))
            PV = np.concatenate([np.asarray(p[0], float) for p in parts])
            rec["bbox_off_diff_mm"] = round(float(max(np.abs(PV.min(0) - V.min(0)).max(), np.abs(PV.max(0) - V.max(0)).max())) * 1000, 4)
            rec["n_hulls_off"] = len(parts); rec["auto_equals_off"] = bool(same)
        i = len(names); names.append(os.path.basename(fn))
        arrs[f"p{i}_n"] = np.int64(len(hulls))
        for k, h in enumerate(hulls):
            arrs[f"p{i}_v{k}"] = h.vertices; arrs[f"p{i}_f{k}"] = h.faces
        res["pieces"].append(rec)
        print(json.dumps(rec), flush=True)
arrs["names"] = np.array(names)
np.savez(out + ".npz", **arrs)
json.dump(res, open(out + ".json", "w"), indent=1)
print("FC_HULLS_DONE", out, len(names), flush=True)
