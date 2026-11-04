import json, sys
for ep in map(int, sys.argv[1:]):
    d = json.load(open(f"ep{ep}_reference.json")); g = d["geometry"]
    for o in dict.fromkeys(r["object"] for r in g):
        rows = [r for r in g if r["object"] == o]
        z0 = [r["object_z"] for r in rows if r["frame"] == 0][0]
        print("ep", ep, o.split("attach_")[-1])
        for f in sorted(set(r["frame"] for r in rows)):
            L = [r for r in rows if r["frame"] == f]
            s = " ".join(f"{r['side'][0]}:{min(r['tip_surface_gap_cm']):5.1f}/{r['wrist_surface_gap_cm']:5.1f}" for r in L)
            print(f"  f{f:3d} dz={100*(L[0]['object_z']-z0):5.1f}cm  tip/wrist {s}")
