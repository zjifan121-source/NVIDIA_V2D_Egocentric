import json
for ep in (5,19,30,20,10,1,14,3):
    d = json.load(open(f"ep{ep}_reference.json")); g = d["geometry"]; per = d["metrics"]["add_auc_per_body"]
    objs = list(dict.fromkeys(r["object"] for r in g))
    print(f"== ep{ep} auc={d['metrics']['add_auc']:.3f}")
    for i, o in enumerate(objs):
        rows = [r for r in g if r["object"] == o]
        z0 = [r["object_z"] for r in rows if r["frame"] == 0][0]
        name = o.split("attach_")[-1]
        lift = [r for r in rows if r["object_z"] - z0 > 0.02]
        frames = sorted(set(r["frame"] for r in lift))
        if not frames:
            print(f"  {name:18s} auc={per[i]:.2f} no lift >2cm (max dz {max(r['object_z'] for r in rows)-z0:.3f} m)"); continue
        best = {}
        for r in lift:
            best[(r["frame"])] = min(best.get(r["frame"], 99), min(r["tip_surface_gap_cm"]))
        b = [best[f] for f in frames]
        print(f"  {name:18s} auc={per[i]:.2f} lifted {frames[0]}-{frames[-1]} n={len(frames)} min-tip-gap median {sorted(b)[len(b)//2]:.1f} max {max(b):.1f} cm; >2cm {sum(x>2 for x in b)}/{len(b)}")
