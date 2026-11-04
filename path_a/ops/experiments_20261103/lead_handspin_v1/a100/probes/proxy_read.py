import json, collections
rows=[json.loads(l) for l in open("/root/probes/incoming/proxy_scores.jsonl")]
print(len(rows),"rows; fields:",sorted(rows[0].keys()))
print("kinds:",dict(collections.Counter(r.get("kind") for r in rows)),"mesh_source:",dict(collections.Counter(r.get("mesh_source") for r in rows)))
print()
skip={"run","step","episode","own_ref","own_per_body","mesh_source","termination_m","proxy","kind","proxy_all"}
for r in sorted(rows,key=lambda r:(r.get("episode") or 0,str(r.get("run")),r.get("step") or 0)):
    extra={k:v for k,v in r.items() if k not in skip}
    pb=r.get("own_per_body"); pb=[round(x,2) for x in pb] if isinstance(pb,list) else pb
    print(f"ep{r.get('episode')} {str(r.get('run'))[:46]:46s} {str(r.get('step'))[:9]:>9s} own={r.get('own_ref')} {pb} proxy={r.get('proxy')} all={str(r.get('proxy_all'))[:36]} {json.dumps(extra)[:160]}")
