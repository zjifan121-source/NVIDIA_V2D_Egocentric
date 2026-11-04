import glob, json
from safetensors import safe_open
f=sorted(glob.glob("/root/runs/ep28_r2com_def_s200/train/state_*.safetensors"))[-1]
with safe_open(f,"numpy") as s: m=s.metadata() or {}
print("meta keys:",list(m.keys())[:20])
def walk(o,p=""):
    if isinstance(o,dict):
        for k,v in o.items(): walk(v,p+"/"+k)
    elif isinstance(o,list):
        if p.lower().find("context")>=0 or p.lower().find("objective")>=0 or p.lower().find("term")>=0: print(p,len(o),str(o[:3])[:90])
    elif any(x in p.lower() for x in ("_dim","context_dim","observation_dim","action_dim")): print(p,o)
for k,v in m.items():
    try: walk(json.loads(v),k)
    except Exception:
        if any(x in k.lower() for x in ("dim","context","objective")): print(k,v[:120])
