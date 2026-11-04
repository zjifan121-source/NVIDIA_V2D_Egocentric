# Prints --swap arguments for every current non-bob1 incumbent except the episode being probed. Usage: base_swaps.py <ep>
import json, sys
ep=sys.argv[1]
d=json.load(open('/root/probes/incumbents/INCUMBENTS.json'))['incumbents']
items=d.items() if isinstance(d,dict) else [(str(v.get('episode')),v) for v in d]
out=[]
for k,v in sorted(items,key=lambda x:int(x[0])):
    sub=str(v.get('submission',''))
    if k!=ep and sub.isdigit() and v.get('source')!='bob1' and not str(v.get('source','')).startswith('bob1'):
        out.append(f"--swap {k}=/root/probes/incumbents/raw/{sub}/{sub}.raw")
print(' '.join(out))
