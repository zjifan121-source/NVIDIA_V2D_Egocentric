#!/bin/bash
# Queue Ken-recipe-on-Wentao-v2 scenes (ep32_tr, ep33_tr) right after the Ken-fix scenes
O=/data/user_data/edmondwa/v2d/campaigns/wentao3
: > .tr_new.txt
for e in 32 33; do
  echo "{\"$e\": {\"scene\": \"$O/scenes/ep${e}_tr\"}}" > $O/plan_ep${e}_tr.json
  python3 plan_scene_trials.py $O/plan_ep${e}_tr.json $e 600,601 TR general >/dev/null
  sed -i "s/Best-of-both: more seeds on the plan scene/Ken ep32 recipe (raw hands, cup respin, seated) on Wentao v2 objects: ep${e}_tr/" $(cat .plan_scene_trials.txt)
  cat .plan_scene_trials.txt >> .tr_new.txt
done
python3 - <<"P"
import os,re
q=[l.strip() for l in open(".w3_queue.txt") if l.strip()]+[l.strip() for l in open(".tr_new.txt") if l.strip()]
pri=["ep6_kw","ep20_kw","ep32_tr","ep33_tr","ep32_v2","ep1_wa","ep6_wa","ep20_wa","ep25_wa"]
def key(t):
    n=os.path.basename(t); s=int(re.search(r"_s(\d+)_trial",n)[1])
    p=next((i for i,x in enumerate(pri) if n.startswith(x+"_")), len(pri))
    return (s,p,q.index(t))
q=sorted(dict.fromkeys(q),key=key); open(".w3_queue.txt","w").write("\n".join(q)+"\n"); print(len(q))
P
