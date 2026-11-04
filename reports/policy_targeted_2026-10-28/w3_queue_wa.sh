#!/bin/bash
# Prepend trials for the Wentao addendum scenes (ep1/25/6/20 _wa) to the w3 queue
O=/data/user_data/edmondwa/v2d/campaigns/wentao3
rm -rf $O/scenes/ep37_wa
: > .wa_new.txt
for e in 1 25 6 20; do
  echo "{\"$e\": {\"scene\": \"$O/scenes/ep${e}_wa\"}}" > $O/plan_ep${e}_wa.json
  python3 plan_scene_trials.py $O/plan_ep${e}_wa.json $e 600,601 WA general >/dev/null
  sed -i "s/Best-of-both: more seeds on the plan scene/Wentao addendum ef19d76f: ep${e}_wa scene/" $(cat .plan_scene_trials.txt)
  cat .plan_scene_trials.txt >> .wa_new.txt
done
cat .wa_new.txt .w3_queue.txt > .w3_queue.tmp && mv .w3_queue.tmp .w3_queue.txt
echo queue $(wc -l < .w3_queue.txt)
