#!/bin/bash
# orchard: add the two ep5 fix scenes (pot centre-of-mass fix; + hands re-attached) to the live queue, seeds 42/43
B=/tmp/haisenk_v2d; O=$HOME/v2d_orchard; P=$B/v2d_train/baseline/robotic_grounding/flash_chord/.venv/bin/python; S=$B/scenes; Q=$O/queue.txt
[ -d $S/w7_com_ep5 ] || { tar xzf $O/ep5fix.tgz -C $S && for d in $S/w7_com_ep5 $S/w7_attcom_ep5; do $P $O/relocate_scene.py $d | head -1; done; }
for a in "o_ep5_com_s42 5 w7_com_ep5 42" "o_ep5_attcom_s42 5 w7_attcom_ep5 42" "o_ep5_com_s43 5 w7_com_ep5 43" "o_ep5_attcom_s43 5 w7_attcom_ep5 43"; do
  set -- $a; grep -q "^$1 " $Q && continue
  pq=$(ls -d $S/$3/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave | head -1); echo "$1 $2 $pq $4 0 1024" >> $Q; done
grep -c . $Q; tail -4 $Q | cut -c1-80; echo ADD_EP5_DONE
