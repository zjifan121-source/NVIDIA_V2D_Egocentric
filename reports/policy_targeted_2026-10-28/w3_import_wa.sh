#!/bin/bash
# Import Wentao lead_sprint_v2 addendum (ef19d76f) scenes for ep1/6/20/25/37 into campaigns/wentao3/scenes/ep*_wa
V=/data/user_data/edmondwa/v2d/wentao_sprint/dl_ef19d76f/experiments/wentacc/scored20_lead_sprint_v2_20261103/scenes
O=/data/user_data/edmondwa/v2d/campaigns/wentao3/scenes
PY=/data/user_data/edmondwa/v2d/venvs/flash_chord/bin/python
for e in 1 6 20 25 37; do
  E=$(printf %06d $e); RB=$V/episode_$E/bundle_rawexcl; [ -d $RB ] || RB=$V/episode_$E/bundle
  rm -rf $O/ep${e}_wa
  $PY -I /home/edmondwa/NVIDIA_V2D_Egocentric/tools/track3/import_scene.py $V/episode_$E/scene $O/ep${e}_wa --episode $e --raw-bundle $RB | cut -c1-110
done
$PY contact_union.py $O/ep1_wa $O/ep6_wa $O/ep20_wa $O/ep25_wa $O/ep37_wa
