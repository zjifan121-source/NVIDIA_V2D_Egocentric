#!/bin/bash
# Import Ken's 3 Nov hand-fix scenes on Wentao fb11 + Wentao sprint v2 re-attached scenes into campaigns/wentao3/scenes
DL=/data/user_data/edmondwa/v2d/wentao_sprint/dl_55846863/experiments
K=$DL/haisenk/hand_fixes_wentao_v1_20261103; V=$DL/wentacc/scored20_lead_sprint_v2_20261103/scenes
OUT=/data/user_data/edmondwa/v2d/campaigns/wentao3/scenes; mkdir -p $OUT
PY=/data/user_data/edmondwa/v2d/venvs/flash_chord/bin/python
for e in 1 3 6 10 14 20 32; do
  $PY -I /home/edmondwa/NVIDIA_V2D_Egocentric/tools/track3/import_scene.py $K/episode_$(printf %06d $e)/scene $OUT/ep${e}_kw --episode $e || echo FAIL ep$e kw
done
for e in 32 33; do
  $PY -I /home/edmondwa/NVIDIA_V2D_Egocentric/tools/track3/import_scene.py $V/episode_$(printf %06d $e)/scene $OUT/ep${e}_v2 --episode $e --raw-bundle $V/episode_$(printf %06d $e)/bundle_rawexcl || echo FAIL ep$e v2
done
