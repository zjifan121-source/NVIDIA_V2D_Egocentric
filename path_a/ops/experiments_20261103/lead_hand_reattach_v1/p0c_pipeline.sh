#!/bin/zsh
# P0b: extend seated + exclusive re-attach scenes to the remaining episodes (raw hands on HF: 3 8 10 14 15 16 19 28 30; control-only: 35 36 38).
set -u
cd /Volumes/E/Code/V2D/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1
P=/Volumes/E/Code/V2D/work/.venv-bundle-cpu/bin/python
RT=/Volumes/E/Code/V2D/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261029/scored_scene_consistency_v1/runtime
python3 - <<'PY'
import sys; sys.path.insert(0,'/Volumes/E/Code/V2D/work/depth-model-preflight-deps')
from huggingface_hub import hf_hub_download
out='/private/tmp/claude-501/-Volumes-E-Code-V2D/9e418383-d795-43bf-92a7-b5721bcd24f8/scratchpad/raw_hands'
for ep in [10,14,15,16,19,28,30]:
    hf_hub_download('rose-tea-cafe/v2d-t3-recon',f'night_2026-10-24/scenes/episode_{ep:06d}/ken_c_scored_attach/bundle_merged/result.npz',repo_type='dataset',local_dir=out,force_download=False); print('raw hands',ep,'ok',flush=True)
PY
for ep in 10 14 15 16 19 28 30; do
  e=$(printf %02d $ep)
  echo "=== ep$e prep"; $P prep_rawhands.py $ep 2>&1 | grep -e PREP_OK -e Error -e assert
  rm -rf bundles/ep${e}_rawclean; cp -r bundles/ep${e}_rawhands bundles/ep${e}_rawclean
  for m in control rawclean; do echo "=== ep$e $m"; $P build_scene.py $ep $m 2>&1 | grep -v -e Warning -e warnings.warn | grep -e 'loader exit' -e Error -e assert; done
  echo "=== ep$e attach_exclusive"; PYTHONPATH=$RT $P attach_exclusive.py $ep 2>&1 | grep -e EXCL -e Error -e assert
  echo "=== ep$e rawexcl"; $P build_scene.py $ep rawexcl 2>&1 | grep -v -e Warning -e warnings.warn | grep -e 'loader exit' -e Error -e assert
  $P hand_contact.py scenes/ep${e}_control scenes/ep${e}_rawclean scenes/ep${e}_rawexcl 2>&1 | grep -v Warning | cut -c1-160
done
for ep in 35 36 38; do
  e=$(printf %02d $ep)
  echo "=== ep$e control-only"; $P prep_rawhands.py $ep 2>&1 | grep -e PREP_OK -e Error | grep control
  $P build_scene.py $ep control 2>&1 | grep -v -e Warning -e warnings.warn | grep -e 'loader exit' -e Error -e assert
  $P hand_contact.py scenes/ep${e}_control 2>&1 | grep -v Warning | cut -c1-160
done
$P compare_hands.py > compare.log 2>&1; echo P0C_DONE
