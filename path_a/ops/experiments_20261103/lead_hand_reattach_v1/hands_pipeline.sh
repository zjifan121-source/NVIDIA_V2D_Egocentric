#!/bin/zsh
# P0 (2026-11-03 evening): seated + exclusive re-attach scenes for the episodes Edmond flagged (1, 6, 20, 25, 37); objects = fb11 = lead_sprint_v1.
set -u
cd /Volumes/E/Code/V2D/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1
P=/Volumes/E/Code/V2D/work/.venv-bundle-cpu/bin/python
RT=/Volumes/E/Code/V2D/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261029/scored_scene_consistency_v1/runtime
python3 - "$@" <<'PY'
import sys; sys.path.insert(0,'/Volumes/E/Code/V2D/work/depth-model-preflight-deps')
from huggingface_hub import hf_hub_download
out='/private/tmp/claude-501/-Volumes-E-Code-V2D/9e418383-d795-43bf-92a7-b5721bcd24f8/scratchpad/raw_hands'
for ep in [int(a) for a in sys.argv[1:]] or [1,6,20,25,37]:
    p=hf_hub_download('rose-tea-cafe/v2d-t3-recon',f'night_2026-10-24/scenes/episode_{ep:06d}/ken_c_scored_attach/bundle_merged/result.npz',repo_type='dataset',local_dir=out,force_download=False); print('raw hands',ep,'ok')
PY
for ep in "$@"; do
  e=$(printf %02d $ep)
  echo "=== ep$e prep"; $P prep_rawhands.py $ep 2>&1 | grep -e PREP_OK -e Error -e assert
  rm -rf bundles/ep${e}_rawclean; cp -r bundles/ep${e}_rawhands bundles/ep${e}_rawclean
  for m in control rawclean; do echo "=== ep$e $m"; $P build_scene.py $ep $m 2>&1 | grep -v -e Warning -e warnings.warn | grep -e 'loader exit' -e Error -e assert; done
  echo "=== ep$e attach_exclusive"; PYTHONPATH=$RT $P attach_exclusive.py $ep 2>&1 | grep -e EXCL -e Error -e assert
  echo "=== ep$e rawexcl"; $P build_scene.py $ep rawexcl 2>&1 | grep -v -e Warning -e warnings.warn | grep -e 'loader exit' -e Error -e assert
  $P hand_contact.py scenes/ep${e}_control scenes/ep${e}_rawclean scenes/ep${e}_rawexcl 2>&1 | grep -v Warning | cut -c1-160
done
$P compare_hands.py > compare.log 2>&1; echo PIPELINE_DONE
