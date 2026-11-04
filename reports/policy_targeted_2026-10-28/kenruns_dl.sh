#!/bin/bash
set -e
cd ~
DL=/data/user_data/edmondwa/v2d/ken_handfix
K=experiments/haisenk/hand_fixes_v1_20261102
/home/edmondwa/NVIDIA_V2D_Egocentric/tools/.venv/bin/python -I -c "
from huggingface_hub import HfApi, snapshot_download
K='$K'
fs=[f for f in HfApi().list_repo_files('rose-tea-cafe/v2d-t3-recon', repo_type='dataset') if f.startswith(K)]
import collections; print(len(fs), 'files'); print('\n'.join(sorted({'/'.join(f.split('/')[3:6]) for f in fs})[:40]))
pats=[f'{K}/episode_000003/runs/ep3_h5A1_s42/*', f'{K}/episode_000010/runs/ep10_h5respinf_s43/*', f'{K}/episode_000020/runs/ep20_h5objrel_s43/*']
pats+=[f'{K}/episode_0000{e}/scene/score_input/*' for e in ('03','10','20')]
pats+=[f'{K}/episode_0000{e}/runs/*/results/*' for e in ('03','10','20')]
snapshot_download('rose-tea-cafe/v2d-t3-recon', repo_type='dataset', local_dir='$DL', allow_patterns=pats)"
find $DL/$K -path '*results*' -o -path '*score_input*' | grep -E 'parquet|json|obj' | head -40
du -sh $DL
