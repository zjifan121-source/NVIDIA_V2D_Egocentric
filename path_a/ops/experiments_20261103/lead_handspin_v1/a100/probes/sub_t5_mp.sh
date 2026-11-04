#!/bin/bash
cd /root/v2d/NVIDIA_V2D_Egocentric/assets/data/v2d_submission_kit
F=/root/probes/ken_swaps_20261104/merged/bob1_ep1_ep3_ep5_ep8.parquet; R=/root/probes/ken_swaps_20261104/merged
M="Wentao base: bob1 + ep1 (T2) + ep3 (Ken T5) + ep5 (T3) + ep8 (T4), row-level merge, AUC 0.76057"
for i in 1 2 3 4; do
  /root/venv-cpu/bin/python - <<PY > /root/probes/sub_t5_mp.check$i.log 2>&1
import json,urllib.request,os
t=open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
from kaggle.api.kaggle_api_extended import KaggleApi
a=KaggleApi(); a.authenticate()
for s in a.competition_submissions("v2d-challenge-track3-mp-sr")[:3]:
    print(getattr(s,"date",None),getattr(s,"status",None),getattr(s,"public_score",None),getattr(s,"description",None))
PY
  echo CHECK $i; tail -4 /root/probes/sub_t5_mp.check$i.log | cut -c1-170
  if grep -q "ep3 (Ken T5)" /root/probes/sub_t5_mp.check$i.log; then echo ALREADY_ON_BOARD; break; fi
  /root/venv-kit/bin/python submit/v2d_submit.py "$F" --track track_3 --only maniptrans_sr --wait -m "$M" --receipt "$R/full_t5_mp.receipt.json" > /root/probes/sub_t5_mp.try$i.log 2>&1
  rc=$?; echo TRY $i rc=$rc; tail -4 /root/probes/sub_t5_mp.try$i.log | cut -c1-170
  grep -q "COMPLETE" /root/probes/sub_t5_mp.try$i.log && break
  sleep 25
done
echo MP_DONE
