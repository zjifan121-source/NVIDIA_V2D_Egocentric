#!/bin/bash
cd /root/v2d/NVIDIA_V2D_Egocentric/assets/data/v2d_submission_kit
F=/root/probes/ken_swaps_20261104/merged/bob1_ep1_ep3_ep5_ep8.parquet; R=/root/probes/ken_swaps_20261104/merged
M="Wentao base: bob1 + ep1 (T2) + ep3 (Ken T5) + ep5 (T3) + ep8 (T4), row-level merge, AUC 0.76057"
for i in 1 2 3 4 5 6; do
  /root/venv-kit/bin/python submit/v2d_submit.py "$F" --track track_3 --only spider_sr maniptrans_sr rpe_cm mppe_cm cd_o_cm --wait -m "$M" --receipt "$R/full_t5_five.receipt.json" > /root/probes/sub_t5_five.try$i.log 2>&1
  rc=$?; echo TRY $i rc=$rc; tail -8 /root/probes/sub_t5_five.try$i.log | cut -c1-160
  if ! grep -q "Preflight failed" /root/probes/sub_t5_five.try$i.log; then break; fi
  sleep 20
done
echo FIVE_DONE
