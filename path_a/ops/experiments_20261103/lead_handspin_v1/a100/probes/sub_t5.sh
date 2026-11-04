#!/bin/bash
cd /root/v2d/NVIDIA_V2D_Egocentric/assets/data/v2d_submission_kit
/root/venv-kit/bin/python submit/v2d_submit.py "/root/probes/ken_swaps_20261104/merged/bob1_ep1_ep3_ep5_ep8.parquet" --track track_3 --only add_auc --wait -m "Wentao base: bob1 + ep1 (T2) + ep3 (Ken T5) + ep5 (T3) + ep8 (T4), row-level merge, expected AUC 0.76056" --receipt "/root/probes/ken_swaps_20261104/merged/full_t5_auc.receipt.json"
echo AUC_DONE $?
/root/venv-kit/bin/python submit/v2d_submit.py "/root/probes/ken_swaps_20261104/merged/bob1_ep1_ep3_ep5_ep8.parquet" --track track_3 --only spider_sr maniptrans_sr rpe_cm mppe_cm cd_o_cm --wait -m "Wentao base: bob1 + ep1 (T2) + ep3 (Ken T5) + ep5 (T3) + ep8 (T4), row-level merge, expected AUC 0.76056" --receipt "/root/probes/ken_swaps_20261104/merged/full_t5_five.receipt.json"
echo FIVE_DONE $?
