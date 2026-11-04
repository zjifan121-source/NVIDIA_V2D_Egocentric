#!/bin/bash
cd /root/v2d/NVIDIA_V2D_Egocentric/assets/data/v2d_submission_kit
F=/root/probes/ken_swaps_20261104/merged/bob1_ep1_ep3_ep5_ep8T6.parquet; R=/root/probes/ken_swaps_20261104/merged
M="Base: bob1 + ep1 T2 + ep3 T5 + ep5 T3 + ep8 T6, row-level merge, expected AUC 0.76159"
run(){ for i in 1 2 3 4 5 6; do /root/venv-kit/bin/python submit/v2d_submit.py "$F" --track track_3 --only $1 --wait -m "$M" --receipt "$2" > /root/probes/sub_t6base.try.log 2>&1; rc=$?; tail -7 /root/probes/sub_t6base.try.log | cut -c1-170; echo "rc=$rc try=$i"; grep -q "Preflight failed" /root/probes/sub_t6base.try.log || break; sleep 25; done; /root/venv-cpu/bin/python /root/probes/five_board/retry_file.py "$F" "$M" "$2"; }
run add_auc $R/full_t6_auc.receipt.json
run "spider_sr maniptrans_sr rpe_cm mppe_cm cd_o_cm" $R/full_t6_five.receipt.json
echo BASE_DONE
cd /root/probes/five_board && ./run_list.sh t8 t7 bob6
