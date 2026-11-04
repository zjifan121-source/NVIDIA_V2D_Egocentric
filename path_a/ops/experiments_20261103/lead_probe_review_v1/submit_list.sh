#!/bin/bash
# Reads already-scored team files on the boards they have not been on. No AUC slot is used. Files are byte-identical to what Kaggle holds.
# Usage: submit_list.sh <name>   where name is one of t6 t5 bob5 p1 p2. Check each receipt: a board can come back UPLOAD_FAILED.
F=/root/probes/five_board; KIT=/root/v2d/NVIDIA_V2D_Egocentric/assets/data/v2d_submission_kit; PY=/root/venv-kit/bin/python
FIVE="spider_sr maniptrans_sr rpe_cm mppe_cm cd_o_cm"; FOUR="spider_sr maniptrans_sr rpe_cm mppe_cm"   # FOUR: mesh unchanged, so CD-O is exactly bob1's
case "$1" in
  t6)   f=t6_ep8;    b=$FIVE; m="Other boards: T6 bob1 + ep8 h_ep8_sc_K_s42 (same file as AUC 57974532)";;
  t5)   f=t5_ep3;    b=$FOUR; m="Other boards: T5 bob1 + ep3 h_ep3_sc_K_s42@95M (same file as AUC 57974215)";;
  bob5) f=bob5_ep14; b=$FOUR; m="Other boards: bob5 bob1 + ep14 hold continuation (same file as AUC 57973091)";;
  p1)   f=p1_ep3;    b=$FIVE; m="Other boards: P1 bob1 + ep3 w_ep3_ygrip_s42 (same file as AUC 57971454)";;
  p2)   f=p2_ep20;   b=$FIVE; m="Other boards: P2 bob1 + ep20 w_ep20_fixseat_s43 (same file as AUC 57971553)";;
  t8)   f=t8_ep5;    b=$FIVE; m="Other boards: T8 bob1 + ep5 r_ep5_rl_K_s43@95M (same file as AUC 57976828)";;
  t7)   f=t7_ep6;    b=$FIVE; m="Other boards: T7 bob1 + ep6 m_ep6_K_s42@90M (same file as AUC 57976007)";;
  bob6) f=bob6_ep14; b=$FIVE; m="Other boards: bob6 bob1 + ep14 on Wentao recon (same file as AUC 57976199)";;
  *) echo "unknown name: $1"; exit 2;;
esac
[ -e "$F/$f.receipt.json" ] && { echo "$f already has a receipt; not resubmitting"; exit 3; }
cd "$KIT" && $PY submit/v2d_submit.py "$F/$f.parquet" --track track_3 --only $b --wait -m "$m" --receipt "$F/$f.receipt.json"
