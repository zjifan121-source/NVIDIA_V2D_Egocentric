#!/bin/bash
# hand5w round 2 prebuilds on Wentao's bundle: as-delivered controls and fix-free raw-hand re-cleans
HW=$HOME/v2d_ops/kenc/hand5w; L=/scratch/haisenk-v2d/hand5w/logs; mkdir -p $L
for e in 1 5 6 14 15 16 32; do
  bash $HW/wbuild.sh w_ctl_ep$e $e control > $L/w_ctl_ep$e.out 2>&1 &
  bash $HW/wbuild.sh w_raw_ep$e $e rawhands > $L/w_raw_ep$e.out 2>&1 &
done
wait; for e in 1 5 6 14 15 16 32; do tail -n 1 $L/w_ctl_ep$e.out $L/w_raw_ep$e.out; done; echo BUILDR2_DONE
