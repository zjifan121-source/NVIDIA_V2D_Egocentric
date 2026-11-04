#!/bin/bash
# kqq.sh: queue lines with recipe, prio and state (taken / pending), plus GPU use
cd /tmp/haisenk_v2d
while read -r n e p s r pr; do
  [ -z "$n" ] || [ "${n:0:1}" = "#" ] && continue
  st=pending; grep -qxF "$n" kq.txt.taken && st=taken
  [ -f runs/$n/best.txt ] && st="done $(cut -c1-40 runs/$n/best.txt)"
  echo "$n ep$e $r prio=${pr:-50} $st"
done < kq.txt
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ';'; echo
