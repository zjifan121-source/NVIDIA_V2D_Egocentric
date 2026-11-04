#!/bin/bash
# refine1104 ep8 step 1 (inside job 10774619, CPU only): copy the installed adopted scene sc_ep8 OUT to my own dir (read-only on the source),
# list md5s, and list ep8 runs (read-only) for the judge plan.
set -u
O=/data/user_data/haisenk/v2d/refine1104/ep8; S=/tmp/haisenk_v2d/scenes/sc_ep8
mkdir -p $O
ls -la $S | head; 
if [ -e $O/src_sc_ep8 ]; then echo "src copy exists"; else cp -a $S $O/src_sc_ep8 && echo "copied"; fi
(cd $O/src_sc_ep8 && find . -type f | sort | xargs md5sum) > $O/src_sc_ep8.md5
(cd $S && find . -type f | sort | xargs md5sum) > $O/installed_sc_ep8.md5
cmp $O/src_sc_ep8.md5 $O/installed_sc_ep8.md5 && echo "COPY_MD5_OK"
grep -E "parquet|usda|urdf|beige_cup.obj|result.npz" $O/src_sc_ep8.md5
echo "== ep8 runs on s5-16"
ls /tmp/haisenk_v2d/runs | grep -i ep8 | head -40
for r in /tmp/haisenk_v2d/runs/*ep8_sc*; do [ -d $r ] || continue; echo "-- $r"; cat $r/best.txt 2>/dev/null; ls $r | head -30 | tr '\n' ' '; echo; done
echo "== queue tail"; tail -n 15 /tmp/haisenk_v2d/kq_babel.txt 2>/dev/null
echo B1_DONE
