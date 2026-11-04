#!/bin/bash
# hand5w setup: scratch repo tree = shipped main_buffer run_load_retarget.py (+2 hooks) + hand_fixes, baseline/assets -> v2d_train
T=/data/user_data/haisenk/v2d/v2d_train; SG=$HOME/v2d_ops/kenc/hand5/ship_git; R=/scratch/haisenk-v2d/hand5w/repo
mkdir -p $R/tools/track3 /scratch/haisenk-v2d/hand5w/{bundles,scenes}
cp $SG/tools/track3/run_load_retarget.py $R/tools/track3/; cp -r $SG/tools/track3/hand_fixes $R/tools/track3/
cp $T/tools/track3/episodes.json $T/tools/track3/support_stacked.py $R/tools/track3/
for d in baseline assets; do [ -e $R/$d ] || ln -s $T/$d $R/$d; done
ls -la $R $R/tools/track3; echo SETUP_OK
