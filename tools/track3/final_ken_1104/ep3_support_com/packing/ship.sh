#!/bin/bash
# ship.sh NAME SCENE EP [STEP]  (local Git Bash): copy an orchard run's best-checkpoint rollout (early_<STEP>/results; STEP
# from best.txt by default, "final" = the old driver's results/) + its scene's score_input/meshes to Babel
# ~/v2d_ops/kenc/tests/NAME/in/ ; prints the paths to pass to swap_submit.sh.
set -u
NAME=$1 SCENE=$2 EP=$3 STEP=${4:-}
E6=$(printf %06d $EP); TMP=$(mktemp -d)
timeout 300 ssh orchard "srun --jobid=157612 --overlap -n1 bash -c 'cd /tmp/haisenk_v2d && S=$STEP && if [ -z \"\$S\" ]; then S=\$(cut -d\" \" -f1 runs/$NAME/best.txt); fi && if [ \"\$S\" = final ]; then R=runs/$NAME/results; else R=runs/$NAME/early_\$S/results; fi && echo \$R > /tmp/haisenk_v2d/ship_$NAME.txt && tar czf - \$R/episode_$E6.parquet \$R/episode_$E6.json \$(ls runs/$NAME/best.txt runs/$NAME/sweep.json 2>/dev/null) /tmp/haisenk_v2d/ship_$NAME.txt scenes/$SCENE/score_input/meshes 2>/dev/null'" > $TMP/x.tgz 2>/dev/null
ls -la $TMP/x.tgz
timeout 400 ssh -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=5 babel "rm -rf ~/v2d_ops/kenc/tests/$NAME/in && mkdir -p ~/v2d_ops/kenc/tests/$NAME/in && tar xzf - -C ~/v2d_ops/kenc/tests/$NAME/in && cd ~/v2d_ops/kenc/tests/$NAME/in && R=\$(cat tmp/haisenk_v2d/ship_$NAME.txt) && echo RES=\$HOME/v2d_ops/kenc/tests/$NAME/in/\$R && echo MESH=\$HOME/v2d_ops/kenc/tests/$NAME/in/scenes/$SCENE/score_input/meshes && ls \$R scenes/$SCENE/score_input/meshes/* && cat runs/$NAME/best.txt 2>/dev/null" < $TMP/x.tgz
rm -rf $TMP
