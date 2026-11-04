set -u
C=$HOME/v2d_ops/kenc/cdo; PY=/home/haisenk/v2d_ops/kaggle_env/bin/python
K=/data/user_data/haisenk/v2d/v2d_train/assets/data/v2d_submission_kit
URL=https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/commit/1689cf6396bee0f6a3bf4cced610f2fe0f71ebf6
N=$HOME/v2d_ops/kenc/tests/n_ep10_nb_K_s42/in; T6=$HOME/v2d_ops/kenc/tests/h_ep8_sc_K_s42/in
RES=$N/$(cat $N/tmp/haisenk_v2d/ship_n_ep10_nb_K_s42.txt); echo RES=$RES; ls $RES | head -3
SRC=$N/scenes/k10_ep10_nest_b/score_input/meshes/episode_000010
B6=$(ls $T6/scenes/*/score_input/meshes/episode_000008/beige_cup.obj); echo T6 beige: $B6; md5sum $B6 $SRC/*.obj
M=$C/m10/meshes/episode_000010; mkdir -p $M; cp $SRC/* $M/; cp $B6 $M/beige_cup.obj; md5sum $M/*.obj
rm -f $C/m10/one.parquet $C/m10/full.parquet $C/m10/full.cd-o.parquet
bash $HOME/v2d_ops/pack_one.sh 10 $RES $C/m10/meshes $URL $C/m10/one.parquet > $C/m10/pack.log 2>&1; tail -4 $C/m10/pack.log
$PY $HOME/v2d_ops/swap_episode.py $HOME/v2d_ops/kenc/combos/base4/track3_submission.parquet $C/m10/one.parquet 10 $C/m10/full.parquet --adopt-commit-url 2>&1 | tail -2
cd $K && $PY submit/v2d_submit.py $C/m10/full.parquet --track track_3 --only cd_o_cm --dry-run -m test > $C/m10/dry_full.log 2>&1; tail -6 $C/m10/dry_full.log
ls -la $C/m10/
