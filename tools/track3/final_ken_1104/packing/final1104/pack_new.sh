set -u
B=/tmp/haisenk_v2d; F=$HOME/v2d_ops/kenc/final1104
URL=https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/commit/1689cf6396bee0f6a3bf4cced610f2fe0f71ebf6
echo "== $(hostname) $(date)"
for r in b_ep1_wh_K_s48 rf_ep8_ds_K_s43 o_ep1_hold_s44 h_ep8_sc_K_s42; do echo "-- scene refs in $r hydra/train.log:"; grep -rhoE "scenes/[A-Za-z0-9_]+" $B/runs/$r/early_100003840/results/hydra 2>/dev/null | sort | uniq -c | head -3; grep -m2 -oE "scenes/[A-Za-z0-9_]+" $B/runs/$r/train.log 2>/dev/null | sort | uniq -c; done
echo "-- mesh md5 w_hold_ep1 / rf_ep8_ds / sc_ep8 score_input"
md5sum $B/scenes/w_hold_ep1/score_input/meshes/episode_000001/*.obj $B/scenes/rf_ep8_ds/score_input/meshes/episode_000008/*.obj $B/scenes/sc_ep8/score_input/meshes/episode_000008/*.obj
mkdir -p $F/b_ep1_wh_K_s48 $F/rf_ep8_ds_K_s43 $F/local_best
pk() { # EP CAND STEP SCENE
  local EP=$1 C=$2 S=$3 SC=$4 E6=$(printf %06d $1)
  local RES=$B/runs/$C/early_$S/results MESH=$B/scenes/$SC/score_input/meshes T=$F/$C
  [ -e $T/one.parquet ] && { echo "$T/one.parquet exists - not overwriting"; return; }
  mkdir -p $T/in/runs/$C/early_$S/results $T/in/scenes/$SC/score_input/meshes
  cp -a $RES/episode_$E6.parquet $RES/episode_$E6.json $T/in/runs/$C/early_$S/results/
  cp -a $B/runs/$C/best.txt $B/runs/$C/sweep.json $B/runs/$C/recipe.txt $T/in/runs/$C/
  cp -aL $MESH/episode_$E6 $T/in/scenes/$SC/score_input/meshes/
  echo "$EP $C step=$S scene=$SC RES=$RES MESH=$MESH URL=$URL" > $T/pack_args.txt
  bash $HOME/v2d_ops/pack_one.sh $EP $RES $MESH $URL $T/one.parquet > $T/pack.log 2>&1
  echo "-- pack $C:"; tail -12 $T/pack.log; ls -la $T/one.parquet; sha256sum $T/one.parquet $RES/episode_$E6.parquet
}
pk 1 b_ep1_wh_K_s48 100003840 w_hold_ep1
pk 8 rf_ep8_ds_K_s43 95006720 rf_ep8_ds
echo "== v2d_submit.py dry-run code"
K=/data/user_data/haisenk/v2d/v2d_train/assets/data/v2d_submission_kit
grep -n "dry_run\|dry-run\|def \|api\.\|competition_submit\|submissions\|KaggleApi\|requests\." $K/submit/v2d_submit.py | head -80
echo "== dryrun.log examples"; cat $HOME/v2d_ops/kenc/tests/n_ep10_nb_K_s42/dryrun.log; cat $HOME/v2d_ops/kenc/combos/base4_n10/dryrun.log
echo "== swap_episode.py head"; head -30 $HOME/v2d_ops/swap_episode.py
