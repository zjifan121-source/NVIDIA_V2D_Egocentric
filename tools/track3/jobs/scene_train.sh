#!/bin/bash
# One episode: build the tracking scene from bundles in namespace $NS (load+retarget, attach hands,
# --merge for two-object clips, extra load flags $LR_EXTRA, default --smooth-hands 2), then one training run
# named ep<EP>_$TAG (one self-requeuing 1 h preempt job) with train_diag.sbatch's default recipe (100M steps,
# object keypoint reward x2; CURRICULUM/STEPS override it). SCORE=1 scores public episodes.
# LR_CPU=1 builds the scene on maxlab-cpu (--device cpu; the GPU queue is the bottleneck);
# SKIP_LR=1 trains on an existing scene. SCENE_SUFFIX=_x writes the scene to <ns>_attach_x (keeps the old one).
# MOTION_SPEED=0.5 trains at half reference speed; TRAIN_EXTRA adds training overrides (e.g. termination settings);
# STEPS changes the run length (default 100M); SEED the training seed (default 42). Seeds differ a lot on
# some clips: train 2-4 seeds per episode and keep the best by own-reference score (select_policies.py).
# CONTACTS=big: 2048-contact buffer on 80 GB cards, for scenes that overflow the default (train.log: 'nefc overflow').
# Usage: NS=fs_noref TAG=obj2x_s42 scene_train.sh EP [EP ...]   (LR_EXTRA="" builds without hand smoothing)
J=/home/$USER/NVIDIA_V2D_Egocentric/tools/track3/jobs
O=/data/user_data/$USER/v2d/recon_data/outputs
: "${NS:?set NS}" "${TAG:?set TAG}"
EXTRA="task.motion_speed=${MOTION_SPEED:-1.0} sim.fps=20.254119873046875 termination.wrist_position.threshold=0.3 training.checkpoint.save_interval_environment_steps=5000192 ${TRAIN_EXTRA:-}"
LR_EXTRA=${LR_EXTRA---smooth-hands 2}
export V2D_BUNDLE_NS=$NS
for EP in "$@"; do
  E6=$(printf %06d "$EP"); OBJS=($(ls "$O/$NS/episode_$E6"))
  if [ ${#OBJS[@]} -gt 1 ]; then SEL=(--merge); else SEL=(--object "${OBJS[0]}"); fi
  OUT=$O/load_retarget/episode_$E6/${NS}_attach${SCENE_SUFFIX:-}; SEQ=ep${EP}_${NS}_attach${SCENE_SUFFIX:-}
  DEP=(); LR=none
  if [ "${SKIP_LR:-0}" != 1 ]; then
    if [ "${LR_CPU:-0}" = 1 ]; then
      LR=$(sbatch --parsable -p maxlab-cpu --qos=maxlab_cpu_qos -c 8 --mem=32G -t 45 -J lr-ep$EP-$NS \
           -o /data/user_data/$USER/v2d/jobs/logs/%x-%j.log --export=ALL --wrap \
           "/data/user_data/$USER/v2d/venvs/retarget/bin/python /home/$USER/NVIDIA_V2D_Egocentric/tools/track3/run_load_retarget.py --episode $EP ${SEL[*]} --attach-hands --skip-video --sequence-id $SEQ --out $OUT --device cpu ${LR_EXTRA:-}")
    else
      # shellcheck disable=SC2086
      LR=$(sbatch --parsable --qos=preempt_qos -J lr-ep$EP-$NS "$J/load_retarget.sbatch" --episode "$EP" "${SEL[@]}" \
           --attach-hands --skip-video --sequence-id "$SEQ" --out "$OUT" ${LR_EXTRA:-})
    fi
    DEP=(--dependency=afterok:$LR)
  fi
  P=$OUT/assets/human_motion_data/ego_recon/processed/sequence_id=$SEQ/robot_name=sharpa_wave
  BIG=(); X=""
  [ "${CONTACTS:-}" = big ] && BIG=(--constraint='VRAM_80GB|VRAM_141GB') && X=" sim.njmax=2048 sim.nconmax=1024"
  T=$(sbatch --parsable -J t-ep${EP}_$TAG --qos=preempt_qos "${DEP[@]}" "${BIG[@]}" \
      --export=ALL,NAME=ep${EP}_$TAG,EP=$EP,STEPS=${STEPS:-100003840},SCORE=${SCORE:-0},PARQUET="$P",EXTRA="$EXTRA$X" "$J/train_diag.sbatch")
  echo "ep$EP (${OBJS[*]}) $NS: lr $LR train $T"
done
