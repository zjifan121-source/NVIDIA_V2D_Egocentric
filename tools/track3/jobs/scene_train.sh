#!/bin/bash
# One episode: build the tracking scene from bundles in namespace $NS (load+retarget, attach hands,
# --merge for two-object clips, extra load flags $LR_EXTRA), then a full-recipe 250M-step run
# named ep<EP>_$TAG (one self-requeuing 1 h preempt job). SCORE=1 scores public episodes.
# LR_CPU=1 builds the scene on maxlab-cpu (--device cpu; the GPU queue is the bottleneck);
# SKIP_LR=1 trains on an existing scene.
# Usage: NS=fs_noref TAG=noref250M LR_EXTRA="--smooth-hands 2" scene_train.sh EP [EP ...]
J=/home/$USER/NVIDIA_V2D_Egocentric/tools/track3/jobs
O=/data/user_data/$USER/v2d/recon_data/outputs
: "${NS:?set NS}" "${TAG:?set TAG}"
EXTRA="task.motion_speed=1.0 sim.fps=20.254119873046875 termination.wrist_position.threshold=0.3 training.checkpoint.save_interval_environment_steps=5000192"
export V2D_BUNDLE_NS=$NS
for EP in "$@"; do
  E6=$(printf %06d "$EP"); OBJS=($(ls "$O/$NS/episode_$E6"))
  if [ ${#OBJS[@]} -gt 1 ]; then SEL=(--merge); else SEL=(--object "${OBJS[0]}"); fi
  OUT=$O/load_retarget/episode_$E6/${NS}_attach; SEQ=ep${EP}_${NS}_attach
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
  T=$(sbatch --parsable -J t-ep${EP}_$TAG --qos=preempt_qos "${DEP[@]}" \
      --export=ALL,NAME=ep${EP}_$TAG,EP=$EP,STEPS=250003456,SCORE=${SCORE:-0},PARQUET="$P",EXTRA="$EXTRA" "$J/train_diag.sbatch")
  echo "ep$EP (${OBJS[*]}) $NS: lr $LR train $T"
done
