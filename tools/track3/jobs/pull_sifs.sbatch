#!/bin/bash
# Pull the team's published reconstruction images (ghcr.io/ryanzhou123232, public) as .sif files.
# Usage: sbatch pull_sifs.sbatch [module ...]   (default: every module the ego hamer path needs)
#SBATCH -J v2d-pull
#SBATCH -p maxlab-cpu
#SBATCH --qos=maxlab_cpu_qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o /data/user_data/%u/v2d/build/logs/%x-%j.log
set -uo pipefail
REG=ghcr.io/ryanzhou123232
SIF=/data/user_data/$USER/v2d/sif
W=/scratch/$USER/pull/$SLURM_JOB_ID
mkdir -p "$SIF" "$W"
trap 'rm -rf "$W"' EXIT
MODS=("$@")
[ ${#MODS[@]} -gt 0 ] || MODS=(foundation_pose grounding_dino sam2 sam3d wilor hamer geocalib droid_slam gsplat_refinement mediapipe moge anycalib)
pull() {
  local m=$1 tag=afe7dad
  [ "$m" = foundation_pose ] && tag=afe7dad-trt-fix   # adds the TensorRT runtime the checkout's default backend needs
  if [ -f "$SIF/v2d_$m.sif" ]; then echo "have v2d_$m.sif"; return 0; fi
  export APPTAINER_CACHEDIR=$W/cache_$m APPTAINER_TMPDIR=$W/tmp_$m
  mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
  for try in 1 2 3; do  # ghcr drops long HTTP/2 streams now and then; cached layers survive a retry
    echo "== $(date +%T) pulling v2d_$m:$tag (try $try)"
    if apptainer pull "$SIF/v2d_$m.sif.part" "docker://$REG/v2d_$m:$tag" > "$W/$m.log" 2>&1; then
      mv "$SIF/v2d_$m.sif.part" "$SIF/v2d_$m.sif"; echo "== $(date +%T) done v2d_$m"; break
    fi
    echo "== $(date +%T) FAILED v2d_$m"; tail -3 "$W/$m.log"; rm -f "$SIF/v2d_$m.sif.part"
  done
  rm -rf "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
}
export -f pull; export REG SIF W
printf '%s\n' "${MODS[@]}" | xargs -P 3 -I{} bash -c 'pull {}'
ls -lh "$SIF"
