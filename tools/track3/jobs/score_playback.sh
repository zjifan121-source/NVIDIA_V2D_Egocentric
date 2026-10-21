#!/bin/bash
# Official-metric playback scores (no simulation) for reconstruction namespaces on public episodes.
# Usage: sbatch -p maxlab-cpu --qos=maxlab_cpu_qos -c 16 --mem=96G -t 40 --wrap "bash tools/track3/jobs/score_playback.sh <out.json> <ns,...> <ep,...>"
# (runs out of memory in a small interactive allocation; ~2 min for 8 episodes)
cd "$(dirname "$0")/../../.." || exit 1
uv run --project tools --with networkx python tools/track3/score_playback.py "$@"
