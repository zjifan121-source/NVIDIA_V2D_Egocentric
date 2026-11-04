#!/bin/bash
# Pack ONE Track 3 episode exactly as the team wrapper (tools/track3/pack_submission.py, edm_training) + kit packer do,
# by narrowing COPIES of the kit config and sample to that episode. Inputs are only read.
# usage: pack_one.sh <episode> <results dir with episode_XXXXXX.parquet(+json)> <scene meshes dir (episode_XXXXXX/<obj>.obj inside)> <code commit URL> <out.parquet>
set -u
EP=$1 RES=$2 MESH=$3 URL=$4 OUT=$5
K=/data/user_data/haisenk/v2d/v2d_train/assets/data/v2d_submission_kit; PY=/home/haisenk/v2d_ops/kaggle_env/bin/python
W=$(mktemp -d /tmp/pack_one_XXXX); mkdir -p $W/config $W/results $W/meshes
E6=$(printf %06d $EP)
$PY - "$K" "$W" "$EP" "$RES" <<'PY'
import json, sys, shutil, importlib.util, pandas as pd
from pathlib import Path
K, W, ep, res = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
c = json.load(open(K / "config/tracks.json")); c["tracks"]["track_3"]["scored_episodes"] = [ep]
json.dump(c, open(W / "config/tracks.json", "w"), indent=1)
d = pd.read_parquet(K / "data/track_3_sample_submission.parquet"); d[d.row_id.str.startswith(f"track_3/e{ep:06d}/")].to_parquet(W / "sample.parquet", index=False)
spec = importlib.util.spec_from_file_location("team_pack", "/home/haisenk/v2d_ops/team_pack_submission.py"); tp = importlib.util.module_from_spec(spec); spec.loader.exec_module(tp)
roster = {int(k): v for k, v in json.loads((K / "data/track_3_evaluation_objects.json").read_text())["episodes"].items()}
src = res / f"episode_{ep:06d}.parquet"
problems = tp.relabel(src, W / "results" / src.name, roster[ep])
js = res / f"episode_{ep:06d}.json"
if js.exists(): shutil.copy2(js, W / "results" / js.name)
print("roster", roster[ep], "relabel problems:", problems)
PY
mkdir -p $W/meshes/episode_$E6 && cp -L "$MESH"/episode_$E6/* $W/meshes/episode_$E6/ 2>/dev/null || cp -L "$MESH"/* $W/meshes/episode_$E6/
cd $K && $PY tools/pack_submission.py --track track_3 --results $W/results --meshes $W/meshes --code-commit "$URL" --config $W/config/tracks.json --sample $W/sample.parquet -o "$OUT" 2>&1 | tail -12
[ -f "$OUT" ] && $PY - "$OUT" <<'PY'
import sys, pandas as pd
d = pd.read_parquet(sys.argv[1]); print("rows", len(d), "columns", len(d.columns))
for c in ["eval_script_sha256","eval_code_sha256","checkpoint_sha256","packer_script_sha256","uploader_script_sha256","code_commit_url"]:
    if c in d: print(" ", c, d[c].astype(str).unique()[:2])
PY
