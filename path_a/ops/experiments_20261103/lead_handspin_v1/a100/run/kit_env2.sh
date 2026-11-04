set -u
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/root/.cache/uv-lead UV_HTTP_TIMEOUT=600
uv pip install --python /root/venv-kit/bin/python pillow scipy --index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -2 | cut -c1-120
K=/root/v2d/NVIDIA_V2D_Egocentric/assets/data/v2d_submission_kit
M=/root/v2d/NVIDIA_V2D_Egocentric; W=/root/probes/rehearsal_ep33
cd $M; rm -rf $W/work
time /root/venv-cpu/bin/python tools/track3/pack_submission_handoff.py --plan $W/plan.json --work $W/work --code-commit https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/commit/de66fea3 --python /root/venv-kit/bin/python -o $W/rehearsal.parquet 2>&1 | tail -20 | cut -c1-210
ls -la $W/rehearsal.parquet 2>&1 | cut -c20-
echo "== submit dry-run (no upload)"
cd $K && /root/venv-kit/bin/python submit/v2d_submit.py $W/rehearsal.parquet --track track_3 --dry-run -m "rehearsal" 2>&1 | tail -14 | cut -c1-210
echo "== swap verification"
/root/venv-kit/bin/python - <<PY
import pandas as pd
d=pd.read_parquet("$W/rehearsal.parquet"); print(d.shape, [c for c in d.columns][:14])
d['ep']=d.row_id.str.extract(r'track_3/e(\d+)/').astype(int)
print("episodes",d.ep.nunique(),"distinct checkpoints",d.checkpoint_sha256.nunique(),"eval digests",d.eval_script_sha256.nunique() if 'eval_script_sha256' in d else '-',d.eval_code_sha256.nunique() if 'eval_code_sha256' in d else '-')
print("ep33 checkpoint",d[d.ep==33].checkpoint_sha256.iloc[0][:16])
PY
echo KIT_REHEARSAL2_END
