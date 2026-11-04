set -u
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/root/.cache/uv-lead UV_HTTP_TIMEOUT=600
K=/root/v2d/NVIDIA_V2D_Egocentric/assets/data/v2d_submission_kit
cat $K/requirements.txt | cut -c1-100
uv venv /root/venv-kit --python 3.11 2>&1 | tail -1
uv pip install --python /root/venv-kit/bin/python -r $K/requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -3 | cut -c1-160
/root/venv-kit/bin/python -c "import fast_simplification,trimesh,pandas,pyarrow;print('KIT_IMPORTS_OK')"
M=/root/v2d/NVIDIA_V2D_Egocentric; W=/root/probes/rehearsal_ep33
cd $M; rm -rf $W/work
time /root/venv-cpu/bin/python tools/track3/pack_submission_handoff.py --plan $W/plan.json --work $W/work --code-commit https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/commit/de66fea3 --python /root/venv-kit/bin/python -o $W/rehearsal.parquet 2>&1 | tail -22 | cut -c1-230
ls -la $W/rehearsal.parquet 2>&1 | cut -c20-
echo "== submit dry-run (no upload)"
cd $K && /root/venv-kit/bin/python submit/v2d_submit.py $W/rehearsal.parquet --track track_3 --dry-run -m "rehearsal" 2>&1 | tail -14 | cut -c1-230
echo KIT_REHEARSAL_END
