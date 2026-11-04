export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/root/.cache/uv-lead UV_HTTP_TIMEOUT=600
uv pip install --python /root/venv-kit/bin/python six --index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -1 | cut -c1-100
K=/root/v2d/NVIDIA_V2D_Egocentric/assets/data/v2d_submission_kit; W=/root/probes/rehearsal_ep33
cd $K && timeout 150 /root/venv-kit/bin/python submit/v2d_submit.py $W/rehearsal.parquet --track track_3 --dry-run -m "rehearsal" 2>&1 | tail -16 | cut -c1-200
echo "rc=$?"; ls -la --time-style=+%H:%M $W | cut -c30-; ls $W/*receipt* 2>/dev/null | wc -l
