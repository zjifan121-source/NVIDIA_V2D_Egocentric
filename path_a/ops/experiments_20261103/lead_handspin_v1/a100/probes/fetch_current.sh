set -u
cd /tmp
export HF_ENDPOINT=https://hf-mirror.com
timeout 400 /root/venv-cpu/bin/python /tmp/fetch_current.py 2>&1 | grep -v "it/s\|Fetching" | cut -c1-260
R=/root/lead_current/experiments/wentacc/scored20_lead_v3_hands_r1_20261104/scenes
K=/root/lead_current/experiments/haisenk/hand_fixes_wentao_v1_20261103/episode_000001/scene
S="$R/episode_000025/scene $R/episode_000028/scene $R/episode_000033/scene $K"
find /root/lead_current -name '._*' -type f -delete
/root/venv-cpu/bin/python /root/edmond_handoff_de66fea/tools/track3/relocate_scene.py $S 2>&1 | sed 's#^.*parquet: ##' | cut -c1-140
V2D_ROOT=/root/v2d LEAD_ROSTER=/root/edmond_handoff_de66fea/tools/track3/episodes.json /root/venv-cpu/bin/python /root/v2d/tools_lead/export_score_input.py $S 2>&1 | cut -c1-200
/root/venv-cpu/bin/python /tmp/verify_current.py 2>&1 | cut -c1-420
du -sh /root/lead_current | cut -f1
echo CURRENT_FETCH_END
