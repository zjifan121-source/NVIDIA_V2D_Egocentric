"""k10 check e: run the A2 skeptic's sk10a2.py UNCHANGED except for its scene (SK_SC) and output dir (SK_OUT),
so the original's outputs in ~/v2d_ops/kenc/diag/skeptic_ep10A2/ are never touched.
usage: SK_SC=<scene dir> SK_OUT=<new dir> /home/haisenk/anaconda3/bin/python sk10a2_k10.py"""
import os

SRC = "/home/haisenk/v2d_ops/kenc/diag/skeptic_ep10A2/sk10a2.py"
src = open(SRC).read()
o_out = "OUT = os.path.expanduser('~/v2d_ops/kenc/diag/skeptic_ep10A2')"
o_sc = ("SC = ('/scratch/haisenk-v2d/hand6/bob1_scenes/experiments/haisenk/hand_fixes_v1_20261102/'\n"
        "      'episode_000010/scene')")
assert src.count(o_out) == 1 and src.count(o_sc) == 1, "sk10a2.py changed: refusing"
out = os.environ["SK_OUT"]
assert "skeptic_ep10A2" not in out
src = src.replace(o_out, "OUT = os.environ['SK_OUT']").replace(o_sc, "SC = os.environ['SK_SC']")
exec(compile(src, SRC + " (k10 copy)", "exec"), {"__name__": "__main__", "__file__": SRC})
