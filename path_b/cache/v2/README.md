# cache/v2/ — Path B v2, parked

Path B v3 (`../../V3_BUILD_SPEC.md`) does not run this code; it is kept for reference. Phase 2's
two-object Do as I Do scene takes its slot logic from `spider_scene.py` (spec §8).

| File | Was |
|---|---|
| `PATH_B_V2_PHYSICS_INVERSION.md` | the v2 design (S1–S7 with SPIDER physics retargeting), exported from its Claude doc |
| `spider_scene.py` | stage 2–3 output → SPIDER's bimanual layout (table at z = 0, slots by contact, support plan) |
| `spider_settle.py` | frame-0 settle check of a SPIDER scene in MuJoCo |
| `s4_scene.py` | v2's `s4_scene` stage (was `pathb/stages/s4_scene.py`) |
| `merge_runner.py` | `run_load_retarget.py --merge` on Path B's own P4 bundles (was `pathb/merge_runner.py`) |
| `test_spider_scene.py` | its 10 unit tests (not collected by `pytest path_b/tests`; imports point at the old paths) |

## SPIDER setup (v2, once per instance)

SPIDER (facebookresearch/spider, CC BY-NC 4.0) is installed beside the repo, never copied into it,
in its own Python 3.12 environment: its pins (mujoco 3.7.0, mujoco-warp 3.7.0.1, warp-lang 1.12.1)
differ from FlashCHORD's (mujoco 3.8.1, warp-lang 1.14.0). The commit below is the one Path B's
adapter was written against.

```bash
git clone https://github.com/facebookresearch/spider /root/v2d/spider
git -C /root/v2d/spider checkout 44717007de41cbef7565dff7ff9f4453557a2d3d
cd /root/v2d/spider && uv sync --python 3.12 && uv pip install coacd trimesh   # decompose.py needs both
export SPIDER_DIR=/root/v2d/spider SPIDER_PY=/root/v2d/spider/.venv/bin/python
export SPIDER_ROBOT=xhand            # ships with SPIDER; Sharpa needs SPIDER_ROBOT_ASSETS=<dir with bimanual.xml>
```

Without `SPIDER_DIR` and `SPIDER_PY`, `s4_scene` is skipped with that reason.
