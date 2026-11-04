# 我们的 scene 依赖哪些未入库脚本（--code-commit 推送清单，2026-11-04 05:20 UTC 更新）

`check_probe_plan.py` 每次审 plan 都会对变化的段跑 `scene_provenance.py`，在 summary 的 `scripts_to_push` 列出该段 scene 的依赖；下面是全集，按 scene 类型分。
团队仓库里已有（不用推）：`tools/track3/run_load_retarget.py`（loader）、`baseline/robotic_grounding/scripts/retarget/ego_recon_to_sharpa.py`（retarget）、Edmond de66fea3 里的 `tools/track3/relocate_scene.py`、`reports/policy_targeted_2026-10-28/*`（训练/评测/选 plan）、kit 打包器。

| scene 类型 | 位置 | 标记文件 | 依赖脚本 |
|---|---|---|---|
| lead_v3 stage scene（训练/探针主用） | `lead_v3/stage/scenes/episode_X/scene`（A100、A6000） | `../PROVENANCE.json`、`OUTPUT_CHECK.json`、`SUPPORT_V1_SEATED_INSTALLED.json`、`HAND_CONTACT.json`、`SCORE_INPUT_EXPORT.json`、`bundle/MESH_POSE_PAIR.json` | `work/lead-v3-release-20261104/build_v3.py`；`work/lead-sprint-release-20261103/build_stage.py`（+ `publish.py`）；`lead_hand_reattach_v1/{prep_rawhands.py, attach_exclusive.py, build_scene.py, install_support.py, hand_contact.py, compare_hands.py, hand_ledger.py, export_score_input.py, run_hand_stage_remote.sh, p0_pipeline.sh}`；`lead_support_seating_v1/{seat_audit.py, seat_supports.py}` |
| hand-stage scene（rawexcl / kenexcl / control） | `lead_hand_reattach_v1/scenes/epXX_<mode>`、A6000 `extra_scenes/` | 同上减 `PROVENANCE.json`，加 `bundle/HANDFIX_PREP.json`、`bundle/ATTACH_EXCLUSIVE.json` | 同上减 `build_v3.py`；kenexcl 用 `prep_kenhands.py` 代替 `prep_rawhands.py`（Ken 的手来自 HF `haisenk` 前缀） |
| hold scene（逐帧持握 + 杯子 respin） | A100 `lead_v3/handspin/scenes/epXX_hold`、`lead_v3/handspin28/scenes/ep28_hold`；A6000 `extra_scenes/ep28_hold_{old,new}pot` | OUTPUT_CHECK mode=hold，`bundle/ATTACH_HOLD.json` | hand-stage 全集，`attach_exclusive.py` 换成 **A100 专有** `/root/v2d/lead_v3/handspin/{attach_hold.py, run_hold.sh}` 和生成的 `lead_hand_reattach_v1/build_scene_hold.py`；新 pot 再加 v8 那组 |
| rawspin scene（排他 attach + 杯子 respin） | A100 `lead_v3/handspin/scenes/epXX_rawspin` | OUTPUT_CHECK mode=rawspin，`bundle/ATTACH_RESPIN.json` | hand-stage 全集，加 **A100 专有** `/root/v2d/lead_v3/handspin/{attach_respin.py, run_spin.sh}` 和生成的 `lead_hand_reattach_v1/build_scene_spin.py` |
| v8 赢家 scene（ep28 新 pot） | `lead_winners_v8/mini_stage/hand_stage/scenes/ep28_rawexcl` | 同 hand-stage，上层有 `winners.json`、`candidate_bundle/MESH_POSE_PAIR.json` | hand-stage 全集 + `lead_hand_reattach_v1/apply_winners_remote.py` + `lead_retrack_recovery_v8/{build_cases_v8.py, launch_batch.py, evaluate_remote.py, flatten_assets.py, flatten_host.py}`（容器 `v2d_foundation_pose:afe7dad-trt10.7` 内 FoundationPose 重跟踪）+ `lib/`、`metric_scene_refinement/common.py` 代理评分 |

物体层上游：所有 scene 的物体 = lead_sprint_v1（HF `experiments/wentacc/scored20_lead_sprint_v1_20261103`）= fb11（`scored20_contactshape_proxy_mix_v1_20261103`）加 ep05/32/33 换过的勺/杯 mesh 与轨迹。fb11 本身由 path_a 重建管线（FoundationPose 容器 + `metric_scene_refinement`）产生，这部分代码由主线列清单。
手层上游：团队原始手 HF `night_2026-10-24/scenes/episode_X/ken_c_scored_attach/bundle_merged/result.npz`（ep05/35/36/37/38 用 fb11 交付手）；Ken 修手 HF `experiments/haisenk/hand_fixes_wentao_v1_20261103`。
路径全部相对 `NVIDIA_V2D_Egocentric/`；`path_a/ops/experiments_20261103/` 被 .gitignore 忽略，推送时需 `git add -f` 或改 ignore 规则。

**hold 和 respin 脚本**：08:24 UTC 已从 A100 拷到 `path_a/ops/experiments_20261103/lead_handspin_v1/`：`attach_hold.py`、`attach_respin.py`、`com_fix.py`、`repair_grip_align.py`、`run_hold.sh`、`run_spin.sh`、`build_scene_hold.py`、`build_scene_spin.py`，以及运行脚本 `launch_ep.sh`、`watch_run2.sh`、`run_train_v3.sh`。A100 上若再修改，要重新同步。`repair_grip_align.py` 没有标记文件，plan 的 ep25 若用到要手工列入。 `run_hold.sh`/`run_spin.sh` 用 `PYTHONPATH=path_a/ops/experiments_20261029/scored_scene_consistency_v1/runtime`，这个目录同样被 ignore。ep28 HF overlay 另依赖 `work/lead-v3-ep28-release-20261104/{build_ep28.py, publish_ep28.py}`。

**提交层**：若最终文件含行级合并，`lead_probe_review_v1/merge_swaps.py` 也要入库；bob1+ep1+ep5 的 ep1/ep5 来自 Ken 的 T2/T3，最终 commit 还必须包含 Ken 的 hold-window 和 attach/COM 修复代码（待向 Ken 要）。
