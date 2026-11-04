# Scripts behind our scenes, training, evaluation, packing and submission

Copied from the hosts by `work/lead-v3-release-20261104/sync_scripts.py`. Re-run it after any edit. Paths are relative to this directory. Host addresses are replaced by `<host>` in the copies.

| Path | What | Host | Source on host | sha256 |
|---|---|---|---|---|
| `attach_hold.py` | per-frame hold attach | A100 | see top level | `4c133b0a1854` |
| `attach_respin.py` | exclusive attach and cup respin | A100 | see top level | `1e7d79208285` |
| `build_scene_hold.py` | scene build with the hold mode | A100 | see top level | `2eeb80f852f1` |
| `build_scene_spin.py` | scene build with the rawspin mode | A100 | see top level | `00d41f138702` |
| `com_fix.py` | audits or applies the URDF inertial origin fix | A100 | see top level | `14acd416d752` |
| `launch_ep.sh` | launches one A100 training | A100 | see top level | `719170aab1c5` |
| `repair_grip_align.py` | moves the hand to a thin object and aligns the pinch | A100 | see top level | `7452c1f2f95f` |
| `run_hold.sh` | builds one hold scene | A100 | see top level | `0d179e3a4ea4` |
| `run_spin.sh` | builds one rawspin scene | A100 | see top level | `d9cbc900f440` |
| `run_train_v3.sh` | trains one policy on the A100 | A100 | see top level | `7ca2ad7eedfe` |
| `watch_run2.sh` | evaluates an A100 run as it finishes | A100 | see top level | `702858271ec4` |
| `a100/audit/blast_check2.py` | verifies the ep10/ep14 blast scenes | A100 | `/root/probes/blast_check2.py` | `aae046aad44b` |
| `a100/audit/com_check.py` | file-level diff of the ep28 COM scene and centroid check | A100 | `/root/probes/com_check.py` | `abd734f2d86c` |
| `a100/audit/hold_audit.py` | wrist before and after attach on held frames, found the false grasps | A100 | `/root/probes/hold_audit.py` | `ac0c61662127` |
| `a100/audit/hold_audit2.py` | wrist drag by attach on all frames, per hand | A100 | `/root/probes/hold_audit2.py` | `d35d731e2fdc` |
| `a100/audit/level_check.py` | verifies a re-levelled scene against lead_v3 | A100 | `/root/probes/level_check.py` | `bd289d61cf9a` |
| `a100/audit/oldpot_check.py` | verifies the old-pot ep28 hold+COM scene and the pot mesh size | A100 | `/root/probes/oldpot_check.py` | `edbf88ca62c3` |
| `a100/audit/replay_size.py` | first replay memory probe | A100 | `/root/probes/replay_size.py` | `0eb677e813d7` |
| `a100/audit/replay_size2.py` | replay memory probe, network shapes | A100 | `/root/probes/replay_size2.py` | `49584d2f4e91` |
| `a100/audit/replay_size3.py` | replay memory probe, checkpoint shapes | A100 | `/root/probes/replay_size3.py` | `bd7f8ba6ddbf` |
| `a100/audit/replay_size4.py` | replay memory probe, critic schema from checkpoint metadata | A100 | `/root/probes/replay_size4.py` | `56598e3b8069` |
| `a100/audit/resthold_check.py` | verifies the rest-hold ep33 scene against the spin-and-roll scene | A100 | `/root/probes/resthold_check.py` | `b02835b9fe68` |
| `a100/audit/roll_check.py` | measures what the spoon roll fix changed | A100 | `/root/probes/roll_check.py` | `21f6c7da76d9` |
| `a100/audit/scene_diff.py` | column-level diff of training scenes against lead_v3 | A100 | `/root/probes/scene_diff.py` | `317b6c5af669` |
| `a100/audit/scene_fp.py` | content fingerprint of training scenes | A100 | `/root/probes/scene_fp.py` | `ff5c10e56076` |
| `a100/audit/scene_sig.py` | content fingerprint of a scene for cross-host comparison | A100 | `/root/probes/scene_sig.py` | `8911c79463b6` |
| `a100/audit/spin_check.py` | measures cup spin in the plain and fixed ep33 scenes | A100 | `/root/probes/spin_check.py` | `89497fe6903b` |
| `a100/blast/relevel.py` | re-level or identity re-seat with updated bundle hashes, drops flush disks | A100 | `/root/v2d/lead_v3/blast/relevel.py` | `b58aa4e23919` |
| `a100/blast/run_blast.sh` | blast chain: Ken hands, loader clean, cups shifted to the depth table, hold or exclusive attach with respin, re-seat, scene, COM, score_input | A100 | `/root/v2d/lead_v3/blast/run_blast.sh` | `8bb446376a6a` |
| `a100/blast/shift_cups_to_table.py` | moves a floating cup mesh in its body frame down to the depth-fitted table at frame 0 | A100 | `/root/v2d/lead_v3/blast/shift_cups_to_table.py` | `7fed5dac474c` |
| `a100/blastx/extend_cups_down.py` | extrudes the lowest band of a floating cup down to the depth-fitted table and closes the bottom | A100 | `/root/v2d/lead_v3/blastx/extend_cups_down.py` | `1ff52232110e` |
| `a100/blastx/run_blastx.sh` | blast chain with the cup extension instead of the shift | A100 | `/root/v2d/lead_v3/blastx/run_blastx.sh` | `5695968f6af6` |
| `a100/diag/jump_diag.py` | finds wrist jumps between a bundle and raw hands | A100 | `/root/jump_diag.py` | `15c4c8bdafd5` |
| `a100/diag/roll_diag.py` | spoon axis, hold radius and pose jumps | A100 | `/root/roll_diag.py` | `05fda22ce7fc` |
| `a100/diag/roll_eval.py` | compares spoon roll rules | A100 | `/root/roll_eval.py` | `be448085eb45` |
| `a100/diag/roll_rest.py` | tracked spoon roll at rest against the hold | A100 | `/root/roll_rest.py` | `95fa52c64c9e` |
| `a100/diag/spin_check.py` | spin and orbit check for held objects | A100 | `/root/spin_check.py` | `dc6042811e87` |
| `a100/handspin25/run.sh` | builds the repaired ep25 scene | A100 | `/root/v2d/lead_v3/handspin25/run.sh` | `4b9646f4d7ae` |
| `a100/handspin25/scan.py` | scans grip repair parameters for ep25 board 2 | A100 | `/root/v2d/lead_v3/handspin25/scan.py` | `7806974c6950` |
| `a100/handspin25/tips.py` | measures fingertip distance to the board | A100 | `/root/v2d/lead_v3/handspin25/tips.py` | `98e94b88d807` |
| `a100/handspin28r2/run.sh` | builds the ep28 r2 scene | A100 | `/root/v2d/lead_v3/handspin28r2/run.sh` | `f594778c8ff2` |
| `a100/handspinroll/attach_respinroll.py` | exclusive attach, cup respin and frozen spoon roll; LEAD_SPIN_SRC overrides the input bundle | A100 | `/root/v2d/lead_v3/handspinroll/attach_respinroll.py` | `1d28542eb50c` |
| `a100/handspinroll/build_scene_spinroll.py` | scene build with the rawspinroll mode | A100 | `/root/v2d/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1/build_scene_spinroll.py` | `c28d593d6cdd` |
| `a100/handspinroll/run_roll.sh` | builds one spinroll scene | A100 | `/root/v2d/lead_v3/handspinroll/run_roll.sh` | `ceec29ce94d5` |
| `a100/kg/kg_board.py` | reads our rank on each board | A100 | `/root/kg_board.py` | `530413cc7edd` |
| `a100/kg/kg_desc.py` | prints recent submission descriptions | A100 | `/root/kg_desc.py` | `9676bb04fb8a` |
| `a100/kg/kg_latest.py` | lists recent team submissions | A100 | `/root/kg_latest.py` | `dc08240bae11` |
| `a100/kg/kg_mppe.py` | reads the MPPE board | A100 | `/root/kg_mppe.py` | `fd9e43ae26cc` |
| `a100/level/relevel.py` | re-levels a bundle to the depth-fitted table normal and re-seats its support | A100 | `/root/v2d/lead_v3/level/relevel.py` | `9c25fd956fc4` |
| `a100/level/run_level.sh` | builds a re-levelled hold scene with COM patch and score_input | A100 | `/root/v2d/lead_v3/level/run_level.sh` | `09ef8b782a6f` |
| `a100/probes/base_swaps.py` | prints merge arguments for the current incumbents | A100 | `/root/probes/base_swaps.py` | `408e060ca22d` |
| `a100/probes/checkpoint_feed.py` | appends every newly evaluated checkpoint to NEW_CHECKPOINTS.jsonl/.md for the review session | A100 | `/root/probes/checkpoint_feed.py` | `1247cddc0bf0` |
| `a100/probes/checkpoint_feed.sh` | daemon around checkpoint_feed.py | A100 | `/root/probes/checkpoint_feed.sh` | `a3b9fd730579` |
| `a100/probes/fetch_current.py` | downloads the scenes named by the current manifest | A100 | `/root/probes/fetch_current.py` | `c024694fef8f` |
| `a100/probes/fetch_current.sh` | downloads, relocates and verifies the manifest scenes | A100 | `/root/probes/fetch_current.sh` | `9a59d1885f65` |
| `a100/probes/five_board/retry_failed.py` | retries a failed board alone after checking it holds no copy | A100 | `/root/probes/five_board/retry_failed.py` | `eb3085f8e174` |
| `a100/probes/five_board/retry_file.py` | same for a base file | A100 | `/root/probes/five_board/retry_file.py` | `1bc56791e718` |
| `a100/probes/five_board/run_list.sh` | runs the five-board list one file at a time | A100 | `/root/probes/five_board/run_list.sh` | `f78ed758ab3c` |
| `a100/probes/five_board/run_list2.sh` | same, also retries network auth failures | A100 | `/root/probes/five_board/run_list2.sh` | `e138183886a0` |
| `a100/probes/five_board/submit_list.sh` | uploads a scored team file to boards it has not been on | A100 | `/root/probes/five_board/submit_list.sh` | `605005cf1ac6` |
| `a100/probes/hf_ken.py` | lists what Ken has on HF for ep1, ep5 and ep28 | A100 | `/root/probes/hf_ken.py` | `71b660d44920` |
| `a100/probes/incoming/sync_a6000.sh` | pulls every evaluated A6000 checkpoint to the A100 for the review session proxy | A100 | `/root/probes/incoming/sync_a6000.sh` | `74eff3be09e0` |
| `a100/probes/kaggle_read.py` | reads quota and recent submissions per board, no upload | A100 | `/root/probes/kaggle_read.py` | `47d6e60c1d25` |
| `a100/probes/kaggle_read2.py` | reads the recent AUC submission history, no upload | A100 | `/root/probes/kaggle_read2.py` | `28da9bb67883` |
| `a100/probes/ken28_packer.sh` | auto-packer: packs the judged checkpoint with the largest emulator margin, dry-run, sha read-back, PACK_REPORT.json | A100 | `/root/probes/ken28_packer.sh` | `f7038bd8ac5b` |
| `a100/probes/manifest_r2.py` | reads manifest r2 and fingerprints the ep28 r2 scene | A100 | `/root/probes/manifest_r2.py` | `dee47c92dac9` |
| `a100/probes/manifest_read.py` | reads manifest r1 from HF | A100 | `/root/probes/manifest_read.py` | `219903e5ad9e` |
| `a100/probes/manifest_read2.py` | prints manifest r1 entries, the ledger and the overlay README | A100 | `/root/probes/manifest_read2.py` | `9268cb528c68` |
| `a100/probes/pack_only.sh` | check + pack + merge onto the current base for one checkpoint (PACK_STEP), no Kaggle call | A100 | `/root/probes/pack_only.sh` | `35b4c7735be2` |
| `a100/probes/probe.sh` | single-episode swap probe on the A100 | A100 | `/root/probes/probe.sh` | `4d9310cace68` |
| `a100/probes/proxy_read.py` | prints the review proxy records with their fields | A100 | `/root/probes/proxy_read.py` | `552fc63ef57f` |
| `a100/probes/sub_t5.sh` | six-board upload of the T5 base | A100 | `/root/probes/sub_t5.sh` | `97f783f0b78c` |
| `a100/probes/sub_t5_five.sh` | retry of the T5 base on five boards | A100 | `/root/probes/sub_t5_five.sh` | `7ca0bff45397` |
| `a100/probes/sub_t5_mp.sh` | retry of the T5 base on MP-SR | A100 | `/root/probes/sub_t5_mp.sh` | `e94b6327360c` |
| `a100/probes/sub_t6base.sh` | six-board upload of the T6 base, then the five-board list | A100 | `/root/probes/sub_t6base.sh` | `bbb542bfc8a0` |
| `a100/probes/verify2.py` | identifies the ep28 parquets and verifies the ep28, ep33 and ep1 scenes | A100 | `/root/probes/verify2.py` | `e33210362905` |
| `a100/probes/verify_current.py` | compares manifest scenes with the running scenes | A100 | `/root/probes/verify_current.py` | `f0029e47fe26` |
| `a100/q/check_first.py` | reads the first checkpoint results of a run | A100 | `/root/v2d/q/check_first.py` | `547d7dda80ce` |
| `a100/q/collect_host.py` | read-only run status for the live page | A100 | `/root/v2d/q/collect_host.py` | `8134305a5cb4` |
| `a100/q/env.sh` | paths and environment for A100 training | A100 | `/root/v2d/q/env.sh` | `63e19a6291a8` |
| `a100/q/eval_daemon.sh` | evaluates new checkpoints of running trainings | A100 | `/root/v2d/q/eval_daemon.sh` | `c146aff596cf` |
| `a100/q/evaluate_checkpoints.py` | official full rollouts for checkpoints from 70M on | A100 | `/root/v2d/q/evaluate_checkpoints.py` | `4d7233f6146c` |
| `a100/q/queue_v3.sh` | slot queue on the A100 | A100 | `/root/v2d/q/queue_v3.sh` | `2f80c3c6cc63` |
| `a100/q/run_train_v3.sh` | trains one policy with the team recipe on the A100 | A100 | `/root/v2d/q/run_train_v3.sh` | `ca23f159fc9c` |
| `a100/q/watch_eval_ep33.sh` | waits for the ep33 evaluation | A100 | `/root/v2d/q/watch_eval_ep33.sh` | `4bbface1b4d9` |
| `a100/reports_policy_targeted_2026-10-28/evaluate_all70m.py` | official full rollouts for saved checkpoints, used by the A100 run watcher | A100 | `/root/v2d/NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28/evaluate_all70m.py` | `44b79f800b6e` |
| `a100/reports_policy_targeted_2026-10-28/evaluate_checkpoints.py` | team evaluation script that evaluate_all70m.py calls | A100 | `/root/v2d/NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28/evaluate_checkpoints.py` | `22f156c5589f` |
| `a100/resthold/rest_hold_patch.py` | holds each object at its frame-0 pose until its first attach frame, blended over 3 frames | A100 | `/root/v2d/lead_v3/resthold/rest_hold_patch.py` | `c78c07c8bc4e` |
| `a100/resthold/run_resthold.sh` | rest-hold patch of a spinroll bundle, scene build, COM, score_input, spin check | A100 | `/root/v2d/lead_v3/resthold/run_resthold.sh` | `8d4048976584` |
| `a100/run/a100_ops.sh` | relocates scenes, writes launch_scene.sh, restarts ep28 on the new pot | A100 | `/root/a100_ops.sh` | `7d45b6dbf91c` |
| `a100/run/a100_ops2.sh` | writes evaluate_all70m.py, replaces watchers, starts the first queue | A100 | `/root/a100_ops2.sh` | `612a418fbb37` |
| `a100/run/a100_ops3.sh` | stops the superseded ep33 run and reads the other A100 queue | A100 | `/root/a100_ops3.sh` | `ca038f668f96` |
| `a100/run/a100_status.sh` | read-only status of the A100 lead runs | A100 | `/root/a100_status.sh` | `e35c3ea02462` |
| `a100/run/dryrun_pack.sh` | first packing rehearsal, no upload | A100 | `/root/dryrun_pack.sh` | `f2581b128ecd` |
| `a100/run/dryrun_pack2.sh` | packing rehearsal with the handoff packer, no upload | A100 | `/root/dryrun_pack2.sh` | `247009e5c67b` |
| `a100/run/kit_env.sh` | builds the kit environment and rehearses packing | A100 | `/root/kit_env.sh` | `af8e84b7cb88` |
| `a100/run/kit_env2.sh` | kit environment fix and packing rehearsal, no upload | A100 | `/root/kit_env2.sh` | `bb2ffd9c824e` |
| `a100/run/kit_env3.sh` | kit environment fix and uploader dry run, no upload | A100 | `/root/kit_env3.sh` | `9ac77b4a4f7f` |
| `a100/run/launch_after.sh` | waits for a run to exit and launches the next arm (ep28 holdold kenf16 s203) | A100 | `/root/launch_after.sh` | `30a5738b60f6` |
| `a100/run/launch_ep33.sh` | first A100 timing launch of ep33, kept for the record | A100 | `/root/launch_ep33.sh` | `790c37554b9f` |
| `a100/run/launch_scene.sh` | launches one A100 training on an explicit scene, with optional Hydra overrides | A100 | `/root/launch_scene.sh` | `338506600719` |
| `a100/run/queue_a100.sh` | first A100 lead queue, count cap only, superseded | A100 | `/root/queue_a100.sh` | `e4a6b8d5cc3f` |
| `a100/run/queue_a100b.sh` | A100 lead queue with a launch cutoff, superseded; started the scene-fix runs | A100 | `/root/queue_a100b.sh` | `5b75d50d2657` |
| `a100/run/queue_a100c.sh` | A100 lead queue with a GPU memory gate, superseded | A100 | `/root/queue_a100c.sh` | `81d49bb22a7d` |
| `a100/run/queue_a100d.sh` | A100 lead queue, Ken exact with float16 fallback, superseded | A100 | `/root/queue_a100d.sh` | `a53d623da915` |
| `a100/run/queue_a100e.sh` | current A100 lead queue, Ken termination arms with float16 replay and a capacity fallback | A100 | `/root/queue_a100e.sh` | `6e245127ce22` |
| `a100/run/watch_run.sh` | first A100 run watcher, replaced by watch_run2.sh | A100 | `/root/watch_run.sh` | `04f587fea99f` |
| `a100/tools_track3/pack_submission_handoff.py` | Edmond's plan-based packer from the de66fea handoff, unchanged | A100 | `/root/v2d/NVIDIA_V2D_Egocentric/tools/track3/pack_submission_handoff.py` | `24925802731d` |
| `a6000/check_first.py` | reads the first checkpoint results of a run | A6000 | `/mnt/v2d/outputs/experiments_20261103/lead_v3_train/check_first.py` | `547d7dda80ce` |
| `a6000/collect_host.py` | read-only run status for the live page | A6000 | `/mnt/v2d/outputs/experiments_20261103/lead_v3_train/collect_host.py` | `8134305a5cb4` |
| `a6000/env.sh` | paths and environment for A6000 training | A6000 | `/mnt/v2d/outputs/experiments_20261103/lead_v3_train/env.sh` | `e5d0923519fc` |
| `a6000/eval_daemon.sh` | evaluates new checkpoints of running trainings | A6000 | `/mnt/v2d/outputs/experiments_20261103/lead_v3_train/eval_daemon.sh` | `c146aff596cf` |
| `a6000/evaluate_checkpoints.py` | official full rollouts for checkpoints from 70M on, writes checkpoint_selection.json | A6000 | `/mnt/v2d/outputs/experiments_20261103/lead_v3_train/evaluate_checkpoints.py` | `4d7233f6146c` |
| `a6000/probe/check_probe_plan.py` | gate that a probe plan changes only the intended episode and passes the checks | A6000 | `/mnt/v2d/outputs/experiments_20261103/lead_v3_train/probe/check_probe_plan.py` | `94d34df287ec` |
| `a6000/probe/com_fix.py` | audits or applies the URDF inertial origin fix | A6000 | `/mnt/v2d/outputs/experiments_20261103/lead_v3_train/probe/com_fix.py` | `14acd416d752` |
| `a6000/probe/export_score_input.py` | adds score_input/ to scenes so they can be packed | A6000 | `/mnt/v2d/outputs/experiments_20261103/lead_v3_train/probe/export_score_input.py` | `dea8cf0a044f` |
| `a6000/probe/kg_status.py` | reads our rank on each board | A6000 | `/mnt/v2d/outputs/experiments_20261103/lead_v3_train/probe/kg_status.py` | `3363f39809bb` |
| `a6000/probe/make_final.sh` | builds the current base citing a given commit, uploads nothing | A6000 | `/mnt/v2d/outputs/experiments_20261103/lead_v3_train/probe/make_final.sh` | `636ee1a7b567` |
| `a6000/probe/make_probe.py` | builds, gates, packs and merges a single-episode probe on the current base | A6000 | `/mnt/v2d/outputs/experiments_20261103/lead_v3_train/probe/make_probe.py` | `74816246de81` |
| `a6000/probe/merge_swaps.py` | row-level merge of packed submissions | A6000 | `/mnt/v2d/outputs/experiments_20261103/lead_v3_train/probe/merge_swaps.py` | `37d2ed5723d4` |
| `a6000/probe/rollout_diag.py` | per-object tracking diagnosis of an evaluated rollout | A6000 | `/mnt/v2d/outputs/experiments_20261103/lead_v3_train/probe/rollout_diag.py` | `d3913c630493` |
| `a6000/probe/show_prov.py` | prints the digest columns of a packed parquet | A6000 | `/mnt/v2d/outputs/experiments_20261103/lead_v3_train/probe/show_prov.py` | `9fb9dde95ca3` |
| `a6000/probe/submit_base.sh` | uploads a merged base to AUC, then the other five boards, token on stdin | A6000 | `/mnt/v2d/outputs/experiments_20261103/lead_v3_train/probe/submit_base.sh` | `5df5f627b924` |
| `a6000/probe/submit_probe.sh` | uploads a probe with the kit uploader, token on stdin | A6000 | `/mnt/v2d/outputs/experiments_20261103/lead_v3_train/probe/submit_probe.sh` | `3d0922667caf` |
| `a6000/probe/swap_pack_a6000.py` | packs bob1 with named episodes swapped for other real rollouts | A6000 | `/mnt/v2d/outputs/experiments_20261103/lead_v3_train/probe/swap_pack_a6000.py` | `041e0f566c5f` |
| `a6000/probe/track3_pack.py` | wrapper around the kit packer | A6000 | `/mnt/v2d/outputs/experiments_20261103/lead_v3_train/probe/track3_pack.py` | `98ec135fa081` |
| `a6000/queue_v3.sh` | slot queue, starts the next training when a slot frees | A6000 | `/mnt/v2d/outputs/experiments_20261103/lead_v3_train/queue_v3.sh` | `2f80c3c6cc63` |
| `a6000/relocate_scene.py` | points absolute paths in a scene at this copy of the scene | A6000 | `/mnt/v2d/outputs/experiments_20261103/lead_v3_train/stage/tools/relocate_scene.py` | `d1f55474130d` |
| `a6000/run_train_v3.sh` | trains one policy with the team recipe, then evaluates checkpoints | A6000 | `/mnt/v2d/outputs/experiments_20261103/lead_v3_train/run_train_v3.sh` | `ca23f159fc9c` |
| `a6000/tools_lead/plane_for_level.py` | depth-plane fit per episode that also prints the normal and plane point (review session fit, unchanged math) | A6000 | `/home/ubuntu/tools_lead/plane_for_level.py` | `c933b34e6f87` |
| `mac/build_manifest.py` | writes CURRENT_MANIFEST and VERSION_LEDGER | Mac | `work/lead-v3-release-20261104/manifest/build_manifest.py` | `58020780025f` |
| `mac/build_overlay_r1.py` | builds hands overlay r1 | Mac | `work/lead-v3-release-20261104/build_overlay_r1.py` | `883c76abf117` |
| `mac/build_overlay_r2.py` | builds hands overlay r2 | Mac | `work/lead-v3-release-20261104/build_overlay_r2.py` | `0cff6d37504b` |
| `mac/build_overlay_r3.py` | builds hands overlay r3 | Mac | `work/lead-v3-release-20261104/build_overlay_r3.py` | `da93a1deefa4` |
| `mac/build_v3.py` | builds the lead_v3 release | Mac | `work/lead-v3-release-20261104/build_v3.py` | `7957bd166719` |
| `mac/publish_note_r1.py` | add-only HF upload of the r1 note | Mac | `work/lead-v3-release-20261104/publish_note_r1.py` | `7a88ede3b507` |
| `mac/publish_overlay_r1.py` | add-only HF upload of r1 | Mac | `work/lead-v3-release-20261104/publish_overlay_r1.py` | `56d3e6a03091` |
| `mac/publish_r2.py` | add-only HF upload of r2 | Mac | `work/lead-v3-release-20261104/publish_r2.py` | `1c414075802d` |
| `mac/publish_r3.py` | add-only HF upload of r3 | Mac | `work/lead-v3-release-20261104/publish_r3.py` | `409b46fb1a5a` |
| `mac/publish_v3.py` | add-only HF upload of v3 with SHA read-back | Mac | `work/lead-v3-release-20261104/publish_v3.py` | `4e476b59f318` |
