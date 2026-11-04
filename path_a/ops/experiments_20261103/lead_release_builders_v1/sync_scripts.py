"""Copy every script that touches our scenes, training, evaluation, packing or submission into the repo and write SCRIPTS_INDEX.md.
Re-run after any edit on a host. Reads the hosts only. Refuses if a copied file holds a secret or a host address.
usage: python3 sync_scripts.py"""
import hashlib, re, subprocess, tarfile, io, shutil
from pathlib import Path
W = Path(__file__).resolve().parent
OUT = Path('/Volumes/E/Code/V2D/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_handspin_v1')
K = str(Path.home() / '.ssh/hyperstack_v2d')
SSH6 = ['ssh', '-i', K, '-o', 'StrictHostKeyChecking=yes', '-o', 'HostKeyAlias=185.216.22.156', 'ubuntu@185.216.20.121']
SSH1 = ['ssh', '-i', K, '-o', 'StrictHostKeyChecking=yes', '-p', '16824', 'root@223.109.239.36']
B = '/mnt/v2d/outputs/experiments_20261103/lead_v3_train'
H = '/root/v2d/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1'
A6 = {  # repo path under OUT: (source on host, what)
 'a6000/env.sh': (f'{B}/env.sh', 'paths and environment for A6000 training'),
 'a6000/queue_v3.sh': (f'{B}/queue_v3.sh', 'slot queue, starts the next training when a slot frees'),
 'a6000/run_train_v3.sh': (f'{B}/run_train_v3.sh', 'trains one policy with the team recipe, then evaluates checkpoints'),
 'a6000/evaluate_checkpoints.py': (f'{B}/evaluate_checkpoints.py', 'official full rollouts for checkpoints from 70M on, writes checkpoint_selection.json'),
 'a6000/eval_daemon.sh': (f'{B}/eval_daemon.sh', 'evaluates new checkpoints of running trainings'),
 'a6000/collect_host.py': (f'{B}/collect_host.py', 'read-only run status for the live page'),
 'a6000/check_first.py': (f'{B}/check_first.py', 'reads the first checkpoint results of a run'),
 'a6000/relocate_scene.py': (f'{B}/stage/tools/relocate_scene.py', 'points absolute paths in a scene at this copy of the scene'),
 'a6000/probe/make_probe.py': (f'{B}/probe/make_probe.py', 'builds, gates, packs and merges a single-episode probe on the current base'),
 'a6000/probe/merge_swaps.py': (f'{B}/probe/merge_swaps.py', 'row-level merge of packed submissions'),
 'a6000/probe/check_probe_plan.py': (f'{B}/probe/check_probe_plan.py', 'gate that a probe plan changes only the intended episode and passes the checks'),
 'a6000/probe/track3_pack.py': (f'{B}/probe/track3_pack.py', 'wrapper around the kit packer'),
 'a6000/probe/export_score_input.py': (f'{B}/probe/export_score_input.py', 'adds score_input/ to scenes so they can be packed'),
 'a6000/probe/submit_probe.sh': (f'{B}/probe/submit_probe.sh', 'uploads a probe with the kit uploader, token on stdin'),
 'a6000/probe/submit_base.sh': (f'{B}/probe/submit_base.sh', 'uploads a merged base to AUC, then the other five boards, token on stdin'),
 'a6000/probe/swap_pack_a6000.py': (f'{B}/probe/swap_pack_a6000.py', 'packs bob1 with named episodes swapped for other real rollouts'),
 'a6000/probe/kg_status.py': (f'{B}/probe/kg_status.py', 'reads our rank on each board'),
 'a6000/probe/show_prov.py': (f'{B}/probe/show_prov.py', 'prints the digest columns of a packed parquet'),
 'a6000/probe/rollout_diag.py': (f'{B}/probe/rollout_diag.py', 'per-object tracking diagnosis of an evaluated rollout'),
 'a6000/probe/com_fix.py': (f'{B}/probe/com_fix.py', 'audits or applies the URDF inertial origin fix'),
}
A1 = {
 'a100/q/env.sh': ('/root/v2d/q/env.sh', 'paths and environment for A100 training'),
 'a100/q/queue_v3.sh': ('/root/v2d/q/queue_v3.sh', 'slot queue on the A100'),
 'a100/q/run_train_v3.sh': ('/root/v2d/q/run_train_v3.sh', 'trains one policy with the team recipe on the A100'),
 'a100/q/evaluate_checkpoints.py': ('/root/v2d/q/evaluate_checkpoints.py', 'official full rollouts for checkpoints from 70M on'),
 'a100/q/eval_daemon.sh': ('/root/v2d/q/eval_daemon.sh', 'evaluates new checkpoints of running trainings'),
 'a100/q/collect_host.py': ('/root/v2d/q/collect_host.py', 'read-only run status for the live page'),
 'a100/q/check_first.py': ('/root/v2d/q/check_first.py', 'reads the first checkpoint results of a run'),
 'a100/q/watch_eval_ep33.sh': ('/root/v2d/q/watch_eval_ep33.sh', 'waits for the ep33 evaluation'),
 'a100/handspinroll/attach_respinroll.py': ('/root/v2d/lead_v3/handspinroll/attach_respinroll.py', 'exclusive attach, cup respin and frozen spoon roll; builds the r3 spinroll bundles'),
 'a100/handspinroll/run_roll.sh': ('/root/v2d/lead_v3/handspinroll/run_roll.sh', 'builds one spinroll scene'),
 'a100/handspinroll/build_scene_spinroll.py': (f'{H}/build_scene_spinroll.py', 'scene build with the rawspinroll mode'),
 'a100/handspin25/run.sh': ('/root/v2d/lead_v3/handspin25/run.sh', 'builds the repaired ep25 scene'),
 'a100/handspin25/scan.py': ('/root/v2d/lead_v3/handspin25/scan.py', 'scans grip repair parameters for ep25 board 2'),
 'a100/handspin25/tips.py': ('/root/v2d/lead_v3/handspin25/tips.py', 'measures fingertip distance to the board'),
 'a100/handspin28r2/run.sh': ('/root/v2d/lead_v3/handspin28r2/run.sh', 'builds the ep28 r2 scene'),
 'a100/probes/probe.sh': ('/root/probes/probe.sh', 'single-episode swap probe on the A100'),
 'a100/probes/sub_t5.sh': ('/root/probes/sub_t5.sh', 'six-board upload of the T5 base'),
 'a100/probes/sub_t5_five.sh': ('/root/probes/sub_t5_five.sh', 'retry of the T5 base on five boards'),
 'a100/probes/sub_t5_mp.sh': ('/root/probes/sub_t5_mp.sh', 'retry of the T5 base on MP-SR'),
 'a100/probes/sub_t6base.sh': ('/root/probes/sub_t6base.sh', 'six-board upload of the T6 base, then the five-board list'),
 'a100/probes/five_board/run_list.sh': ('/root/probes/five_board/run_list.sh', 'runs the five-board list one file at a time'),
 'a100/probes/five_board/run_list2.sh': ('/root/probes/five_board/run_list2.sh', 'same, also retries network auth failures'),
 'a100/probes/five_board/retry_failed.py': ('/root/probes/five_board/retry_failed.py', 'retries a failed board alone after checking it holds no copy'),
 'a100/probes/five_board/retry_file.py': ('/root/probes/five_board/retry_file.py', 'same for a base file'),
 'a100/probes/five_board/submit_list.sh': ('/root/probes/five_board/submit_list.sh', 'uploads a scored team file to boards it has not been on'),
 'a100/diag/spin_check.py': ('/root/spin_check.py', 'spin and orbit check for held objects'),
 'a100/diag/roll_diag.py': ('/root/roll_diag.py', 'spoon axis, hold radius and pose jumps'),
 'a100/diag/jump_diag.py': ('/root/jump_diag.py', 'finds wrist jumps between a bundle and raw hands'),
 'a100/diag/roll_eval.py': ('/root/roll_eval.py', 'compares spoon roll rules'),
 'a100/diag/roll_rest.py': ('/root/roll_rest.py', 'tracked spoon roll at rest against the hold'),
 'a100/kg/kg_board.py': ('/root/kg_board.py', 'reads our rank on each board'),
 'a100/kg/kg_latest.py': ('/root/kg_latest.py', 'lists recent team submissions'),
 'a100/kg/kg_desc.py': ('/root/kg_desc.py', 'prints recent submission descriptions'),
 'a100/kg/kg_mppe.py': ('/root/kg_mppe.py', 'reads the MPPE board'),
 'a100/reports_policy_targeted_2026-10-28/evaluate_all70m.py': ('/root/v2d/NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28/evaluate_all70m.py', 'official full rollouts for saved checkpoints, used by the A100 run watcher'),
 'a100/reports_policy_targeted_2026-10-28/evaluate_checkpoints.py': ('/root/v2d/NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28/evaluate_checkpoints.py', 'team evaluation script that evaluate_all70m.py calls'),
 'a100/run/launch_scene.sh': ('/root/launch_scene.sh', 'launches one A100 training on an explicit scene, with optional Hydra overrides'),
 'a100/run/launch_ep33.sh': ('/root/launch_ep33.sh', 'first A100 timing launch of ep33, kept for the record'),
 'a100/run/queue_a100.sh': ('/root/queue_a100.sh', 'first A100 lead queue, count cap only, superseded'),
 'a100/run/queue_a100b.sh': ('/root/queue_a100b.sh', 'A100 lead queue with a launch cutoff, superseded; started the scene-fix runs'),
 'a100/run/queue_a100c.sh': ('/root/queue_a100c.sh', 'A100 lead queue with a GPU memory gate, superseded'),
 'a100/run/queue_a100d.sh': ('/root/queue_a100d.sh', 'A100 lead queue, Ken exact with float16 fallback, superseded'),
 'a100/run/queue_a100e.sh': ('/root/queue_a100e.sh', 'current A100 lead queue, Ken termination arms with float16 replay and a capacity fallback'),
 'a100/run/watch_run.sh': ('/root/watch_run.sh', 'first A100 run watcher, replaced by watch_run2.sh'),
 'a100/run/a100_status.sh': ('/root/a100_status.sh', 'read-only status of the A100 lead runs'),
 'a100/run/a100_ops.sh': ('/root/a100_ops.sh', 'relocates scenes, writes launch_scene.sh, restarts ep28 on the new pot'),
 'a100/run/a100_ops2.sh': ('/root/a100_ops2.sh', 'writes evaluate_all70m.py, replaces watchers, starts the first queue'),
 'a100/run/a100_ops3.sh': ('/root/a100_ops3.sh', 'stops the superseded ep33 run and reads the other A100 queue'),
 'a100/run/kit_env.sh': ('/root/kit_env.sh', 'builds the kit environment and rehearses packing'),
 'a100/run/kit_env2.sh': ('/root/kit_env2.sh', 'kit environment fix and packing rehearsal, no upload'),
 'a100/run/kit_env3.sh': ('/root/kit_env3.sh', 'kit environment fix and uploader dry run, no upload'),
 'a100/run/dryrun_pack.sh': ('/root/dryrun_pack.sh', 'first packing rehearsal, no upload'),
 'a100/run/dryrun_pack2.sh': ('/root/dryrun_pack2.sh', 'packing rehearsal with the handoff packer, no upload'),
 'a100/probes/base_swaps.py': ('/root/probes/base_swaps.py', 'prints merge arguments for the current incumbents'),
 'a100/probes/fetch_current.py': ('/root/probes/fetch_current.py', 'downloads the scenes named by the current manifest'),
 'a100/probes/fetch_current.sh': ('/root/probes/fetch_current.sh', 'downloads, relocates and verifies the manifest scenes'),
 'a100/probes/verify_current.py': ('/root/probes/verify_current.py', 'compares manifest scenes with the running scenes'),
 'a100/probes/verify2.py': ('/root/probes/verify2.py', 'identifies the ep28 parquets and verifies the ep28, ep33 and ep1 scenes'),
 'a100/probes/kaggle_read.py': ('/root/probes/kaggle_read.py', 'reads quota and recent submissions per board, no upload'),
 'a100/probes/kaggle_read2.py': ('/root/probes/kaggle_read2.py', 'reads the recent AUC submission history, no upload'),
 'a100/probes/manifest_read.py': ('/root/probes/manifest_read.py', 'reads manifest r1 from HF'),
 'a100/probes/manifest_read2.py': ('/root/probes/manifest_read2.py', 'prints manifest r1 entries, the ledger and the overlay README'),
 'a100/probes/manifest_r2.py': ('/root/probes/manifest_r2.py', 'reads manifest r2 and fingerprints the ep28 r2 scene'),
 'a100/probes/hf_ken.py': ('/root/probes/hf_ken.py', 'lists what Ken has on HF for ep1, ep5 and ep28'),
 'a100/audit/hold_audit.py': ('/root/probes/hold_audit.py', 'wrist before and after attach on held frames, found the false grasps'),
 'a100/audit/hold_audit2.py': ('/root/probes/hold_audit2.py', 'wrist drag by attach on all frames, per hand'),
 'a100/audit/com_check.py': ('/root/probes/com_check.py', 'file-level diff of the ep28 COM scene and centroid check'),
 'a100/audit/roll_check.py': ('/root/probes/roll_check.py', 'measures what the spoon roll fix changed'),
 'a100/audit/spin_check.py': ('/root/probes/spin_check.py', 'measures cup spin in the plain and fixed ep33 scenes'),
 'a100/audit/scene_diff.py': ('/root/probes/scene_diff.py', 'column-level diff of training scenes against lead_v3'),
 'a100/audit/scene_fp.py': ('/root/probes/scene_fp.py', 'content fingerprint of training scenes'),
 'a100/audit/replay_size.py': ('/root/probes/replay_size.py', 'first replay memory probe'),
 'a100/audit/replay_size2.py': ('/root/probes/replay_size2.py', 'replay memory probe, network shapes'),
 'a100/audit/replay_size3.py': ('/root/probes/replay_size3.py', 'replay memory probe, checkpoint shapes'),
 'a100/audit/replay_size4.py': ('/root/probes/replay_size4.py', 'replay memory probe, critic schema from checkpoint metadata'),
 'a100/tools_track3/pack_submission_handoff.py': ('/root/v2d/NVIDIA_V2D_Egocentric/tools/track3/pack_submission_handoff.py', "Edmond's plan-based packer from the de66fea handoff, unchanged"),
}
R = W
LOCAL = {
 'mac/build_v3.py': (R / 'build_v3.py', 'builds the lead_v3 release'),
 'mac/publish_v3.py': (R / 'publish_v3.py', 'add-only HF upload of v3 with SHA read-back'),
 'mac/build_overlay_r1.py': (R / 'build_overlay_r1.py', 'builds hands overlay r1'),
 'mac/publish_overlay_r1.py': (R / 'publish_overlay_r1.py', 'add-only HF upload of r1'),
 'mac/publish_note_r1.py': (R / 'publish_note_r1.py', 'add-only HF upload of the r1 note'),
 'mac/build_overlay_r2.py': (R / 'build_overlay_r2.py', 'builds hands overlay r2'),
 'mac/publish_r2.py': (R / 'publish_r2.py', 'add-only HF upload of r2'),
 'mac/build_overlay_r3.py': (R / 'build_overlay_r3.py', 'builds hands overlay r3'),
 'mac/publish_r3.py': (R / 'publish_r3.py', 'add-only HF upload of r3'),
 'mac/build_manifest.py': (R / 'manifest/build_manifest.py', 'writes CURRENT_MANIFEST and VERSION_LEDGER'),
}
EXISTING = {'attach_hold.py': 'per-frame hold attach', 'attach_respin.py': 'exclusive attach and cup respin', 'build_scene_hold.py': 'scene build with the hold mode', 'build_scene_spin.py': 'scene build with the rawspin mode', 'com_fix.py': 'audits or applies the URDF inertial origin fix', 'launch_ep.sh': 'launches one A100 training', 'repair_grip_align.py': 'moves the hand to a thin object and aligns the pinch', 'run_hold.sh': 'builds one hold scene', 'run_spin.sh': 'builds one rawspin scene', 'run_train_v3.sh': 'trains one policy on the A100', 'watch_run2.sh': 'evaluates an A100 run as it finishes'}
def pull(ssh, table):
    names = ' '.join(v[0] for v in table.values())
    r = subprocess.run(ssh + [f'tar -cf - --absolute-names {names}'], capture_output=True, timeout=240)
    t = tarfile.open(fileobj=io.BytesIO(r.stdout)); got = {}
    for m in t.getmembers():
        if m.isfile(): got['/' + m.name.lstrip('/')] = t.extractfile(m).read()
    miss = [v[0] for v in table.values() if v[0] not in got]
    assert not miss, miss
    return {k: got[v[0]] for k, v in table.items()}
files = {**pull(SSH6, A6), **pull(SSH1, A1), **{k: Path(v[0]).read_bytes() for k, v in LOCAL.items()}}
BAD = re.compile(rb'KGAT_[0-9a-f]{8}|hf_[A-Za-z0-9]{20}')
ADDR = re.compile(rb'185\.216\.\d+\.\d+|223\.109\.\d+\.\d+|180\.127\.\d+\.\d+')
hits = [k for k, b in files.items() if BAD.search(b)]; assert not hits, hits
addr = [k for k, b in files.items() if ADDR.search(b)]
print('host addresses in', addr)
for k, b in files.items():
    if k in addr: b = ADDR.sub(b'<host>', b); files[k] = b
    p = OUT / k; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b)
what = {**{k: ('A6000', v[0], v[1]) for k, v in A6.items()}, **{k: ('A100', v[0], v[1]) for k, v in A1.items()}, **{k: ('Mac', 'work/lead-v3-release-20261104/' + str(Path(v[0]).relative_to(R)), v[1]) for k, v in LOCAL.items()}}
L = ['# Scripts behind our scenes, training, evaluation, packing and submission', '', 'Copied from the hosts by `work/lead-v3-release-20261104/sync_scripts.py`. Re-run it after any edit. Paths are relative to this directory. Host addresses are replaced by `<host>` in the copies.', '', '| Path | What | Host | Source on host | sha256 |', '|---|---|---|---|---|']
for k, d in sorted(EXISTING.items()):
    L.append(f'| `{k}` | {d} | A100 | see top level | `{hashlib.sha256((OUT / k).read_bytes()).hexdigest()[:12]}` |')
for k, (h, s, d) in sorted(what.items()):
    L.append(f'| `{k}` | {d} | {h} | `{s}` | `{hashlib.sha256((OUT / k).read_bytes()).hexdigest()[:12]}` |')
(OUT / 'SCRIPTS_INDEX.md').write_text('\n'.join(L) + '\n')
print('copied', len(files), 'index rows', len(L) - 6)
