"""Frozen-scene policy diagnostics; outputs are never submission candidates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PLAN = REPO / 'reports/submission_candidate_2026-10-28/plan.json'


def reference(ep):
    import pyarrow.parquet as pq
    import trimesh
    from scipy.spatial.transform import Rotation
    entry = json.loads(PLAN.read_text())[str(ep)]
    result = json.loads((Path(entry['run']) / 'results' / f'episode_{ep:06d}.json').read_text())
    # Imported rollout JSONs can retain another user's private build path.
    # The selected scene is the relocated copy used by our pipeline.
    partition = next((Path(entry['scene']) / 'assets/human_motion_data/ego_recon/processed').glob(
        'sequence_id=*/robot_name=sharpa_wave/*.parquet'))
    parquet = partition if partition.is_file() else next(partition.glob('*.parquet'))
    table = pq.read_table(parquet)
    cell = lambda k: np.asarray(table[k][0].as_py())
    pos, quat = cell('object_body_position'), cell('object_body_wxyz')
    n = len(pos)
    frames = sorted(set(range(0, n, 10)) | {n - 1})
    rows = []
    for slot, name in enumerate(table['object_body_names'][0].as_py()):
        # Scoring meshes share the object's canonical body coordinates.
        meshdir = Path(entry['scene']) / 'score_input/meshes' / f'episode_{ep:06d}'
        candidates = [p for p in meshdir.glob('*.obj') if name.endswith('_' + p.stem)]
        if not candidates and len(list(meshdir.glob('*.obj'))) == 1:
            candidates = list(meshdir.glob('*.obj'))  # single-object scene: the body is named after the sequence
        assert len(candidates) == 1, (name, candidates)
        mesh = trimesh.load(candidates[0], force='mesh', process=False)
        rot = Rotation.from_quat(quat[:, slot, [1, 2, 3, 0]]).as_matrix()
        for side in ('left', 'right'):
            names = table[f'{side}_robot_frame_names'][0].as_py()
            ids = [names.index(f'{side}_{d}_tip_site') for d in ('thumb', 'index', 'middle')]
            tips = cell(f'robot_{side}_frames')[:, ids, :3]
            wrist = cell(f'robot_{side}_wrist_position')
            for f in frames:
                points = np.concatenate([tips[f], wrist[f, None]])
                local = (points - pos[f, slot]) @ rot[f]
                _, dist, _ = trimesh.proximity.closest_point(mesh, local)
                rows.append(dict(object=name, side=side, frame=f, tip_surface_gap_cm=(100*dist[:3]).tolist(),
                                 wrist_surface_gap_cm=float(100*dist[3]), object_z=float(pos[f, slot, 2]),
                                 wrist_object_local_m=local[3].tolist()))
    output = dict(episode=ep, entry=entry, checkpoint=result['checkpoint'], parquet=str(parquet),
                  original_reported_parquet=result['parquet'],
                  parquet_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest(), columns=table.column_names,
                  metrics=result['metrics'], geometry=rows)
    for key in table.column_names:
        if 'contact' in key:
            value=table[key][0].as_py()
            output.setdefault('contact_columns', {})[key] = value
    (HERE/f'ep{ep}_reference.json').write_text(json.dumps(output, indent=2)+'\n')
    print(f'ep{ep}: reference written {parquet}', flush=True)


def trace(ep, mode, limit):
    import jax
    import jax.numpy as jnp
    import warp as wp
    from omegaconf import OmegaConf
    from flash_chord.configuration import instantiate_typed
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    from flash_chord.training.checkpoint_metadata import checkpoint_config
    from flash_chord.training.flash_sac.config import TrainingConfig
    from flash_chord.training.flash_sac.evaluation import load_policy_for_inference
    from flash_chord.training.flash_sac.runner import resolve_jax_device
    from flash_chord.evaluation.harness import apply_evaluation_composition, _evaluation_curriculum_stage
    from flash_chord.scene.setup import setup_scene
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.rl import RLEnv, RLEnvConfig
    from flash_chord.training.environment import WarpRLEnv
    entry=json.loads(PLAN.read_text())[str(ep)]
    result=json.loads((Path(entry['run'])/'results'/f'episode_{ep:06d}.json').read_text())
    checkpoint=Path(result['checkpoint'])
    metadata=read_checkpoint_metadata(checkpoint)
    config=checkpoint_config(metadata)
    apply_evaluation_composition(config,world_count=1,reset_mode='explicit')
    cfg=OmegaConf.create(config)
    ec=instantiate_typed(cfg.env,RLEnvConfig)
    tc=instantiate_typed(cfg.training,TrainingConfig)
    setup=setup_scene(parquet=cfg.task.parquet,control_fps=ec.sim.fps,motion_speed=cfg.task.motion_speed,
        embodiment=instantiate_typed(cfg.embodiment,Embodiment),collision=instantiate_typed(cfg.collision,CollisionPolicy),
        world_count=1,include_support=cfg.scene.support,decompose_objects=cfg.scene.decompose_objects,
        object_scale_min=cfg.scene.get('object_scale_min',1.),object_scale_max=cfg.scene.get('object_scale_max',1.),
        object_scale_seed=cfg.scene.get('object_scale_seed',0),contact_friction=cfg.scene.get('contact_friction',1.),
        object_free_joint_damping=cfg.scene.get('object_free_joint_damping',0.),
        source_frame_playback=bool(cfg.task.get('source_frame_playback',False)),
        motion_start_frame=int(cfg.task.get('motion_start_frame',0)),motion_end_frame=int(cfg.task.get('motion_end_frame',-1)))
    env=RLEnv(setup.scene,setup.reference,config=ec)
    env.reset();env.capture_step()
    wrapper=WarpRLEnv(env,publish_diagnostics=True,publish_step_metrics=True)
    wrapper.set_curriculum(_evaluation_curriculum_stage(tc.curriculum.stages[int(metadata['curriculum_stage'])],'explicit'))
    env.reset(frame_id=0)
    device=resolve_jax_device(tc.device)
    obs=jax.device_put(jnp.asarray(env.observation.numpy().reshape(1,env.observation_dim)),device)
    if mode=='trained':
        actor,policy,key,_=load_policy_for_inference(str(checkpoint),tc,obs,env.action.action_dim,deterministic=True,device=device)
    arrays={k:[] for k in ('body_q','joint_q','joint_target','timestep','action','contact_active','contact_force_w')}
    for i in range(min(limit,setup.reference.num_frames)):
        for k,v in [('body_q',env.state_0.body_q),('joint_q',env.state_0.joint_q),
                    ('joint_target',env.action.joint_target),('timestep',env.timestep)]:
            arrays[k].append(v.numpy().copy())
        if env.contact is not None:
            arrays['contact_active'].append(env.contact.contact_active.numpy().copy())
            arrays['contact_force_w'].append(env.contact.contact_force_w.numpy().copy())
        if mode=='trained':
            action,key=policy(actor,obs,key)
        else:
            action=jnp.zeros((1,env.action.action_dim),dtype=jnp.float32)
        arrays['action'].append(np.asarray(action))
        obs=wrapper.step(action).observation
    model=setup.scene.model
    arrays={k:np.asarray(v) for k,v in arrays.items()}
    arrays.update(object_body_ids=env.command.body_ids_w.numpy(),joint_q_start=model.joint_q_start.numpy())
    info=dict(episode=ep,mode=mode,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              reference_parquet=str(cfg.task.parquet),body_labels=list(model.body_label),joint_labels=list(model.joint_label),
              frames=len(arrays['body_q']),diagnostic_only=True,config=config)
    if env.contact is not None:
        arrays['slot_hand_ids']=env.contact.slot_hand_ids.numpy()
        arrays['slot_object_ids']=env.contact.slot_object_ids.numpy()
        info['contact_layout']=str(env.contact.layout)
    np.savez_compressed(HERE/f'ep{ep}_{mode}_trace.npz',**arrays)
    (HERE/f'ep{ep}_{mode}_trace.json').write_text(json.dumps(info,indent=2,default=str)+'\n')
    print(f'ep{ep} {mode}: {info["frames"]} frames recorded',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['reference','trained','nominal'])
    p.add_argument('--episode',type=int,required=True)
    p.add_argument('--frames',type=int,default=220)
    a=p.parse_args()
    if a.mode=='reference':reference(a.episode)
    else:trace(a.episode,a.mode,a.frames)
