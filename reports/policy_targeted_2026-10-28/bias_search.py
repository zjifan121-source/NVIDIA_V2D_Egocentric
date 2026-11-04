"""Per-episode policy-parameter search, not best-of-world trajectory submission.

Each search world runs a different CONSTANT mean-head bias for the whole episode.
Only right-hand output biases vary. Save ordinary FlashSAC checkpoints and validate
the best candidates separately with the untouched official single-policy evaluator.
Uses reconstructed references only; never hidden/public ground truth or Kaggle scores.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]


def main(ep,population,rounds):
    import jax
    import jax.numpy as jnp
    from flax.core import FrozenDict
    from omegaconf import OmegaConf
    from flash_chord.configuration import instantiate_typed
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    from flash_chord.training.checkpoint_metadata import checkpoint_config
    from flash_chord.training.flash_sac.config import TrainingConfig
    from flash_chord.training.flash_sac.checkpoint import load_actor_checkpoint,save_actor_checkpoint
    from flash_chord.training.flash_sac.evaluation import create_actor_template,compile_policy_action
    from flash_chord.training.flash_sac.network import deterministic_action
    from flash_chord.training.flash_sac.runner import resolve_jax_device
    from flash_chord.evaluation.harness import apply_evaluation_composition,_evaluation_curriculum_stage
    from flash_chord.evaluation.rollout import record_object_pose_rollout
    from flash_chord.evaluation.metrics import object_add,ADD_THRESHOLDS_M
    from flash_chord.scene.setup import setup_scene
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.rl import RLEnv,RLEnvConfig
    from flash_chord.training.environment import WarpRLEnv
    assert population>=16 and rounds>=1
    plan=json.loads((HERE/'baseline_plan.json').read_text());entry=plan[str(ep)]
    result=json.loads((Path(entry['run'])/'results'/f'episode_{ep:06d}.json').read_text())
    source=Path(result['checkpoint']);metadata=read_checkpoint_metadata(source)
    cfgdict=checkpoint_config(metadata);apply_evaluation_composition(cfgdict,world_count=population,reset_mode='explicit')
    cfg=OmegaConf.create(cfgdict);ec=instantiate_typed(cfg.env,RLEnvConfig);tc=instantiate_typed(cfg.training,TrainingConfig)
    setup=setup_scene(parquet=cfg.task.parquet,control_fps=ec.sim.fps,motion_speed=cfg.task.motion_speed,
        embodiment=instantiate_typed(cfg.embodiment,Embodiment),collision=instantiate_typed(cfg.collision,CollisionPolicy),
        world_count=population,include_support=cfg.scene.support,decompose_objects=cfg.scene.decompose_objects,
        object_scale_min=cfg.scene.get('object_scale_min',1.),object_scale_max=cfg.scene.get('object_scale_max',1.),
        object_scale_seed=cfg.scene.get('object_scale_seed',0),contact_friction=cfg.scene.get('contact_friction',1.),
        object_free_joint_damping=cfg.scene.get('object_free_joint_damping',0.),
        source_frame_playback=bool(cfg.task.get('source_frame_playback',False)),
        motion_start_frame=int(cfg.task.get('motion_start_frame',0)),motion_end_frame=int(cfg.task.get('motion_end_frame',-1)))
    env=RLEnv(setup.scene,setup.reference,config=ec);env.reset();env.capture_step()
    wrapper=WarpRLEnv(env,publish_diagnostics=True,publish_step_metrics=True)
    wrapper.set_curriculum(_evaluation_curriculum_stage(tc.curriculum.stages[int(metadata['curriculum_stage'])],'explicit'))
    device=resolve_jax_device(tc.device);dimension=env.action.action_dim
    model,template=create_actor_template(tc,env.observation_dim,dimension,device=device)
    actor,_=load_actor_checkpoint(source,template)
    layout=env.action.buffers.layout;hand_index=layout.sides.index('right');start=layout.hand_action_starts[hand_index]
    hand=setup.scene.layout.hand('right');indices=list(range(start,start+6));labels=['wrist_x','wrist_y','wrist_z','wrist_rx','wrist_ry','wrist_rz']
    for i,name in enumerate(hand.finger_joint_names):
        if 'thumb' in name or ('index' in name and not name.endswith('_AA')):
            indices.append(start+6+i);labels.append(name)
    right_end=start+6+layout.finger_counts[hand_index]
    assert start==0 and right_end==28 and dimension==56,(start,right_end,dimension)

    def with_bias(state,bias):
        value=state.params['mean_bias']+bias
        params=state.params.copy({'mean_bias':value}) if isinstance(state.params,FrozenDict) else {**state.params,'mean_bias':value}
        return state.replace(params=params)

    def row_action(state,observation,bias):
        changed=with_bias(state,bias)
        distribution=model.apply({'params':changed.params,'batch_stats':changed.batch_stats},observation[None,:],training=False)
        return deterministic_action(distribution,action_transform=tc.network.action_transform,action_scale=tc.network.action_scale)[0]

    @jax.jit
    def population_action(state,observation,key,bias):
        return jax.vmap(row_action,in_axes=(None,0,0))(state,observation,bias),key

    env.reset(frame_id=0)
    obs=jax.device_put(jnp.asarray(env.observation.numpy().reshape(population,env.observation_dim)),device)
    key=jax.device_put(jax.random.PRNGKey(42),device);zeros=jnp.zeros((population,dimension),jnp.float32)
    standard=compile_policy_action(model,actor,obs,key,deterministic=True,
                                  action_transform=tc.network.action_transform,action_scale=tc.network.action_scale)
    actual,_=population_action(actor,obs,key,zeros);expected,_=standard(actor,obs,key)
    action_difference=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    if action_difference>1e-3:raise ValueError(f'Zero-bias actor adapter mismatch {action_difference}')
    out=HERE/f'ep{ep}_bias_search';out.mkdir(exist_ok=True)
    rng=np.random.default_rng(831);center=np.zeros(len(indices));std=np.r_[np.full(3,.30),np.full(3,.20),np.full(len(indices)-6,.40)]
    best_bias=np.zeros(len(indices));best_score=-1.;saved=[];history=[]
    for iteration in range(rounds):
        candidates=np.clip(rng.normal(center,std,size=(population,len(indices))),-1.25,1.25).astype(np.float32)
        candidates[0]=0;candidates[1]=best_bias
        full=np.zeros((population,dimension),np.float32);full[:,indices]=candidates
        bias_device=jax.device_put(jnp.asarray(full),device)
        callback=lambda state,observation,current_key:population_action(state,observation,current_key,bias_device)
        env.reset(frame_id=0)
        obs=jax.device_put(jnp.asarray(env.observation.numpy().reshape(population,env.observation_dim)),device)
        rollout=record_object_pose_rollout(env,wrapper,callback,actor,key,obs,reference=setup.reference,
            step_count=setup.reference.num_frames,start_frame=0,progress_interval=0)
        # Deterministic 128-point approximation for search only; official validation uses its normal metric.
        points=rollout.object_vertices_o[:,::max(1,rollout.object_vertices_o.shape[1]//128)]
        add=object_add(rollout.achieved_pose_w,rollout.reference_pose_w,points,world_chunk=8)
        accuracy=np.stack([(add<t).mean(axis=0) for t in ADD_THRESHOLDS_M])
        per_body=np.trapezoid(accuracy,x=np.linspace(0,1,len(ADD_THRESHOLDS_M)),axis=0)
        scores=per_body.mean(axis=1)
        allowed=per_body[:,0]>=per_body[0,0]-.03
        for invalid in rollout.non_finite_worlds:allowed[invalid]=False
        fitness=np.where(allowed,scores,-1.)
        elites=np.argsort(fitness)[-max(4,population//8):];elites=elites[fitness[elites]>=0]
        if not len(elites):raise ValueError('No finite feasible search candidates')
        winner=int(np.argmax(fitness))
        if fitness[winner]>best_score:
            best_score=float(fitness[winner]);best_bias=candidates[winner].copy()
            vector=np.zeros(dimension,np.float32);vector[indices]=best_bias
            changed=with_bias(actor,jax.device_put(jnp.asarray(vector),device))
            np.testing.assert_array_equal(np.asarray(changed.params['mean_bias'])[28:],np.asarray(actor.params['mean_bias'])[28:])
            policy=out/f'policy_round{iteration}.safetensors'
            md={**metadata,'policy_parameter_search':'right_mean_bias_only',
                'parameter_search_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'parameter_search_indices_json':json.dumps(indices),'parameter_search_bias_json':json.dumps(best_bias.tolist()),
                'parameter_search_rollouts':str((iteration+1)*population),'parameter_search_full_frame_coverage':'true'}
            save_actor_checkpoint(policy,changed,metadata=md)
            saved.append(dict(checkpoint=str(policy),search_auc=best_score,bias=best_bias.tolist(),round=iteration))
        center=.2*center+.8*candidates[elites].mean(0)
        std=np.maximum(.04,.5*std+.5*candidates[elites].std(0))
        row=dict(round=iteration,control_auc=float(scores[0]),control_bodies=per_body[0].tolist(),
                 best_auc=float(scores[winner]),best_bodies=per_body[winner].tolist(),best_bias=best_bias.tolist())
        history.append(row);print(json.dumps(row),flush=True)
        (out/'search.json').write_text(json.dumps(dict(source=str(source),parameter_names=labels,indices=indices,
            population=population,rounds=rounds,adapter_zero_bias_max_difference=action_difference,
            object_references_and_physics_unchanged=True,left_hand_parameters_unchanged=True,
            history=history,candidates=saved),indent=2)+'\n')
    # Each finalist is a single ordinary checkpoint, evaluated alone. Never submit the population rollout.
    finalists=[dict(checkpoint=str(source),search_auc=None)]+sorted(saved,key=lambda r:r['search_auc'],reverse=True)[:3]
    validations=[]
    for i,candidate in enumerate(finalists):
        directory=out/f'official_{i}';directory.mkdir(exist_ok=True)
        checkpoints=directory/'checkpoints.json';checkpoints.write_text(json.dumps({str(ep):candidate['checkpoint']})+'\n')
        with (directory/'eval.log').open('a') as log:
            subprocess.run([sys.executable,str(REPO/'assets/data/v2d_submission_kit/examples/run_policy_evaluation.py'),
                '--checkpoints',str(checkpoints),'--results',str(directory/'results'),
                '--runtime-root',str(REPO/'baseline/robotic_grounding/flash_chord')],
                cwd=REPO/'baseline/robotic_grounding/flash_chord',stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((directory/'results'/f'episode_{ep:06d}.json').read_text())
        validations.append(dict(**candidate,official_auc=result['metrics']['add_auc'],
            per_body=result['metrics']['add_auc_per_body'],run=str(directory),step_count=result['step_count']))
        print('OFFICIAL',json.dumps(validations[-1]),flush=True)
        (out/'official_results.json').write_text(json.dumps(validations,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=int,default=5)
    p.add_argument('--population',type=int,default=96);p.add_argument('--rounds',type=int,default=4)
    a=p.parse_args();main(a.episode,a.population,a.rounds)
