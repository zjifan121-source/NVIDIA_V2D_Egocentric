"""Record a successful trained spoon policy for geometry-based grasp reuse; diagnostic only."""
from pathlib import Path
import json,numpy as np,warp as wp,jax,jax.numpy as jnp
from flash_chord.training.flash_sac.evaluation import load_policy_for_inference
from flash_chord.training.flash_sac.runner import resolve_jax_device
from omegaconf import OmegaConf
from flash_chord.configuration import instantiate_typed
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.evaluation.harness import apply_evaluation_composition,_evaluation_curriculum_stage
from flash_chord.scene.setup import setup_scene
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv,RLEnvConfig
from flash_chord.training.environment import WarpRLEnv
P=Path('/home/edmondwa/NVIDIA_V2D_Egocentric/reports/submission_candidate_2026-10-28');checkpoint=Path('/data/user_data/edmondwa/v2d/flash_runs/diag/ep37_v5_obj2x_s43/train/policy_100003840.safetensors');metadata=read_checkpoint_metadata(checkpoint);config=checkpoint_config(metadata);apply_evaluation_composition(config,world_count=1,reset_mode='explicit');cfg=OmegaConf.create(config);env_config=instantiate_typed(cfg.env,RLEnvConfig);training=instantiate_typed(cfg.training,TrainingConfig)
setup=setup_scene(parquet=cfg.task.parquet,control_fps=env_config.sim.fps,motion_speed=cfg.task.motion_speed,embodiment=instantiate_typed(cfg.embodiment,Embodiment),collision=instantiate_typed(cfg.collision,CollisionPolicy),world_count=1,include_support=cfg.scene.support,decompose_objects=cfg.scene.decompose_objects,object_scale_min=cfg.scene.get('object_scale_min',1.),object_scale_max=cfg.scene.get('object_scale_max',1.),object_scale_seed=cfg.scene.get('object_scale_seed',0),contact_friction=cfg.scene.get('contact_friction',1.),object_free_joint_damping=cfg.scene.get('object_free_joint_damping',0.),source_frame_playback=bool(cfg.task.get('source_frame_playback',False)),motion_start_frame=int(cfg.task.get('motion_start_frame',0)),motion_end_frame=int(cfg.task.get('motion_end_frame',-1)))
env=RLEnv(setup.scene,setup.reference,config=env_config);env.reset();env.capture_step();jax_env=WarpRLEnv(env,publish_diagnostics=True,publish_step_metrics=True);jax_env.set_curriculum(_evaluation_curriculum_stage(training.curriculum.stages[int(metadata['curriculum_stage'])],'explicit'));env.reset(frame_id=0);device=resolve_jax_device(training.device);observation=jax.device_put(jnp.asarray(env.observation.numpy().reshape(env.world_count,env.observation_dim)),device);actor,policy,key,_=load_policy_for_inference(str(checkpoint),training,observation,env.action.action_dim,deterministic=True,device=device);body=[];joints=[];targets=[];times=[];actions=[]
for i in range(min(200,setup.reference.num_frames)):
 body.append(env.state_0.body_q.numpy().copy());joints.append(env.state_0.joint_q.numpy().copy());targets.append(env.action.joint_target.numpy().copy());times.append(env.timestep.numpy().copy());action,key=policy(actor,observation,key);actions.append(np.asarray(action));observation=jax_env.step(action).observation
model=setup.scene.model
np.savez(P/'ep37_successful_grasp_trace.npz',body_q=np.asarray(body),joint_q=np.asarray(joints),joint_target=np.asarray(targets),timestep=np.asarray(times),actions=np.asarray(actions),object_body_ids=env.command.body_ids_w.numpy(),joint_q_start=model.joint_q_start.numpy())
(P/'ep37_successful_grasp_trace.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),body_labels=list(model.body_label),joint_labels=list(model.joint_label),frames=len(body),diagnostic_only=True,reference_parquet=str(cfg.task.parquet)),indent=2)+'\n');print('Recorded successful spoon policy physics frames',flush=True)
