"""Training-only best-actor warm start; native inference/evaluation stay unchanged.

The older source retains100M critic/learner state but no replay and no60M learner
state. Restore that critic, inject the selected60M actor and its BatchNorm state,
reset the actor optimizer, and start a fresh smaller replay. Subsequent preemption
resumes this trial's complete matching learner/replay pairs without re-injection.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    import jax
    import numpy as np
    from flash_chord.training.flash_sac import runner
    from flash_chord.training.flash_sac.checkpoint import actor_inference_state,load_actor_checkpoint
    from flash_chord.training import checkpoint_metadata as cm
    source=Path(os.environ['V2D_WARM_ACTOR']).resolve()
    base=Path(os.environ['V2D_WARM_BASE_STATE']).resolve()
    source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    original_restore=runner.restore_learner_checkpoint

    def restore(path,template,**kwargs):
        restored,metadata=original_restore(path,template,**kwargs)
        if Path(path).resolve()!=base:return restored,metadata
        actor,actor_metadata=load_actor_checkpoint(source,actor_inference_state(restored))
        cm.validate_policy_schema(cm.checkpoint_policy_schema(actor_metadata),cm.checkpoint_policy_schema(metadata))
        old_config=cm.checkpoint_config(metadata);actor_config=cm.checkpoint_config(actor_metadata)
        # cross-scene warm start: actor from Ken's reference, critic from Wentao's; policy schema still validated
        restored=restored.replace(actor=restored.actor.replace(params=actor.params,batch_stats=actor.batch_stats,
            optimizer_state=template.actor.optimizer_state,schedule_step=template.actor.schedule_step),
            exploration=template.exploration)
        for a,b in zip(jax.tree.leaves(actor.params),jax.tree.leaves(restored.actor.params)):
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
        proof=dict(warm_actor=str(source),warm_actor_sha256=source_sha,critic_state=str(base),
            actor_parameters_exact=True,actor_batch_stats_from_best=True,actor_optimizer='fresh',
            initial_replay='fresh',environment_steps=int(np.asarray(restored.environment_steps)))
        (Path(__file__).resolve().parent/f'{os.environ["NAME"]}_warm_start_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
        print(json.dumps(proof),flush=True)
        return restored,metadata

    runner.restore_learner_checkpoint=restore
    original_build=cm.build_checkpoint_metadata

    def build_metadata(*args,**kwargs):
        metadata=original_build(*args,**kwargs)
        metadata.update(warm_actor_source_sha256=source_sha,warm_actor_source_path=str(source),
                        warm_critic_state_path=str(base),warm_start_initial_replay='fresh',
                        warm_actor_optimizer_reset='true',
                        warm_launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return metadata

    cm.build_checkpoint_metadata=build_metadata
    script=Path(__file__).resolve().parents[2]/'baseline/robotic_grounding/flash_chord/scripts/train_flash_sac.py'
    sys.argv[0]=str(script)
    runpy.run_path(str(script),run_name='__main__')


if __name__=='__main__':main()
