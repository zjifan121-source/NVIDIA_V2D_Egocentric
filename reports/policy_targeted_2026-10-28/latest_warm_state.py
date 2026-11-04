"""Select only complete states from this tagged warm start; replay restarts fresh."""
import hashlib
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'submission_candidate_2026-10-28'))
from checkpoint_pairs import metadata

expected=hashlib.sha256(Path(os.environ['V2D_WARM_ACTOR']).read_bytes()).hexdigest()
valid=[]
for state in Path(sys.argv[1]).glob('state_*.safetensors'):
    try:
        md=metadata(state)
        if md.get('user.warm_actor_source_sha256')==expected:valid.append((int(state.stem.split('_')[1]),state))
    except (OSError,ValueError,KeyError):continue
if valid:print(max(valid)[1])
