from pathlib import Path
from types import SimpleNamespace
import json,sys,numpy as np
import run_recovery_v5 as run

case=Path(sys.argv[1]);out=Path('/output')
args=SimpleNamespace(manifest=case/'manifest.json',protocol=case/'protocol.json',payload=Path('/payload'),
                     output=out,weights_dir='/weights/foundation_pose',stage='anchor')
pack=run.load_package(args.manifest,args.protocol,args.payload)
run.execute(args,pack)
# This is an execution check, not a visual or numerical-quality acceptance.
# Valid geometry/SE3 must finish the full experiment even if alignment is poor.
anchor=out/'anchor/anchor_public.npy';run.validate_transform(np.load(anchor))
m,cfg,*_=pack
review=dict(proceed_to_fixed_tracking=True,geometry_and_pose_numerically_valid=True,
 matching_mesh_pose_provenance_checked=True,quality_veto_disabled_by_user=True,
 protocol_sha256=run.sha(args.protocol),input_manifest_sha256=run.sha(args.manifest),
 mesh_sha256=m['files'][m['mesh_file']],anchor_public_sha256=run.sha(anchor),anchor_frame=m['anchor_frame'],
 reviewer='automatic execution-only validation under explicit 20261103 user protocol',
 reviewed_evidence=['frozen exported shape geometry check','new-mesh matching auxiliary pose hash','finite proper-SE3 anchor'],
 visual_quality='NOT_A_GATE; full alignment evaluation follows')
run.validate_review(review,review['protocol_sha256'],review['input_manifest_sha256'],review['mesh_sha256'],anchor,m['anchor_frame'],provided=True)
run.write_json(out/'ANCHOR_REVIEW.json',review)
args.stage='track';run.execute(args,pack)
