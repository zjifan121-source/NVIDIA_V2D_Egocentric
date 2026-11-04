"""Write ep32's two table cylinders (Wentao's support spec, bundle world) into our scene's frame.

Our scene is levelled/cleaned, so map with the frame-0 pose of object 0 only and report the
residual tilt instead of refusing. Usage: python ep32_support.py SPEC BUNDLE PARQUET OUT
"""
import json,sys
from pathlib import Path
import numpy as np,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
spec,bundle,parquet,out=map(Path,sys.argv[1:5])
d=json.loads(spec.read_text()); rec={k:v[0] for k,v in pq.read_table(parquet).to_pydict().items()}
with np.load(bundle/'result.npz') as raw: src=raw['object_to_world_transform'][0].astype(float)
p=np.asarray(rec['object_body_position'],float)[0,0]; q=np.asarray(rec['object_body_wxyz'],float)[0,0]
tgt=np.eye(4); tgt[:3,3]=p; tgt[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
T=tgt@np.linalg.inv(src)
tilt=np.degrees(np.arccos(np.clip((T[:3,:3]@[0,0,1])[2],-1,1)))
from pxr import Gf,Usd,UsdGeom,UsdPhysics
assert not out.exists(),out
st=Usd.Stage.CreateNew(str(out)); UsdGeom.SetStageUpAxis(st,UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st,1.0)
for i,c in enumerate(d['cylinders']):
  ctr=T@np.r_[c['center_bundle_world_m'],1.0]
  g=UsdGeom.Cylinder.Define(st,f'/support_{i}'); g.CreateAxisAttr('Z'); g.CreateRadiusAttr(c['radius_m']); g.CreateHeightAttr(c['height_m'])
  UsdGeom.Xformable(g).AddTranslateOp().Set(Gf.Vec3d(*ctr[:3])); UsdPhysics.CollisionAPI.Apply(g.GetPrim())
  print('cylinder',i,'center',np.round(ctr[:3],3),'r',round(c['radius_m'],3))
st.GetRootLayer().Save(); print('tilt of mapped Z axis %.2f deg'%tilt, '->', out)
