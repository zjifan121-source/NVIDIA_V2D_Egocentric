"""Install validated slot-preserving convex pieces in copied scene URDFs."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import numpy as np,trimesh
P=Path('reports/submission_candidate_2026-10-28').resolve();S=P/'ep25_assembly2_scene';reports=[]
for i in [1,2]:
 z=np.load(P/f'ep25_board{i}_off0.020_hulls.npz');urdf=next(S.glob(f'assets/human_motion_data/ego_recon/processed/*wooden_piece_{i}_rigid.urdf'));tree=ET.parse(urdf);link=tree.getroot().find('link');old=link.findall('collision');assert len(old)==1
 before_visual=ET.tostring(link.find('visual'));before_inertial=ET.tostring(link.find('inertial'));link.remove(old[0]);out=urdf.parent/'slot_collision';out.mkdir(exist_ok=True)
 for j in range(int(z['n'])):
  m=trimesh.Trimesh(z[f'v{j}'],z[f'f{j}'],process=True);assert m.is_watertight
  assert abs(m.convex_hull.volume-m.volume)<max(1e-12,m.volume*1e-4)
  name=f'board{i}_hull{j:03d}.stl';m.export(out/name);c=ET.SubElement(link,'collision',name=f'board{i}_slot_hull{j:03d}');ET.SubElement(c,'origin',xyz='0 0 0',rpy='0 0 0');g=ET.SubElement(c,'geometry');ET.SubElement(g,'mesh',filename=f'./slot_collision/{name}',scale='1 1 1')
 assert ET.tostring(link.find('visual'))==before_visual and ET.tostring(link.find('inertial'))==before_inertial
 ET.indent(tree,space='  ');tree.write(urdf,encoding='utf-8',xml_declaration=True);reports.append(dict(board=i,hulls=int(z['n']),urdf=str(urdf),method='CoACD threshold0.02, preprocessoff, original watertight scoring mesh; each emitted piece convex',scoring_mesh_unchanged=True))
(S/'collision_repair.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports,indent=2))
