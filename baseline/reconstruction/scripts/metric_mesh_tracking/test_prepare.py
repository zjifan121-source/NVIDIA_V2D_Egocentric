import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import cv2
import prepare
import run


class PackageTest(unittest.TestCase):
    def test_freeze_decode_and_reject_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);payload=p/'payload';payload.mkdir();source=p/'source'
            for n in ('v2d_foundation_pose/lib/foundation_pose_tracker.py','v2d_foundation_pose/lib/FoundationPose/estimater.py'):
                f=source/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('# fixture')
            (payload/'mesh.obj').write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
            (payload/'K.json').write_text(json.dumps(dict(fx=10,fy=10,cx=1,cy=1,width=3,height=2)))
            frames=[]
            for i in range(2):
                row=dict(frame_id=i,rgb=f'rgb{i}.png',depth=f'depth{i}.png',mask=f'mask{i}.png');frames.append(row)
                cv2.imwrite(str(payload/row['rgb']),np.zeros((2,3,3),np.uint8))
                cv2.imwrite(str(payload/row['depth']),np.full((2,3),32768,np.uint16))
                cv2.imwrite(str(payload/row['mask']),np.ones((2,3),np.uint8))
            spec=dict(episode=19,object='mini_sweeper',frame_count=2,anchor_frame=1,
                      metric_mesh_reviewed=True,anchor_selection_reviewed=True,mesh_file='mesh.obj',K_file='K.json',frames=frames)
            (p/'spec.json').write_text(json.dumps(spec))
            (p/'runtime.json').write_text(json.dumps(dict(backend='nvidia_tensorrt',model_registry='fixture',model_version='v',tensorrt_version='10',gpu_compute_capability='8.6',models={})))
            out=prepare.freeze(p/'spec.json',payload,p/'runtime.json',source,p/'package')
            result=run.load_package(out/'manifest.json',out/'protocol.json',payload)
            self.assertEqual(result[4],{'forward':[],'backward':[0]})
            with self.assertRaises(FileExistsError):prepare.freeze(p/'spec.json',payload,p/'runtime.json',source,out)
            (payload/'mesh.obj').write_text('changed')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):run.load_package(out/'manifest.json',out/'protocol.json',payload)

    def test_unreviewed_input_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'spec.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Review metric'):
                prepare.freeze(p/'spec.json',p,p/'missing',p,p/'output')
            self.assertFalse((p/'output').exists())

if __name__=='__main__':unittest.main()
