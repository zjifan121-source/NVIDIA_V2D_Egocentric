"""CPU reference audit using exact triangle distances without the optional rtree package."""
import io
import json
import subprocess
from pathlib import Path

import trimesh
from PIL import Image, ImageDraw

import diagnose

# Only four query points per frame; exhaustive triangle distance is inexpensive here.
trimesh.proximity.closest_point=trimesh.proximity.closest_point_naive
episodes={e['episode']:e for e in json.loads((diagnose.REPO/'tools/track3/episodes.json').read_text())['episodes']}
for ep in (5,19,30):
    diagnose.reference(ep)
    video=diagnose.REPO/episodes[ep]['video']
    canvas=Image.new('RGB',(1200,900),'white')
    draw=ImageDraw.Draw(canvas)
    for i,frame in enumerate((0,20,40,60,100,180)):
        raw=subprocess.check_output(['ffmpeg','-loglevel','error','-i',str(video),'-vf',f'select=eq(n\\,{frame})',
                                     '-frames:v','1','-f','image2pipe','-vcodec','png','-'])
        im=Image.open(io.BytesIO(raw)).convert('RGB');im.thumbnail((600,280))
        x,y=(i%2)*600,(i//2)*300
        canvas.paste(im,(x,y+20));draw.text((x+4,y+3),f'episode {ep} frame {frame}',fill='black')
    canvas.save(diagnose.HERE/f'ep{ep}_video_sheet.jpg',quality=90)
    print(f'ep{ep}: video contact sheet saved',flush=True)
