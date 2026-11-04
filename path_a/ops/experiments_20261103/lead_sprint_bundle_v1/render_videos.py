"""Full-frame before/after overlay videos for episodes whose object tracks change in the lead sprint release.
Left panel: fb11 (previous HF) objects; right panel: lead sprint candidate objects. Both objects of the scene are
drawn (slot 0 orange-blue, slot 1 blue-orange as in the fb11 videos); green outline = observed mask of the changed
object(s). RGB shows the real observed hands; no reconstructed hands, no physics. Mac CPU, no GPU.
Usage: render_videos.py <episode> [<episode> ...]   (writes videos/episode_XXXXXX/{before_after,selected_overlay}.mp4)"""
from pathlib import Path
import json, sys, subprocess, hashlib, time, shutil
import numpy as np, cv2, trimesh
ROOT=Path('/Volumes/E/Code/V2D'); OPS=ROOT/'NVIDIA_V2D_Egocentric/path_a/ops'
D=OPS/'experiments_20261029/scored_scene_consistency_v1'; sys.path.insert(0,str(D/'ep5_observed_geometry_v1'))
from surface_renderer import raster_surface
O=Path(__file__).resolve().parent; B=OPS/'experiments_20261030/scored20_dual_anchor_v1'; REL=ROOT/'work/scored11-fulltrack-20261103/release'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
fps=20.254119873046875  # same native consumer FPS as the fb11 videos
summary=json.loads((O/'SELECTION_SUMMARY.json').read_text())
def load(bundle,slot):
    z=np.load(bundle/'result.npz',allow_pickle=False); k='object_to_world_transform' if slot==0 else f'object_{slot}_to_world_transform'
    T=np.array([np.linalg.solve(c.astype(float),w.astype(float)) for c,w in zip(z['camera_to_world_transform'],z[k])])
    m=trimesh.load(bundle/('mesh.obj' if slot==0 else f'mesh_{slot}.obj'),force='mesh',process=False)
    return T,np.asarray(m.vertices,float),np.asarray(m.faces)
def encoder(path,w,h):
    return subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
for ep in [int(a) for a in sys.argv[1:]]:
    e=summary['episodes'][str(ep)]; objects=e['objects']; changed=[c['object'] for c in e['changed']]
    mans={obj:json.loads((B/'cases'/f'ep{ep:02d}_{obj}'/'INPUT_MANIFEST.json').read_text()) for obj in objects}
    man=mans[objects[0]]; cache=Path(man['payload_local']); K=json.loads((cache/'K.json').read_text())
    Km=np.array([[K['fx'],0,K['cx']],[0,K['fy'],K['cy']],[0,0,1.]]); shape=(K['height'],K['width'])
    before={s:load(REL/f'scenes/episode_{ep:06d}/bundle',s) for s in range(len(objects))}
    after={s:load(O/f'scenes/episode_{ep:06d}/bundle',s) for s in range(len(objects))}
    n=len(man['frames']); assert all(len(t[0])==n for t in list(before.values())+list(after.values())),(ep,n)
    out=O/'videos'/f'episode_{ep:06d}'
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    h,w=380,508; pair=encoder(out/'before_after.mp4',w*2,h+40); sel=encoder(out/'selected_overlay.mp4',w,h+40)
    colors={0:(0,190,255),1:(255,150,30)}; tick=time.monotonic()
    for f in range(n):
        rgb=cv2.imread(str(cache/man['frames'][f]['rgb'])); assert rgb is not None
        panels=[]
        for label,src in [('PREVIOUS HF fb11',before),('LEAD SPRINT v1 CANDIDATE',after)]:
            im=rgb.copy()
            for s,(T,V,F) in src.items():
                dep=raster_surface(V@T[f][:3,:3].T+T[f][:3,3],F,Km,shape)[0]; mask=dep>0; col=colors[s]
                im[mask]=(im[mask]*.58+np.array(col)*.42).astype(np.uint8)
                cs,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(im,cs,-1,col,1)
            for obj in changed:
                mk=cv2.imread(str(Path(mans[obj]['payload_local'])/mans[obj]['frames'][f]['mask']),0)>0
                cs,_=cv2.findContours(mk.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(im,cs,-1,(50,240,40),1)
            im=cv2.resize(im,(w,h),interpolation=cv2.INTER_AREA); canvas=np.zeros((h+40,w,3),np.uint8); canvas[40:]=im
            cv2.putText(canvas,f'ep{ep} | {label} | frame {f}',(8,17),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(canvas,'Objects only; green = observed mask of changed object; RGB hands are observed. Not physics.',(8,33),cv2.FONT_HERSHEY_SIMPLEX,.33,(180,180,180),1,cv2.LINE_AA)
            panels.append(canvas)
        pair.stdin.write(np.concatenate(panels,axis=1).tobytes()); sel.stdin.write(panels[1].tobytes())
    for p in [pair,sel]: p.stdin.close(); assert p.wait(timeout=180)==0
    files={}
    for p in sorted(out.glob('*.mp4')):
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(p)],text=True))['streams'][0]
        assert int(probe['nb_read_frames'])==n,(p,probe['nb_read_frames'],n)
        files[p.name]=dict(sha256=sha(p),frames=n,width=probe['width'],height=probe['height'],duration=float(probe['duration']))
    (out/'VIDEO_RECEIPT.json').write_text(json.dumps(dict(episode=ep,state='FULL_FRAME_H264_VERIFIED',fps=fps,frames=n,seconds=time.monotonic()-tick,changed_objects=changed,
        left='fb11 previous HF objects',right='lead sprint v1 candidate objects',
        overlay='All object meshes; green = observed mask of changed object(s); RGB shows observed hands; no reconstructed hands; no physics',files=files),indent=2)+'\n')
    print('VIDEO',ep,n,round(time.monotonic()-tick,1),flush=True)
