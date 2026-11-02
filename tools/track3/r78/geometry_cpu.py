"""Low-resolution triangle depth for local metrics, integer OpenCV pixels."""
import numpy as np


def raster(vertices, faces, K, shape):
    h,w=shape;zbuf=np.full((h,w),np.inf)
    p=vertices@K.T
    uv=p[:,:2]/p[:,2,None]
    for face in faces:
        xyz=vertices[face]
        if np.any(xyz[:,2]<=1e-5):continue
        q=uv[face];z=xyz[:,2]
        low=np.maximum(np.ceil(q.min(0)).astype(int),0)
        high=np.minimum(np.floor(q.max(0)).astype(int),[w-1,h-1])
        if np.any(low>high):continue
        den=(q[1,1]-q[2,1])*(q[0,0]-q[2,0])+(q[2,0]-q[1,0])*(q[0,1]-q[2,1])
        if abs(den)<1e-12:continue
        y,x=np.mgrid[low[1]:high[1]+1,low[0]:high[0]+1]
        a=((q[1,1]-q[2,1])*(x-q[2,0])+(q[2,0]-q[1,0])*(y-q[2,1]))/den
        b=((q[2,1]-q[0,1])*(x-q[2,0])+(q[0,0]-q[2,0])*(y-q[2,1]))/den
        c=1-a-b;inside=(a>=-1e-10)&(b>=-1e-10)&(c>=-1e-10)
        inv=a/z[0]+b/z[1]+c/z[2];depth=np.full(inv.shape,np.inf)
        np.divide(1,inv,out=depth,where=inside&(inv>0))
        old=zbuf[low[1]:high[1]+1,low[0]:high[0]+1];np.minimum(old,depth,out=old)
    zbuf[~np.isfinite(zbuf)]=0
    return zbuf
