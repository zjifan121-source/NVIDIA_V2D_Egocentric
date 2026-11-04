"""Symmetry-aware motion statistics for candidate trajectories (no GT).
Cups (blue_cup, beige_cup): rotation step = tilt of the symmetry axis (axial spin is unscored by the official metric).
white_pot_lid: rotation step modulo a half turn about its axis. Others: full rotation step.
Translation step uses the mesh centroid, not the mesh origin, so axial spin does not inflate it."""
import numpy as np
AXIAL={'blue_cup','beige_cup'}; HALF={'white_pot_lid'}
def symmetry_axis(V):
    c=V.mean(0); w,E=np.linalg.eigh(np.cov((V-c).T))
    # the axis is the eigenvector whose two perpendicular eigenvalues are the most equal (circular cross-section)
    k=int(np.argmin([abs(w[(i+1)%3]-w[(i+2)%3]) for i in range(3)])); return E[:,k]/np.linalg.norm(E[:,k]), c
def rot_angle(Ra,Rb): return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T@Rb)-1)/2,-1,1))))
def axis_rot(a,ang):
    a=a/np.linalg.norm(a); K=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]]); return np.eye(3)+np.sin(ang)*K+(1-np.cos(ang))*K@K
def motion(W,obj,V):
    a,c=symmetry_axis(V); R=W[:,:3,:3]; P=np.einsum('nij,j->ni',R,c)+W[:,:3,3]
    dT=np.linalg.norm(np.diff(P,axis=0),axis=1)*1000
    if obj in AXIAL:
        d=np.einsum('nij,j->ni',R,a); d/=np.linalg.norm(d,axis=1,keepdims=True)
        dR=np.degrees(np.arccos(np.clip(np.einsum('ni,ni->n',d[:-1],d[1:]),-1,1)))
    elif obj in HALF:
        F=axis_rot(a,np.pi); dR=np.array([min(rot_angle(R[i],R[i+1]),rot_angle(R[i],R[i+1]@F)) for i in range(len(R)-1)])
    else:
        dR=np.array([rot_angle(R[i],R[i+1]) for i in range(len(R)-1)])
    return dict(trans_p95_mm=float(np.percentile(dT,95)),trans_max_mm=float(dT.max()),rot_p95_deg=float(np.percentile(dR,95)),rot_max_deg=float(dR.max()),
                symmetry='axial' if obj in AXIAL else ('half_turn' if obj in HALF else 'none'),n_rot_steps_over_30=int((dR>30).sum()))
