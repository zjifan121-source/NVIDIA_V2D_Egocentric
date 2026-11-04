import numpy as np, pyarrow.parquet as pq, glob, sys, json
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import ConvexHull
SC=sys.argv[1]; TAG=sys.argv[2] if len(sys.argv)>2 else 'A1'
pqp=glob.glob(SC+'/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=*/*.parquet')[0]
t=pq.read_table(pqp); g=lambda k: np.array(t.column(k)[0].as_py(),dtype=float)
P=g('object_body_position'); Q=g('object_body_wxyz')
bn=t.column('object_body_names')[0].as_py()
proc=glob.glob(SC+'/assets/human_motion_data/ego_recon/processed/')[0]
def load(p):
    V=[];F=[]
    for l in open(p):
        if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith('f '):
            i=[int(s.split('/')[0])-1 for s in l.split()[1:]]
            for k in range(1,len(i)-1): F.append([i[0],i[k],i[k+1]])
    return np.array(V),np.array(F)
out={}
for k,b in enumerate(bn):
    nm='white_pot' if b.endswith('white_pot') else 'wooden_spoon'
    V,F=load(proc+b+'.obj')
    a,bb,c=V[F[:,0]],V[F[:,1]],V[F[:,2]]
    vol6=np.einsum('ij,ij->i',a,np.cross(bb,c)); vc=((vol6[:,None]*(a+bb+c)/4).sum(0)/vol6.sum())
    ar=0.5*np.linalg.norm(np.cross(bb-a,c-a),axis=1); sc=(ar[:,None]*(a+bb+c)/3).sum(0)/ar.sum()
    res={'origin_to_volcentroid_cm':round(float(np.linalg.norm(vc)*100),2),'origin_to_surfcentroid_cm':round(float(np.linalg.norm(sc)*100),2)}
    for f in (0,312):
        R=Rot.from_quat(np.r_[Q[f,k,1:],Q[f,k,0]]).as_matrix(); W=V@R.T+P[f,k]
        zmin=W[:,2].min(); base=W[W[:,2]<zmin+0.003][:,:2]
        hull=ConvexHull(base); eq=hull.equations
        def margin(pt): return float((-(eq[:,:2]@pt[:2]+eq[:,2])).min())
        com=P[f,k]; vcw=vc@R.T+P[f,k]; scw=sc@R.T+P[f,k]
        r={'base_hull_extent_cm':[round(float(x)*100,1) for x in base.max(0)-base.min(0)],
           'margin_simCoM_origin_cm':round(margin(com)*100,2),'margin_volcentroid_cm':round(margin(vcw)*100,2),'margin_surfcentroid_cm':round(margin(scw)*100,2),
           'simCoM_height_cm':round(float(com[2]-zmin)*100,2),'simCoM_horizontal_offset_from_volcentroid_cm':round(float(np.linalg.norm((com-vcw)[:2]))*100,2)}
        # tipping: rotate about the hull edge nearest the sim CoM until another vertex touches z=zmin
        i=int(np.argmin(-(eq[:,:2]@com[:2]+eq[:,2])))
        n=eq[i,:2]; d=eq[i,2]
        # edge line points: hull vertices on facet i
        e0,e1=base[hull.simplices[i][0]],base[hull.simplices[i][1]]
        axis=np.r_[e1-e0,0]; axis/=np.linalg.norm(axis)
        piv=np.r_[e0,zmin]
        # direction: tipping outward (toward n) means points beyond the edge go down
        best=None
        for ang in np.arange(0.5,60.01,0.25):
            for sgn in (1,-1):
                Rt=Rot.from_rotvec(sgn*np.radians(ang)*axis).as_matrix()
                Wt=(W-piv)@Rt.T+piv
                # proper sign: the CoM must move toward/over the edge (outward along n)
                comt=(com-piv)@Rt.T+piv
                if (n@comt[:2]+d) < (n@com[:2]+d): continue
                if Wt[:,2].min()<zmin-1e-4:
                    best=(ang,sgn) if best is None else best
                    break
            if best: break
        r['tip_about_nearest_edge_contact_angle_deg']=None if best is None else best[0]
        if best:
            Rt=Rot.from_rotvec(best[1]*np.radians(best[0])*axis).as_matrix(); comt=(com-piv)@Rt.T+piv
            Wt=(W-piv)@Rt.T+piv; low=Wt[Wt[:,2]<zmin+0.003]
            far=np.linalg.norm(low[:,:2]-piv[:2],axis=1).max()*100
            r['at_tip_contact_second_support_dist_from_edge_cm']=round(float(far),1)
            r['at_tip_simCoM_over_edge_cm']=round(float((n@comt[:2]+d))*100,2)
        res[f'f{f}']=r
    out[nm]=res
print(TAG, json.dumps(out, indent=1))
