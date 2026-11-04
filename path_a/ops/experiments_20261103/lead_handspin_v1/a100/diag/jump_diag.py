import sys,numpy as np
b,r=sys.argv[1:3]
z=np.load(b+"/result.npz"); zr=np.load(r+"/result.npz")
print(b.split("/")[-1],"n",len(z["camera_to_world_transform"]),[str(x) for x in z["object_names"]] if "object_names" in z.files else "")
for side in ("left","right"):
    W=z["camera_to_world_transform"].astype(float)@z[f"hand_{side}_wrist_to_camera_transform"].astype(float); Wr=zr["camera_to_world_transform"].astype(float)@zr[f"hand_{side}_wrist_to_camera_transform"].astype(float)
    d=np.linalg.norm(W[:,:3,3]-Wr[:,:3,3],axis=1)*100; m=d>0.01
    st=np.linalg.norm(np.diff(W[:,:3,3],axis=0),axis=1)*100; str_=np.linalg.norm(np.diff(Wr[:,:3,3],axis=0),axis=1)*100
    top=np.argsort(-d)[:4]
    print(" ",side,"moved frames",int(m.sum()),"median",round(float(np.median(d[m])) if m.any() else 0,1),"max",round(float(d.max()),1),"at",[int(x) for x in top],"| wrist step max attached",round(float(st.max()),1),"at",int(st.argmax()),"raw",round(float(str_.max()),1),"at",int(str_.argmax()))
    i=int(d.argmax()); print("    around max:",[ (j,round(float(d[j]),1)) for j in range(max(0,i-3),min(len(d),i+4))])
    print("    raw wrist z around:",[round(float(Wr[j,2,3]),2) for j in range(max(0,i-3),min(len(d),i+4))], "bundle:",[round(float(W[j,2,3]),2) for j in range(max(0,i-3),min(len(d),i+4))])
