import numpy as np
from common import ROOT,load,write_results,write_csv
X,y,v=load(); F=X[:,:5]; mu=F.mean(0); sd=F.std(0,ddof=1); Z=(F-mu)/sd; n=len(Z); r={}
assignments={}
for k in range(2,7):
    cent=Z[np.arange(k)*(n//k)].copy(); old=None
    while True:
        d=((Z[:,None,:]-cent[None,:,:])**2).sum(2); lab=np.argmin(d,axis=1)
        if old is not None and np.array_equal(lab,old): break
        old=lab.copy()
        for j in range(k):
            if np.any(lab==j): cent[j]=Z[lab==j].mean(0)
    wcss=float(np.sum((Z-cent[lab])**2)); r[f'wcss_k{k}']=wcss
    if k==3: assignments[3]=(lab.copy(),cent.copy())
drops=[r[f'wcss_k{k}']-r[f'wcss_k{k+1}'] for k in range(2,6)]
elbow=next((k for k in (3,4,5) if r[f'wcss_k{k}']-r[f'wcss_k{k+1}'] < .5*(r[f'wcss_k{k-1}']-r[f'wcss_k{k}'])),5); r['elbow_k']=elbow
lab,_=assignments[3]; order=sorted(range(3),key=lambda j:F[lab==j,2].mean()); seglabels={}
segrows=[]
for name,j in zip('ABC',order):
    mask=lab==j; seglabels[name]=mask; r[f'size_segment_{name}']=int(mask.sum())
    segrows.append((name,int(mask.sum()),*F[mask].mean(0),float(y[mask].mean()),float(v[mask].mean())))
write_csv(ROOT/'outputs/part3_segments.csv',['segment','size','mean_X1','mean_X2','mean_X3','mean_X4','mean_X5','outcome_rate','mean_value'],segrows)
# Hierarchical clustering
D=np.array([[0,7.9,7.4,2.9,2.9,6.1],[7.9,0,1.4,5.9,5.3,2.0],[7.4,1.4,0,5.8,4.6,1.3],[2.9,5.9,5.8,0,2.8,4.5],[2.9,5.3,4.6,2.8,0,3.4],[6.1,2.0,1.3,4.5,3.4,0]])
hrows=[]; cutclusters={}
for method in ('single','complete','average'):
    clusters=[{i} for i in range(6)]; heights=[]; merges=[]
    for m in range(1,6):
        cand=[]
        for a in range(len(clusters)):
            for b in range(a+1,len(clusters)):
                vals=[D[i,j] for i in clusters[a] for j in clusters[b]]
                h=min(vals) if method=='single' else max(vals) if method=='complete' else sum(vals)/len(vals)
                cand.append((h,a,b))
        h,a,b=min(cand); ca,cb=clusters[a],clusters[b]
        if min(ca)>min(cb): ca,cb=cb,ca
        hrows.append((method,m,'+'.join(f'E{i+1}' for i in sorted(ca)),'+'.join(f'E{i+1}' for i in sorted(cb)),h))
        merged=ca|cb; clusters=[c for z,c in enumerate(clusters) if z not in (a,b)]+[merged]; clusters.sort(key=lambda c:min(c)); heights.append(h)
    gaps=np.diff(heights); below=int(np.argmax(gaps))+1 # merges below largest gap
    cutclusters[method]=['+'.join(f'E{i+1}' for i in sorted(c)) for c in clusters] if False else None
    # replay clusters through number of merges below cut
    cc=[{i} for i in range(6)]
    for m in range(below):
        # use logged merge strings
        row=hrows[-5+m]; A=set(int(x[1:])-1 for x in row[2].split('+')); B=set(int(x[1:])-1 for x in row[3].split('+')); cc=[c for c in cc if c not in (A,B)]+[A|B]; cc.sort(key=lambda c:min(c))
    cutclusters[method]=['+'.join(f'E{i+1}' for i in sorted(c)) for c in cc]
write_csv(ROOT/'outputs/part3_hierarchical.csv',['linkage','merge_no','cluster_a','cluster_b','height'],hrows)
for method in ('single','complete','average'): r[f'{method}_clusters_at_cut']=';'.join(cutclusters[method])
# PCA
cov=Z.T@Z/(n-1); eig,vec=np.linalg.eigh(cov); ix=np.argsort(eig)[::-1]; eig=eig[ix]; vec=vec[:,ix]
for j in range(5):
    k=np.argmax(np.abs(vec[:,j]));
    if vec[k,j]<0: vec[:,j]*=-1
rat=eig/eig.sum(); cum=np.cumsum(rat); proj=Z@vec[:,:2]
r['eigenvalue_sum']=eig.sum()
for j in range(5): r[f'eigenvalue_{j+1}']=eig[j]; r[f'explained_ratio_{j+1}']=rat[j]; r[f'cumulative_ratio_{j+1}']=cum[j]
r['components_for_80pct']=int(np.argmax(cum>=.8)+1)
for j in range(5): r[f'pc1_loading_X{j+1}']=vec[j,0]; r[f'pc2_loading_X{j+1}']=vec[j,1]
for i in range(3): r[f'row{i+1}_pc1']=proj[i,0]; r[f'row{i+1}_pc2']=proj[i,1]
for name,mask in seglabels.items(): r[f'mean_pc1_segment_{name}']=proj[mask,0].mean()
write_results(ROOT/'outputs/part3_results.txt',r)
