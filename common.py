import csv, math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def load():
    a=np.genfromtxt(ROOT/'data/records.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
    names=a.dtype.names
    X=np.column_stack([a[n] for n in names[1:8]]).astype(float)
    y=np.asarray(a['offer_declined'],float)
    val=np.asarray(a['days_to_hire'],float)
    return X,y,val

def write_results(path,d):
    with open(path,'w') as f:
        for k,v in d.items():
            if isinstance(v,(bool,np.bool_)): s='yes' if v else 'no'
            elif isinstance(v,(int,np.integer)): s=str(int(v))
            elif isinstance(v,str): s=v
            else: s=f'{float(v):.4f}'
            f.write(f'{k} = {s}\n')

def write_csv(path,header,rows):
    with open(path,'w',newline='') as f:
        w=csv.writer(f); w.writerow(header)
        for row in rows:
            w.writerow([f'{x:.4f}' if isinstance(x,(float,np.floating)) else x for x in row])

def split_standardise(X,y):
    i0=np.where(y==0)[0]; i1=np.where(y==1)[0]
    tr=np.sort(np.r_[i0[:7*len(i0)//10],i1[:7*len(i1)//10]])
    te=np.sort(np.r_[i0[7*len(i0)//10:],i1[7*len(i1)//10:]])
    mu=X[tr].mean(axis=0); sd=X[tr].std(axis=0,ddof=1)
    return tr,te,(X[tr]-mu)/sd,(X[te]-mu)/sd,mu,sd

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))
def bce(y,p):
    p=np.clip(p,1e-15,1-1e-15)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
def fit_logistic(X,y,eta=.1,steps=1000,record=()):
    w=np.zeros(X.shape[1]); b=0.; losses={0:bce(y,np.full(len(y),.5))}
    for ep in range(1,steps+1):
        p=sigmoid(X@w+b); e=p-y
        w-=eta*(X.T@e)/len(y); b-=eta*np.mean(e)
        if ep in record: losses[ep]=bce(y,sigmoid(X@w+b))
    return w,b,losses

def metrics(y,pred):
    tp=int(np.sum((y==1)&(pred==1))); fp=int(np.sum((y==0)&(pred==1)))
    tn=int(np.sum((y==0)&(pred==0))); fn=int(np.sum((y==1)&(pred==0)))
    pr=tp/(tp+fp) if tp+fp else 0.; re=tp/(tp+fn) if tp+fn else 0.
    return tp,fp,tn,fn,(tp+tn)/len(y),pr,re,2*pr*re/(pr+re) if pr+re else 0.,tn/(tn+fp) if tn+fp else 0.

def knn_probs(refX,refy,qX,k):
    out=[]
    for q in qX:
        d=np.sum((refX-q)**2,axis=1)
        ix=np.argsort(d,kind='stable')[:k]
        out.append(np.mean(refy[ix]))
    return np.asarray(out)
