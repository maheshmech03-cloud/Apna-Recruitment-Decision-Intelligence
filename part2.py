import numpy as np
from common import ROOT,load,write_results,write_csv,split_standardise,sigmoid,bce,fit_logistic,metrics,knn_probs
X,y,v=load(); tr,te,xt,xv,mu,sd=split_standardise(X,y); yt=y[tr].astype(int); yv=y[te].astype(int)
r={'train_rows':len(tr),'train_positives':int(yt.sum()),'test_rows':len(te),'test_positives':int(yv.sum())}
w,b,loss=fit_logistic(xt,yt,.1,1000,(1,10,100,1000)); ptest=sigmoid(xv@w+b); ptrain=sigmoid(xt@w+b)
for ep in (0,1,10,100,1000): r[f'loss_epoch_{ep}']=loss[ep]
for j,a in enumerate(w,1): r[f'weight_X{j}']=a; r[f'odds_ratio_X{j}']=np.exp(a)
r['bias']=b; r['test_bce']=bce(yv,ptest)
for eta in (.01,.1,1.):
    _,_,ll=fit_logistic(xt,yt,eta,200,(200,)); r[f'loss_epoch_200_eta_{eta}']=ll[200]
for tau in (.5,.3):
    m=metrics(yv,(ptest>=tau).astype(int))
    names=('tp','fp','tn','fn','accuracy','precision','recall','f1','specificity')
    for name,val in zip(names,m): r[f'tau_{tau}_{name}']=val
r['blind_accuracy']=np.mean(yv==0)
rows=[]
for tau in np.arange(.1,1,.1):
    tp,fp,tn,fn,*_=metrics(yv,(ptest>=tau).astype(int)); rows.append((round(tau,1),tp,fp,tn,fn,tp/(tp+fn) if tp+fn else 0,fp/(fp+tn) if fp+tn else 0))
write_csv(ROOT/'outputs/part2_thresholds.csv',['tau','tp','fp','tn','fn','tpr','fpr'],rows)
cvrows=[]
for k in range(1,22,2):
    fs=[]
    for fold in range(5):
        va=np.where(np.arange(len(tr))%5==fold)[0]; rr=np.where(np.arange(len(tr))%5!=fold)[0]
        pp=knn_probs(xt[rr],yt[rr],xt[va],k); mm=metrics(yt[va],(pp>=(k+1)/2/k).astype(int)); fs.append(mm[7])
    cvrows.append((k,float(np.mean(fs))))
best=sorted(cvrows,key=lambda a:(-a[1],a[0]))[0][0]; r['knn_best_k']=best
write_csv(ROOT/'outputs/part2_knn_cv.csv',['K','mean_cv_f1'],cvrows)
kp=knn_probs(xt,yt,xv,best); km=metrics(yv,(kp>=(best+1)/2/best).astype(int))
for name,val in zip(('knn_tp','knn_fp','knn_tn','knn_fn','knn_accuracy','knn_precision','knn_recall','knn_f1'),km[:8]): r[name]=val
costs=[]
for tau in np.arange(1,20)/20:
    mm=metrics(yt,(ptrain>=tau).astype(int)); costs.append((50*mm[1]+400*mm[3],tau))
cost_train,tau=min(costs,key=lambda a:(a[0],a[1])); mt=metrics(yv,(ptest>=tau).astype(int)); m5=metrics(yv,(ptest>=.5).astype(int))
r.update({'cost_tau':tau,'cost_train_at_tau':cost_train,'cost_test_at_tau':50*mt[1]+400*mt[3],'cost_test_at_0.5':50*m5[1]+400*m5[3],'cost_test_flag_nobody':400*int(yv.sum())})
write_results(ROOT/'outputs/part2_results.txt',r)
