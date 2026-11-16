import numpy as np, math
from common import ROOT,load,write_results,write_csv,split_standardise,sigmoid,metrics,knn_probs
X,y,v=load(); tr,te,xt,xv,mu,sd=split_standardise(X,y); yt=y[tr].astype(int); yv=y[te].astype(int)
# use exact Part 2 model
w=np.zeros(7); b=0.
for _ in range(1000):
    p=sigmoid(xt@w+b); e=p-yt; w-=.1*(xt.T@e)/len(yt); b-=.1*np.mean(e)
pl=sigmoid(xv@w+b); pt=sigmoid(xt@w+b)
# K selection by the same 5-fold procedure
cv=[]
for k in range(1,22,2):
    fs=[]
    for fold in range(5):
        va=np.where(np.arange(len(tr))%5==fold)[0]; rr=np.where(np.arange(len(tr))%5!=fold)[0]
        pp=knn_probs(xt[rr],yt[rr],xt[va],k); fs.append(metrics(yt[va],(pp>=(k+1)/2/k).astype(int))[7])
    cv.append((k,np.mean(fs)))
k=sorted(cv,key=lambda z:(-z[1],z[0]))[0][0]; pk=knn_probs(xt,yt,xv,k); pktr=knn_probs(xt,yt,xt,k)
# entropy and gini
def impurity(a,kind):
    if len(a)==0:return 0.
    p=float(np.mean(a)); q=1-p
    if kind=='entropy': return -(p*math.log2(p) if p else 0)-(q*math.log2(q) if q else 0)
    return 1-p*p-q*q
rootE=impurity(yt,'entropy'); rootG=impurity(yt,'gini'); r={'root_entropy':rootE,'root_gini':rootG}; splitrows=[]
conds=[X[tr,5]==1,X[tr,6]==1,X[tr,2]>.5,X[tr,0]<=24]
for i,c in enumerate(conds,1):
    yes=yt[c==1]; no=yt[c==0]; ig=rootE-(len(yes)/len(yt)*impurity(yes,'entropy')+len(no)/len(yt)*impurity(no,'entropy')); gg=rootG-(len(yes)/len(yt)*impurity(yes,'gini')+len(no)/len(yt)*impurity(no,'gini'))
    splitrows.append((f'S{i}',len(yes),int(yes.sum()),len(no),int(no.sum()),ig,gg))
bestE=max(splitrows,key=lambda z:z[5])[0]; bestG=max(splitrows,key=lambda z:z[6])[0]; r['best_split_by_info_gain']=bestE; r['best_split_by_gini_gain']=bestG
write_csv(ROOT/'outputs/part4_splits.csv',['split','n_yes','positives_yes','n_no','positives_no','info_gain','gini_gain'],splitrows)
ci=int(bestE[1:])-1; cond=conds[ci]; rateY=float(yt[cond].mean()); rateN=float(yt[~cond].mean()); r['stump_rate_yes']=rateY; r['stump_rate_no']=rateN
ps=np.where(cond,rateY,rateN) # training side rates for test need test condition
if ci==0: ctest=X[te,5]==1
elif ci==1: ctest=X[te,6]==1
elif ci==2: ctest=X[te,2]>.5
else: ctest=X[te,0]<=24
pst=np.where(ctest,rateY,rateN)
probs={'logistic':pl,'knn':pk,'stump':pst}; preds={'logistic':pl>=.3,'knn':pk>=(k+1)/2/k,'stump':pst>=.3}
preds['hard_vote']=(preds['logistic'].astype(int)+preds['knn'].astype(int)+preds['stump'].astype(int))>=2
probs['soft_vote']=(pl+pk+pst)/3; preds['soft_vote']=probs['soft_vote']>=.3
vrows=[]
for name in ('logistic','knn','stump','hard_vote','soft_vote'):
    m=metrics(yv,preds[name]); vrows.append((name,*m[:4],m[5],m[6],m[7]))
write_csv(ROOT/'outputs/part4_voting.csv',['model','tp','fp','tn','fn','precision','recall','f1'],vrows)
# AdaBoost
Y=np.array([1,1,1,-1,-1,-1,1,-1]); h1=np.array([1,1,-1,-1,-1,-1,-1,-1]); h2=np.array([-1,1,1,-1,1,-1,1,-1]); weights=np.ones(8)/8; adrows=[]; alphas=[]; errs=[]
for rnd,h in ((1,h1),(2,h2)):
    before=weights.copy(); err=float(before[h!=Y].sum()); alpha=.5*math.log((1-err)/err); un=before*np.exp(-alpha*Y*h); after=un/un.sum()
    for j in range(8): adrows.append((rnd,j+1,before[j],un[j],after[j]))
    errs.append(err); alphas.append(alpha); weights=after
r.update(round1_error=errs[0],round1_alpha=alphas[0],round2_error=errs[1],round2_alpha=alphas[1],ensemble_accuracy=float(np.mean(np.sign(alphas[0]*h1+alphas[1]*h2)==Y)))
write_csv(ROOT/'outputs/part4_adaboost.csv',['round','point','weight_before','weight_unnormalised','weight_after'],adrows)
r['oob_fraction']=(1-1/560)**560
# Architecture counts
arch=[('mlp_hidden1',784*16+16),('mlp_hidden2',16*16+16),('mlp_output',16*10+10),('cnn_conv1',5*5*1*6+6),('cnn_conv2',5*5*6*16+16),('cnn_dense1',5*5*16*32+32),('cnn_dense2',32*10+10)]
write_csv(ROOT/'outputs/part4_architecture.csv',['layer','parameters'],arch)
r['mlp_total']=sum(x[1] for x in arch[:3]); r['cnn_total']=sum(x[1] for x in arch[3:])
r['fc_layer_params']=128*128*3*1024+1024; r['conv_layer_params']=3*3*3*64+64
r['rnn_total']=64*128+128*128+128*16+128+16
write_results(ROOT/'outputs/part4_results.txt',r)
