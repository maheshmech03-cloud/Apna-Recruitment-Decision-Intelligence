import numpy as np
from common import ROOT,load,write_results
X,y,v=load(); x=X[:,3]; n=len(v); mx=v.mean(); sx=v.std(ddof=1); se=sx/np.sqrt(n)
r={'n':n,'mean_value':mx,'sd_value':sx,'se_value':se,'ci95_low':mx-1.96*se,'ci95_high':mx+1.96*se}
r['z_stat']=(mx-8)/se; r['reject_h0']=abs(r['z_stat'])>1.96
r['n_required']=int(np.ceil((1.96*sx/.5)**2))
slope=np.sum((x-x.mean())*(v-v.mean()))/np.sum((x-x.mean())**2); intercept=mx-slope*x.mean(); fit=intercept+slope*x; resid=v-fit
TSS=np.sum((v-mx)**2); ESS=np.sum((fit-mx)**2); RSS=np.sum(resid**2)
r.update(reg_slope=slope,reg_intercept=intercept,tss=TSS,ess=ESS,rss=RSS,tss_equals_ess_plus_rss=abs(TSS-ESS-RSS)<1e-6*TSS,r2=1-RSS/TSS,rmse=np.sqrt(RSS/n),mae=np.mean(abs(resid)))
z=(x-x.mean())/x.std(ddof=1); m=c=0.; costs={0:np.mean(v*v)}
for i in range(1,501):
    e=v-(m*z+c); gm=-2*np.mean(z*e); gc=-2*np.mean(e); m-=.1*gm; c-=.1*gc
    if i in (1,10,100,500): costs[i]=np.mean((v-(m*z+c))**2)
r.update(gd_J0=costs[0],gd_J1=costs[1],gd_J10=costs[10],gd_J100=costs[100],gd_J500=costs[500],gd_slope=m/x.std(ddof=1),gd_intercept=c-m*x.mean()/x.std(ddof=1),gd_slope_gap=abs(m/x.std(ddof=1)-slope))
order=np.argsort(fit,kind='stable'); lo=order[:400]; hi=order[400:]; r['resid_sd_lower_raw']=resid[lo].std(ddof=1); r['resid_sd_upper_raw']=resid[hi].std(ddof=1); r['resid_sd_ratio_raw']=r['resid_sd_upper_raw']/r['resid_sd_lower_raw']
lx=np.log(x); lv=np.log(v); ls=np.cov(lx,lv,ddof=1)[0,1]/np.var(lx,ddof=1); li=lv.mean()-ls*lx.mean(); lf=li+ls*lx; lr=lv-lf; lT=np.sum((lv-lv.mean())**2); lRSS=np.sum(lr**2); lr2=1-lRSS/lT
r.update(loglog_intercept=li,loglog_slope=ls,loglog_r2=lr2,loglog_rmse=np.sqrt(lRSS/n))
ord=np.argsort(lf,kind='stable'); llo=ord[:400]; lhi=ord[400:]; r.update(resid_sd_lower_log=lr[llo].std(ddof=1),resid_sd_upper_log=lr[lhi].std(ddof=1),resid_sd_ratio_log=lr[lhi].std(ddof=1)/lr[llo].std(ddof=1))
A=np.column_stack([np.ones(n),np.log(x),X[:,0],X[:,1],X[:,2],X[:,4],X[:,5],X[:,6]])
beta=np.linalg.solve(A.T@A,A.T@lv); pred=A@beta; mrss=np.sum((lv-pred)**2); mr2=1-mrss/lT
for j,b in enumerate(beta): r[f'mr_beta_{j}']=b
r.update(mr_r2=mr2,mr_adj_r2=1-(1-mr2)*(n-1)/(n-7-1),loglog_adj_r2=1-(1-lr2)*(n-1)/(n-1-1))
write_results(ROOT/'outputs/part1_results.txt',r)
