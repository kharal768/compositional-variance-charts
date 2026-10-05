#!/usr/bin/env python3
"""Measurements for review comments 2.3, 2.5 and 3.3."""
from wmmon.paths import WMMON_HOME
import json
from pathlib import Path
import numpy as np
from scipy import stats
from wmmon import composition, md3, varcomp as VC
from wmmon.varcomp import _project_psd
from wmmon.laney_coda import sampling_covariance

CACHE=Path(f"{WMMON_HOME}/cache"); RESULTS=Path(f"{WMMON_HOME}/results"); NOM=5.0
CLASSES=None

def cpi(k,n):
    lo=stats.beta.ppf(0.025,k,n-k+1)*1000 if k else 0.0
    hi=stats.beta.ppf(0.975,k+1,n-k)*1000 if k<n else 1000.0
    return lo,hi

def stream(proba, unit, mode="hard"):
    n=len(proba)//unit; p=proba[:n*unit]
    ids=np.array([f"u{i//unit:06d}" for i in range(len(p))],dtype=object)
    seq=[f"u{i:06d}" for i in range(n)]
    fr=composition.build_stream(ids,p,seq,mode=mode)
    return (composition.ilr_matrix(fr), composition.proportion_matrix(fr),
            fr["lot_size"].to_numpy(), ids, seq, p)

# ---- 2.3 Where does the 20(D-1) Phase I requirement come from? ----
def phase_one_rule(proba, unit=24, lag=4):
    print("2.3  PHASE I LENGTH: where the 20(D-1) rule comes from")
    C,P,S,_,_,_ = stream(proba, unit)
    dim=C.shape[1]; n=len(C); warmup=20
    rows=[]
    for per_dim in (5,10,15,20,30,40,60):
        ph=per_dim*dim
        ex=n-ph-warmup
        if ex<90: continue
        f=VC.calibrate(C[:ph],P[:ph],S[:ph],target_far=NOM/1000,lag=lag,seed=0)
        k=int(VC.run(f,C[ph:],P[ph:],S[ph:])["alarm"][warmup:].sum())
        lo,hi=cpi(k,ex)
        rows.append(dict(per_dim=per_dim,phase_one=ph,exposure=ex,
                         rate=1000*k/ex,ci=[lo,hi],covers=bool(lo<=NOM<=hi)))
        print(f"   {per_dim:>3} units/dim (Phase I {ph:>4}, exposure {ex:>4}): "
              f"rate {1000*k/ex:6.1f} CI [{lo:.1f}, {hi:.1f}] "
              f"{'covers' if rows[-1]['covers'] else 'MISS'}")
    ok=[r['per_dim'] for r in rows if r['covers']]
    print(f"   covering: {ok}")
    return rows

# ---- 2.5 How often does the PSD projection activate? ----
def psd_clipping(proba, unit=24, lag=4):
    print("\n2.5  PSD PROJECTION OF Sigma_b: frequency and magnitude")
    out=[]
    for label, pr, u in (("wm811k", proba, unit),):
        C,P,S,_,_,_ = stream(pr,u)
        dim=C.shape[1]; ph=max(20*dim,int(0.4*len(C)))
        ref=P[:ph].mean(axis=0)
        basis=composition.ilr_basis(P.shape[1])
        samp=np.array([sampling_covariance(ref,nn,basis) for nn in S[:ph]]).mean(axis=0)
        d=C[lag:ph]-C[:ph-lag]; total=(d.T@d)/(2*len(d))
        raw=np.atleast_2d(total)-samp
        vals=np.linalg.eigvalsh(raw)
        clipped=_project_psd(raw)
        n_neg=int((vals<0).sum())
        share=float(-vals[vals<0].sum()/max(np.abs(vals).sum(),1e-12)) if n_neg else 0.0
        out.append(dict(dataset=label,unit=u,n_negative=n_neg,dim=dim,
                        eigenvalues=[float(v) for v in vals],
                        negative_mass_share=share,
                        trace_before=float(np.trace(raw)),
                        trace_after=float(np.trace(clipped))))
        print(f"   {label} unit {u}: {n_neg}/{dim} eigenvalues negative; "
              f"negative mass {100*share:.1f}% of total; "
              f"trace {np.trace(raw):.4f} -> {np.trace(clipped):.4f}")
        print("   eigenvalues: " + ", ".join(f"{v:+.4f}" for v in vals))
    return out

# ---- 3.3 MD3 in the corrected-chart comparison ----
def md3_in_corrected(proba, unit=24, lag=4):
    print("\n3.3  MD3 ALONGSIDE THE CORRECTED CHART, count aggregation")
    C,P,S,ids,seq,praw = stream(proba, unit)
    dim=C.shape[1]; ph=max(20*dim,int(0.4*len(C))); warmup=20; ex=len(C)-ph-warmup
    rows=[]
    f=VC.calibrate(C[:ph],P[:ph],S[:ph],target_far=NOM/1000,lag=lag,seed=0)
    k=int(VC.run(f,C[ph:],P[ph:],S[ph:])["alarm"][warmup:].sum()); lo,hi=cpi(k,ex)
    rows.append(dict(method="variance components (lag 4)",rate=1000*k/ex,ci=[lo,hi],covers=bool(lo<=NOM<=hi)))
    # MD3 on the deployed model's margin, empirically thresholded like the rest
    dens=md3.lot_margin_density(md3.margin_indicator(praw),ids,seq)
    score=md3.md3_score(dens,ph,lam=0.2)
    thr=float(np.quantile(score[:warmup+ex//2],1-NOM/1000,method="higher"))
    k2=int((score[warmup:]>thr).sum()); lo2,hi2=cpi(k2,ex)
    rows.append(dict(method="MD3-conf, empirical threshold",rate=1000*k2/ex,ci=[lo2,hi2],covers=bool(lo2<=NOM<=hi2)))
    # MD3 under its own theta*sigma rule at theta=2
    from sklearn.model_selection import KFold
    from wmmon import data
    xtr=np.load(CACHE/"train.npz")["x"]
    df,_=data.load_lswmd(f"{WMMON_HOME}/data/subset.pkl")
    tr,_,_=data.split_labelled(df,seed=0); ytr=tr["label"].tolist()
    oof=np.zeros(len(xtr))
    for a,b in KFold(5,shuffle=True,random_state=0).split(xtr):
        e=md3.train_random_subspace(xtr[a],[ytr[i] for i in a],list(data.CLASSES),seed=0)
        oof[b]=md3.margin_indicator(e.probabilities(xtr[b]))
    rng=np.random.default_rng(0)
    sigma=float(np.std([oof[rng.choice(len(oof),size=int(np.median(S)),replace=False)].mean()
                        for _ in range(2000)],ddof=1))
    k3=int((score[warmup:]>2.0*sigma).sum()); lo3,hi3=cpi(k3,ex)
    rows.append(dict(method="MD3-conf, own theta=2 rule",rate=1000*k3/ex,ci=[lo3,hi3],covers=bool(lo3<=NOM<=hi3)))
    for r in rows:
        print(f"   {r['method']:<34s} rate {r['rate']:7.1f} CI [{r['ci'][0]:.1f}, {r['ci'][1]:.1f}] "
              f"{'covers' if r['covers'] else 'MISS'}")
    print(f"   exposure {ex} units")
    return {"exposure":ex,"rows":rows}

if __name__=="__main__":
    pw=np.load(CACHE/"cnnproba_clean_1200_perm.npz")["p"]
    out={"phase_one":phase_one_rule(pw),"psd":psd_clipping(pw),"md3":md3_in_corrected(pw)}
    RESULTS.mkdir(exist_ok=True)
    json.dump(out,open(RESULTS/"review_batch2.json","w"),indent=2)
    print("\nwrote results/review_batch2.json")
