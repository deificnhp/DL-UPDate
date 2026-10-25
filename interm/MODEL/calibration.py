import numpy as np
from sklearn.isotonic import IsotonicRegression
from .config import TAUS
PAIRS=[(0,6,.1),(1,5,.2),(2,4,.5)]

def pup(q):
    """Bounded continuous CDF with slope-matched exponential tails.
    Duplicate quantiles: use a right-continuous CDF (largest tau at tie).
    """
    q=np.asarray(q,float);shape=q.shape[:-1];flat=q.reshape(-1,7);out=np.empty(len(flat));eps=1e-12
    tau=np.asarray(TAUS)
    for j,row in enumerate(flat):
        if np.any(np.diff(row)<0) or not np.isfinite(row).all(): raise ValueError('CDF requires finite ordered quantiles')
        if 0<row[0]:
            slope=.05/max(row[1]-row[0],eps)
            f=.05*np.exp(np.clip((0-row[0])*slope/.05,-745,0))
        elif 0>row[-1]:
            slope=.05/max(row[-1]-row[-2],eps)
            f=1-.05*np.exp(np.clip(-(0-row[-1])*slope/.05,-745,0))
        else:
            i=np.searchsorted(row,0,side='right')-1
            if i>=6:f=.95
            else:f=tau[i]+(tau[i+1]-tau[i])*(0-row[i])/max(row[i+1]-row[i],eps)
        out[j]=np.clip(1-f,.001,.999)
    return out.reshape(shape)

def offset(scores,alpha):
    s=np.sort(scores);rank=int(np.ceil((len(s)+1)*(1-alpha)))
    return float(s[np.clip(rank-1,0,len(s)-1)])

def signals(p,q,theta):
    sig=np.where(p>=.5+theta,1,np.where(p<=.5-theta,-1,0))
    return np.where(abs(q[:,3])<.0002,0,sig)

def score(probs):return abs(2*probs.mean(0)-1)*(1-np.minimum(1,probs.std(0)/.10))

class Calibrator:
    def fit(self,member_q,y):
        q=member_q.mean(0);y=np.asarray(y)
        self.bias=float(np.median(y-q[:,3]))
        self.scores=[np.maximum(q[:,lo]-y,y-q[:,hi]) for lo,hi,_ in PAIRS]
        self.alpha=np.array([a for _,_,a in PAIRS]); self.gamma=.005
        cq=self.apply(member_q,online=False);point=cq.mean(0);p=pup(point)
        # Threshold objective explicitly fixed: average after-cost return per opportunity,
        # isolated one-bar trades (2bp each), tie -> smallest theta.
        candidates=[0,.02,.04,.06,.08]
        values=[float(np.mean(signals(p,point,t)*y-.0002*abs(signals(p,point,t)))) for t in candidates]
        self.theta=candidates[int(np.argmax(values))]
        sig=signals(p,point,self.theta);s=score(pup(cq));active=sig!=0
        self.isotonic=None;self.confidence_fallback=False
        if active.sum()>=10 and np.unique(s[active]).size>=2:
            self.isotonic=IsotonicRegression(out_of_bounds='clip',y_min=.5,y_max=1.).fit(s[active],(sig[active]*y[active]>0).astype(float))
        else:self.confidence_fallback=True
        self.threshold_objectives=dict(zip(map(str,candidates),values))
        return self
    def apply(self,member_q,online=False,y=None):
        member_q=np.asarray(member_q,float)
        if not online:
            out=member_q.copy();out[:,:,3]+=self.bias
            for (lo,hi,a),scores in zip(PAIRS,self.scores):
                off=offset(scores,a);out[:,:,lo]-=off;out[:,:,hi]+=off
            return np.sort(out,axis=-1)
        if y is None:raise ValueError('Online ACI needs outcomes, consumed AFTER prediction')
        out=np.empty_like(member_q);alpha=self.alpha.copy();trace=[]
        for t in range(member_q.shape[1]):
            row=member_q[:,t].copy();row[:,3]+=self.bias
            for j,(lo,hi,nominal) in enumerate(PAIRS):
                off=offset(self.scores[j],alpha[j]);row[:,lo]-=off;row[:,hi]+=off
            row.sort(axis=-1);out[:,t]=row
            # Only now reveal y_t (all targets resolve to the next observed decision).
            pooled=row.mean(0)
            for j,(lo,hi,nominal) in enumerate(PAIRS):
                err=float(y[t]<pooled[lo] or y[t]>pooled[hi])
                alpha[j]=np.clip(alpha[j]+self.gamma*(nominal-err),1/(len(self.scores[j])+1),.999)
            trace.append(alpha.tolist())
        self.aci_trace=trace
        assert (np.diff(out,axis=-1)>=0).all()
        return out
    def diagnostics(self,calibrated):
        probs=pup(calibrated);q=calibrated.mean(0);p=pup(q);sig=signals(p,q,self.theta)
        raw=score(probs);conf=np.full(len(p),np.nan);active=sig!=0
        conf[active]=self.isotonic.predict(raw[active]) if self.isotonic is not None else .5
        return dict(p_up=p,signal=sig,confidence=conf,uncertainty_score=raw,member_p_std=probs.std(0))
    def metadata(self):
        return dict(median_bias=self.bias,theta=self.theta,threshold_objectives=self.threshold_objectives,
                    confidence_fallback=self.confidence_fallback,gamma=self.gamma,
                    cqr_offsets=[offset(s,a) for s,(_,_,a) in zip(self.scores,PAIRS)])
