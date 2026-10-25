"""Calendar windows preserved; train moved earlier to make room for exact label embargoes."""
import numpy as np
import pandas as pd
from .data import eligible

def make_fold(d,year,L=300,strict=False):
    t=d.target_trade_date
    start=pd.Timestamp(year,1,1); mid=start-pd.DateOffset(months=6); prior=start-pd.DateOffset(years=1)
    ok=eligible(d,L,strict)
    def ix(mask): return np.flatnonzero(mask.to_numpy() & ok)
    val=ix((t>=prior)&(t<mid))
    cal=ix((t>=mid)&(t<start))
    train=ix(t<prior); test=ix((t>=start)&(t<pd.Timestamp(year+1,1,1)))
    # Embargo counted in underlying bar_seq, not filtered row positions.
    if len(val): train=train[d.bar_seq.iloc[train].to_numpy()<int(d.bar_seq.iloc[val[0]])-25]
    if len(cal): val=val[d.bar_seq.iloc[val].to_numpy()<int(d.bar_seq.iloc[cal[0]])-25]
    if len(test): cal=cal[d.bar_seq.iloc[cal].to_numpy()<int(d.bar_seq.iloc[test[0]])-1]
    parts=dict(train=train,validation=val,calibration=cal,test=test)
    if any(len(v)<(20 if strict else 8) for v in parts.values()): raise ValueError(f'Fold {year} too small: '+str({k:len(v) for k,v in parts.items()}))
    for a,b in [('train','validation'),('validation','calibration'),('calibration','test')]:
        if d.target_end.iloc[parts[a]].max()>=d.target_end.iloc[parts[b]].min(): raise AssertionError('Label overlap')
    return parts

def manifest(d,L=300,include_lockbox=False):
    out={'policy':'Fixed calendar validation Jan-Jun and calibration Jul-Dec; trim earlier split for 25 underlying-bar embargoes and 1-bar purge.',
         'embargo_units':'underlying bar_seq (not surviving rows)', 'shared_eligibility_lookback':L,
         'mode':d.attrs['mode'],'dataset_sha256':d.attrs['dataset_sha256'],
         'sealed_lockbox':{'start':'2026-01-01','end':'end of dataset','indices':'resolved only after dedicated authorization','executed':False},'folds':{}}
    for y in range(2016,2027 if include_lockbox else 2026):
        p=make_fold(d,y,L,d.attrs['mode']=='strict')
        out['folds'][str(y)]={'role':'development' if y<2019 else 'evaluation' if y<2026 else 'lockbox',
                             'indices':{k:v.tolist() for k,v in p.items()},
                             'bounds':{k:dict(n=len(v),first_target=str(d.target_end.iloc[v].min()),last_target=str(d.target_end.iloc[v].max())) for k,v in p.items()}}
    return out
