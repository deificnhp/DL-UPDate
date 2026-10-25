"""Audit schema and resolve labels WITHOUT pretending next retained row is the next bar."""
import numpy as np
import pandas as pd
from .config import FEATURES, ART, dump, sha

def load(path,allow_sparse=False,lockbox=False):
    import datetime
    import pyarrow.parquet as pq
    # File verified as date32 trade_date. Arrow filtering prevents materializing 2026
    # outcomes into pandas during normal execution (hashing file bytes is permitted).
    cutoff=datetime.date(2026,1,1)
    d=pd.read_parquet(path,filters=None if lockbox else [('trade_date','<',cutoff)])
    required=FEATURES+['bar_start_utc','bar_end_utc','trade_date','slot','slot_next','dow_next',
                       'bar_seq','r_next','sigma_ewma','next_valid','close_mid','expected_minutes',
                       'target_bar_seq']
    missing=sorted(set(required)-set(d.columns))
    if missing: raise ValueError(f'Missing columns: {missing}')
    d=d.sort_values('bar_end_utc',kind='stable').reset_index(drop=True)
    for c in ['bar_start_utc','bar_end_utc']: d[c]=pd.to_datetime(d[c],utc=True)
    d['trade_date']=pd.to_datetime(d.trade_date)
    # Seal outcomes before any numerical feature/target audit in normal operation.
    if not lockbox: d=d.loc[d.trade_date<'2026-01-01'].copy().reset_index(drop=True)
    if not d.trade_date.is_monotonic_increasing:raise ValueError('trade_date is not monotonic in decision time')
    if d.bar_end_utc.duplicated().any() or d.bar_seq.duplicated().any(): raise ValueError('Duplicate bar identifier')
    for c in ['slot','slot_next','dow_next']:
        if not d[c].isin(range(5)).all(): raise ValueError(f'Invalid {c}')
    if not np.isfinite(d[FEATURES+['sigma_ewma','r_next']].to_numpy()).all(): raise ValueError('Nonfinite required data')
    if not (d.sigma_ewma>0).all(): raise ValueError('Nonpositive causal scale')
    if not d.expected_minutes.eq(240).all(): raise ValueError('Nonhomogeneous bars')
    if not (d.bar_end_utc-d.bar_start_utc).eq(pd.Timedelta(hours=4)).all():raise ValueError('Observed bar duration is not four hours')
    if 'is_terminal_3h' in d and d.is_terminal_3h.any():raise ValueError('Terminal slot not removed')
    # Resolve the target bar by IDENTITY, not by file adjacency.
    # The feature layer records the target of row t explicitly as target_bar_seq[t] (the
    # bar_seq of the chronological successor in the source frame) and guarantees that
    # r_next[t], slot_next[t] and dow_next[t] all describe that ONE bar. The target bar is
    # only resolvable here if it is itself retained in this file; otherwise the row carries
    # a valid label but no verifiable target timestamp and is excluded (label_ok=False).
    tgt_seq=d.target_bar_seq.to_numpy()
    row_of=np.full(int(d.bar_seq.max())+2,-1,dtype=np.int64)
    row_of[d.bar_seq.to_numpy()]=np.arange(len(d))
    tgt_row=row_of[np.where(tgt_seq>=0,tgt_seq,0)]
    in_file=(tgt_seq>=0)&(tgt_row>=0)
    target_end=pd.Series(pd.NaT,index=d.index,dtype='datetime64[ns, UTC]')
    target_td=pd.Series(pd.NaT,index=d.index,dtype='datetime64[ns]')
    if in_file.any():
        target_end[in_file]=pd.Series(d.bar_end_utc.to_numpy()[tgt_row[in_file]],index=d.index[in_file])
        target_td[in_file]=pd.Series(d.trade_date.to_numpy()[tgt_row[in_file]],index=d.index[in_file])
    d['target_end']=target_end
    d['target_trade_date']=target_td
    # The resolved target must agree with the covariates shipped by the feature layer.
    resolve=in_file&d.next_valid.to_numpy().astype(bool)
    if resolve.any():
        ti=tgt_row[resolve]
        if not (d.slot_next.to_numpy()[resolve]==d.slot.to_numpy()[ti]).all():
            raise ValueError('Misaligned slot_next: covariate does not describe the target bar')
        if not (d.dow_next.to_numpy()[resolve]==d.trade_date.dt.dayofweek.to_numpy()[ti]).all():
            raise ValueError('Misaligned dow_next: covariate does not describe the target bar')
    d['label_ok']=resolve
    if not (d.loc[resolve,'target_end']>d.loc[resolve,'bar_end_utc']).all():raise ValueError('Noncausal target timestamp')
    if resolve.any():
        ti=tgt_row[resolve]
        discrepancy=np.max(np.abs(d.loc[resolve,'r_next'].to_numpy()-np.log(d.close_mid.to_numpy()[ti]/d.close_mid.to_numpy()[resolve])))
        if discrepancy>1e-9: raise ValueError(f'Target identity fails: {discrepancy}')
    counts=d.groupby('trade_date').slot.agg(['size','nunique'])
    complete=(counts['size']==5)&(counts['nunique']==5)
    sparse=(len(complete)<3 or not complete.iloc[1:-1].all()) or (d.bar_seq.diff().iloc[1:]!=1).any()
    if sparse and not allow_sparse:
        raise ValueError('STRICT GEOMETRY FAILED: incomplete sessions / missing bar_seq. Supply full causal feature history and explicit target timestamps. Use --allow-sparse-research only for NONCOMPLIANT research.')
    d.attrs['mode']='sparse_research_NONCOMPLIANT' if sparse else 'strict'
    d.attrs['dataset_sha256']=sha(path)
    return d

def audit(path):
    d=load(path,True,False)
    counts=d.groupby('trade_date').slot.agg(['size','nunique'])
    r={'dataset_sha256':sha(path),'scope':'pre-2026 only; lockbox outcomes excluded',
       'rows_development_evaluation':len(d),'columns':__import__('pyarrow.parquet',fromlist=['ParquetFile']).ParquetFile(path).schema_arrow.names,
       'start':str(d.trade_date.min().date()),'end':str(d.trade_date.max().date()),
       'sessions':len(counts),'complete_5_slot_sessions':int(((counts['size']==5)&(counts['nunique']==5)).sum()),
       'bar_seq_breaks':int((d.bar_seq.diff().iloc[1:]!=1).sum()),
       'resolvable_target_timestamps':int(d.label_ok.sum()),
       'unresolved_targets_excluded':int((~d.label_ok).sum()),
       'strict_geometry_pass':d.attrs['mode']=='strict',
       'features':d[FEATURES].describe(percentiles=[.01,.25,.5,.75,.99]).to_dict(),
       'FE_files':[str(p) for p in __import__('pathlib').Path('FE').rglob('*') if p.is_file()]}
    dump(ART/'data_audit.json',r)
    return r

class FoldTransform:
    def fit(self,d,train):
        from sklearn.preprocessing import RobustScaler
        self.scaler=RobustScaler().fit(d.iloc[train][FEATURES])
        t=d.iloc[train]; u=t.r_next.to_numpy()/t.sigma_ewma.to_numpy()
        self.factors=np.array([np.sqrt(np.mean(u[t.slot_next.to_numpy()==s]**2)) for s in range(5)])
        if not np.isfinite(self.factors).all() or (self.factors<=0).any(): raise ValueError('Train missing slot or zero seasonal RMS')
        self.factors/=np.exp(np.log(self.factors).mean())
        self.train_indices=np.asarray(train); return self
    def apply(self,d):
        x=np.clip(self.scaler.transform(d[FEATURES]),-5,5).astype('float32')
        scale=d.sigma_ewma.to_numpy()*self.factors[d.slot_next.to_numpy()]
        return x,scale,(d.r_next.to_numpy()/scale).astype('float32')
    def provenance(self):
        return dict(train_indices=self.train_indices.tolist(),median=self.scaler.center_.tolist(),
                    iqr=self.scaler.scale_.tolist(),slot_factors=self.factors.tolist())

def eligible(d,L,strict=False):
    ok=d.label_ok.to_numpy().copy();ok[:L-1]=False
    if strict:
        seq=d.bar_seq.to_numpy()
        good=np.zeros(len(d),bool)
        good[L-1:]=seq[L-1:]-seq[:len(d)-L+1]==L-1
        ok &=good
    return ok
