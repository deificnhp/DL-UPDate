import pandas as pd
import numpy as np
from .config import ART
from .evaluate import QCOLS
from .calibration import Calibrator

def frame(d,ids,q,model,year):
    f=pd.DataFrame(q,columns=QCOLS)
    f['row_id']=ids;f['bar_seq']=d.bar_seq.iloc[ids].to_numpy();f['model']=model;f['fold']=year
    f['decision_end']=d.bar_end_utc.iloc[ids].to_numpy();f['target_end']=d.target_end.iloc[ids].to_numpy()
    f['y']=d.r_next.iloc[ids].to_numpy();return f

def save(d,ids,q,model,year,scope,extras=None,variant=''):
    f=frame(d,ids,q,model,year)
    if extras:
        for k,v in extras.items():f[k]=v
    path=ART/'predictions'/scope/f'{year}_{model}{variant}.parquet';path.parent.mkdir(parents=True,exist_ok=True)
    f.to_parquet(path,index=False);return f

def calibrated(d,parts,outputs,model,year,scope):
    if len(parts['test'])>1:
        if not (d.target_end.iloc[parts['test'][:-1]].to_numpy()<=d.bar_end_utc.iloc[parts['test'][1:]].to_numpy()).all():
            raise ValueError('ACI outcome not available before next decision; delayed-feedback scheduling is required')
    c=Calibrator().fit(outputs['calibration'],d.r_next.iloc[parts['calibration']].to_numpy())
    save(d,parts['test'],outputs['test'].mean(0),model+'-uncalibrated',year,scope)
    q=c.apply(outputs['test'],online=True,y=d.r_next.iloc[parts['test']].to_numpy())
    save(d,parts['test'],q.mean(0),model,year,scope,c.diagnostics(q))
    folder=ART/'members'/scope;folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/f'{year}_{model}.npz',raw=outputs['test'],calibrated=q,row_ids=parts['test'])
    return c.metadata(),c.aci_trace
