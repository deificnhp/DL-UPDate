from pathlib import Path
import argparse, json, hashlib, os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'MODEL/ARTIFACTS'
ART.mkdir(parents=True, exist_ok=True)
FEATURES = ['r1_z','r5_z','r30_z','gap_z','lgk_bar','lrv_d','lrv_w','lrv_m','clv','ema_dist','rvol']
TAUS = [.05,.10,.25,.50,.75,.90,.95]
DEFAULT = dict(lookback=150,d_model=64,n_heads=4,n_layers=3,dropout=.2,attn_dropout=.1,
               lr=.001,weight_decay=.0001,epochs=60,patience=8,K=16,B=512,seed=1729)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def tracked_sources(root=None):
    """Every source file a freeze must cover, as repo-root-relative POSIX paths.
    Covers the modeling package (MODEL/), the feature engineering layer (FE/) and the
    root entry points. Files under a 'tests' directory are excluded: they cannot change
    a result, so editing a test must not invalidate a freeze."""
    root=Path(root) if root else ROOT
    files=set()
    for sub in ('MODEL','FE'):
        d=root/sub
        if d.is_dir():
            files.update(p for p in d.rglob('*.py') if p.is_file() and 'tests' not in p.parts)
    files.update(p for p in root.glob('*.py') if p.is_file())
    return sorted(files,key=lambda p:p.relative_to(root).as_posix())
def code_manifest(root=None):
    """{root-relative path: sha256} for every tracked source file."""
    root=Path(root) if root else ROOT
    return {p.relative_to(root).as_posix():sha(p) for p in tracked_sources(root)}
def dump(path,obj):
    def clean(v):
        import numpy as np
        if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)): return [clean(x) for x in v]
        if isinstance(v,np.ndarray): return clean(v.tolist())
        if isinstance(v,np.generic): return clean(v.item())
        if isinstance(v,float) and not __import__('math').isfinite(v): return None
        if isinstance(v,Path): return str(v)
        return v
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(clean(obj),indent=2,allow_nan=False));temp.replace(path)
def parser(description=''):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--data',type=Path,default=ROOT/'DATA/PROCESSED/features_targets.parquet')
    p.add_argument('--allow-sparse-research',action='store_true',help='Noncompliant research only: retained-bar windows; never passes strict QA.')
    return p
def config():
    p=ART/'frozen_config_m3.json'
    if not p.exists(): raise RuntimeError('M3 configuration missing; run HPO first.')
    return json.loads(p.read_text())
