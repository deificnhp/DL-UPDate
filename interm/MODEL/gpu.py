import torch
import numpy as np
from contextlib import nullcontext
from .model_patchtst import Ensemble,PatchTST,pinball
from .model_lstm import LSTM
from .data import FoldTransform

def require_cuda():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required. CPU execution is available only for unit tests.')
    if torch.cuda.get_device_capability()<(7,5):raise RuntimeError('Requires Turing CC 7.5 or newer')

def stationary_indices(n,K,B,device='cuda',mean_block=25):
    """Independent circular stationary bootstrap for every member (geometric blocks)."""
    restart=torch.rand(K,B,device=device)<1/mean_block;restart[:,0]=True
    origins=torch.randint(n,(K,B),device=device)
    pos=torch.arange(B,device=device).expand(K,-1)
    latest=torch.where(restart,pos,torch.zeros_like(pos)).cummax(1).values
    return (origins.gather(1,latest)+pos-latest)%n

def autocast():
    if not torch.cuda.is_available():return nullcontext()
    return torch.autocast('cuda',dtype=torch.bfloat16 if torch.cuda.get_device_capability()[0]>=8 else torch.float16)

class Resident:
    def __init__(self,d,parts,cfg,feature_mask=None):
        require_cuda()
        # Active-fold residency only: never allocate later evaluation years as padding.
        last=max(int(ids[-1]) for ids in parts.values())
        d=d.iloc[:last+1].copy()
        self.transform=FoldTransform().fit(d,parts['train'])
        x,scale,z=self.transform.apply(d)
        if feature_mask is not None:x=x[:,feature_mask]
        self.L=cfg['lookback']; self.K=cfg['K']
        # Materialized windows stay entirely resident for the fold lifetime.
        self.features=torch.as_tensor(x,device='cuda');self.scale=torch.as_tensor(scale,device='cuda',dtype=torch.float32)
        self.windows=self.features.unfold(0,self.L,1).permute(0,2,1).contiguous()
        self.cat=torch.as_tensor(d[['slot_next','dow_next']].to_numpy(),device='cuda',dtype=torch.long)
        self.z=torch.as_tensor(z,device='cuda');self.parts={k:torch.as_tensor(v,device='cuda',dtype=torch.long) for k,v in parts.items()}
    def batch(self,idx):return self.windows[idx-self.L+1],self.cat[idx],self.z[idx]

@torch.no_grad()
def predict(model,data,part,B):
    model.eval();idx=data.parts[part];chunks=[]
    for ids in idx.split(B):
        x,c,_=data.batch(ids)
        with autocast():q=model(x.unsqueeze(0).expand(data.K,-1,-1,-1),c.unsqueeze(0).expand(data.K,-1,-1))
        chunks.append((q.float()*data.scale[ids][None,:,None]).cpu().numpy())
    return np.concatenate(chunks,axis=1)

def fit(d,parts,cfg,architecture='patchtst',feature_mask=None,checkpoint=None,stability=False,enforce_memory_band=False):
    import gc,time
    require_cuda();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    free,total=torch.cuda.mem_get_info();started=time.perf_counter()
    torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    torch.set_float32_matmul_precision('high')
    data=Resident(d,parts,cfg,feature_mask)
    ctor=PatchTST if architecture=='patchtst' else LSTM
    f=len(feature_mask) if feature_mask is not None else 11
    model=Ensemble(lambda c:ctor(c,Fdim=f),cfg)
    optimizer=torch.optim.AdamW(model.params.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=torch.cuda.get_device_capability()[0]<8)
    best=float('inf');state=None;wait=0;logs=[]
    tr=data.parts['train'];steps=max(1,(len(tr)+cfg['B']-1)//cfg['B'])
    for epoch in range(3 if stability else cfg['epochs']):
        epoch_started=time.perf_counter();model.train();losses=[]
        for _ in range(steps):
            ids=tr[stationary_indices(len(tr),cfg['K'],cfg['B'])];x,c,y=data.batch(ids)
            optimizer.zero_grad(set_to_none=True)
            with autocast():q=model(x,c)
            loss=pinball(q,y)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            # Clip independently per member (not globally across K).
            norms=torch.zeros(cfg['K'],device='cuda')
            for p in model.params.values():
                if p.grad is not None:norms+=p.grad.float().flatten(1).square().sum(1)
            factors=(1./norms.sqrt().clamp_min(1.)).detach()
            for p in model.params.values():
                if p.grad is not None:p.grad.mul_(factors.reshape((-1,)+(1,)*(p.ndim-1)))
            scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()/cfg['K']))
        v=predict(model,data,'validation',max(1,cfg['B']//2)).mean(0)
        if stability:
            calibration_stability=predict(model,data,'calibration',max(1,cfg['B']//2))
            del calibration_stability
        from .evaluate import loss_rows
        vl=float(loss_rows(d.r_next.iloc[parts['validation']].to_numpy(),v).mean())
        logs.append(dict(epoch=epoch,train_scaled_pinball=float(np.mean(losses)),validation_raw_pinball=vl,wall_seconds=time.perf_counter()-epoch_started,
                         peak_reserved_bytes_so_far=torch.cuda.max_memory_reserved(),
                         peak_allocated_bytes_so_far=torch.cuda.max_memory_allocated()))
        print(architecture,epoch,logs[-1],flush=True)
        if vl<best:
            best=vl;state={k:v.detach().cpu().clone() for k,v in model.params.items()};wait=0
        else:wait+=1
        if not stability and wait>=cfg['patience']:break
    model.params.load_state_dict(state)
    if checkpoint:
        torch.save(dict(params=state,config=cfg,architecture=architecture,transform=data.transform.provenance(),logs=logs),checkpoint)
    outputs={p:predict(model,data,p,max(1,cfg['B']//2)) for p in ['validation','calibration','test']}
    torch.cuda.synchronize()
    memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                free_bytes_before_fold=free,total_bytes=total,wall_seconds=time.perf_counter()-started,
                device=torch.cuda.get_device_name(),compute_capability=list(torch.cuda.get_device_capability()))
    if enforce_memory_band:
        upper=min(5.30e9,free-.70e9);lower=min(4.80e9,max(0,upper-.10e9))
        if not lower<=memory['peak_reserved_bytes']<=upper:
            raise RuntimeError(f'Fold memory outside certified band [{lower},{upper}]: {memory}')
    return outputs,dict(logs=logs,best_validation=best,transform=data.transform.provenance(),memory=memory)
