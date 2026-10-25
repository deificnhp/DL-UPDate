import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import stack_module_state,functional_call,vmap
from .config import TAUS

def quantiles(raw):
    raw=raw.float();m=raw[...,0:1]; inc=F.softplus(raw[...,1:])
    upper=m+inc[...,:3].cumsum(-1)
    lower=(m-inc[...,3:].cumsum(-1)).flip(-1)
    return torch.cat([lower,m,upper],-1)

class Head(nn.Module):
    def __init__(self,dim):
        super().__init__();self.slot=nn.Embedding(5,8);self.dow=nn.Embedding(5,4)
        self.mlp=nn.Sequential(nn.Linear(dim+12,64),nn.GELU(),nn.Linear(64,7))
    def forward(self,x,c): return quantiles(self.mlp(torch.cat([x,self.slot(c[:,0]),self.dow(c[:,1])],-1)))

class Block(nn.Module):
    def __init__(self,d,h,drop,adrop):
        super().__init__();self.h=h; self.adrop=adrop;self.drop=drop
        self.norm1=nn.LayerNorm(d);self.norm2=nn.LayerNorm(d)
        self.qkv=nn.Linear(d,3*d);self.proj=nn.Linear(d,d)
        self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Dropout(drop),nn.Linear(2*d,d),nn.Dropout(drop))
    def forward(self,x):
        b,t,d=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(b,t,3,self.h,d//self.h).permute(2,0,3,1,4).unbind(0)
        y=F.scaled_dot_product_attention(q,k,v,dropout_p=self.adrop if self.training else 0.)
        x=x+F.dropout(self.proj(y.transpose(1,2).reshape(b,t,d)),self.drop,self.training)
        return x+self.ff(self.norm2(x))

class PatchTST(nn.Module):
    def __init__(self,cfg,Fdim=11):
        super().__init__();L=cfg['lookback'];d=cfg['d_model']
        assert L%5==0 and d%cfg['n_heads']==0
        self.embed=nn.Linear(5*Fdim,d);self.position=nn.Parameter(torch.randn(1,L//5,d)*.02)
        self.blocks=nn.ModuleList([Block(d,cfg['n_heads'],cfg['dropout'],cfg.get('attn_dropout',.1)) for _ in range(cfg['n_layers'])])
        self.norm=nn.LayerNorm(d);self.head=Head(2*d)
    def forward(self,x,c):
        b,l,f=x.shape;x=self.embed(x.reshape(b,l//5,5*f))+self.position
        for block in self.blocks:x=block(x)
        x=self.norm(x)
        return self.head(torch.cat([x[:,-1],x.mean(1)],-1),c)

class Ensemble(nn.Module):
    """Independent stacked parameters. Explicit K-batched contractions execute every member.
    Initialization loops create states, NEVER sequential training/inference member loops.
    """
    def __init__(self,ctor,cfg,device='cuda'):
        super().__init__();import copy
        models=[]
        for k in range(cfg['K']):
            torch.manual_seed(cfg['seed']+k);models.append(ctor(cfg).to(device))
        params,buffers=stack_module_state(models)
        self.params=nn.ParameterDict({k.replace('.','__'):nn.Parameter(v) for k,v in params.items()})
        self.keys=list(params);self.buffers_map=buffers
        self.template=copy.deepcopy(models[0]).to('meta');del models
    def forward(self,x,c):
        self.template.train(self.training)
        p={k:self.params[k.replace('.','__')] for k in self.keys}
        # Explicit K-batched contractions avoid backend vmap fallbacks in SDPA/LSTM.
        # stack_module_state is used for independent parameter initialization only.
        return batched_patch(self.template,p,x,c,self.training) if isinstance(self.template,PatchTST) else batched_lstm(self.template,p,x,c,self.training)

def linear(p,key,x):
    K=x.shape[0];shape=x.shape[:-1]
    weight=p[key+'.weight'];flat=x.reshape(K,-1,x.shape[-1])
    if key+'.bias' in p:
        out=torch.baddbmm(p[key+'.bias'][:,None,:],flat,weight.transpose(1,2))
    else:out=torch.bmm(flat,weight.transpose(1,2))
    return out.reshape(*shape,weight.shape[1])

def norm(p,key,x):
    # FP32 statistics, independent affine parameters per K, cast back for AMP.
    var,mu=torch.var_mean(x.float(),dim=-1,correction=0,keepdim=True)
    y=((x.float()-mu)*torch.rsqrt(var+1e-5)).to(x.dtype)
    extra=(1,)*(x.ndim-2)
    w=p[key+'.weight'].reshape(x.shape[0],*extra,x.shape[-1]).to(x.dtype)
    b=p[key+'.bias'].reshape(x.shape[0],*extra,x.shape[-1]).to(x.dtype)
    return y*w+b

def batched_head(p,x,c):
    k=torch.arange(x.shape[0],device=x.device)[:,None]
    slot=p['head.slot.weight'][k,c[:,:,0]];dow=p['head.dow.weight'][k,c[:,:,1]]
    z=torch.cat([x,slot,dow],-1)
    return quantiles(linear(p,'head.mlp.2',F.gelu(linear(p,'head.mlp.0',z))))

def batched_patch(template,p,x,c,training):
    K,B,L,Fdim=x.shape;T=L//5
    x=linear(p,'embed',x.reshape(K,B,T,Fdim*5))+p['position']
    for i,block in enumerate(template.blocks):
        pre=f'blocks.{i}';D=x.shape[-1];H=block.h
        qkv=linear(p,pre+'.qkv',norm(p,pre+'.norm1',x))
        q,k,v=qkv.reshape(K,B,T,3,H,D//H).permute(3,0,1,4,2,5).unbind(0)
        # ONE fused attention invocation over all K*B sequences, never member loops.
        a=F.scaled_dot_product_attention(q.reshape(K*B,H,T,D//H),k.reshape(K*B,H,T,D//H),
              v.reshape(K*B,H,T,D//H),dropout_p=block.adrop if training else 0.)
        a=a.reshape(K,B,H,T,D//H).transpose(2,3).reshape(K,B,T,D)
        x=x+F.dropout(linear(p,pre+'.proj',a),block.drop,training)
        z=norm(p,pre+'.norm2',x)
        z=F.dropout(F.gelu(linear(p,pre+'.ff.0',z)),block.drop,training)
        x=x+F.dropout(linear(p,pre+'.ff.3',z),block.drop,training)
    x=norm(p,'norm',x)
    return batched_head(p,torch.cat([x[:,:,-1],x.mean(2)],-1),c)

def batched_lstm(template,p,x,c,training):
    K,B,L,_=x.shape
    states=[(x.new_zeros(K,B,64),x.new_zeros(K,B,64)) for _ in range(2)]
    total=x.new_zeros(K,B,64)
    for t in range(L):
        z=x[:,:,t]
        for j in range(2):
            h,cell=states[j]
            i,f,g,o=(linear(p,f'cells.{j}.x',z)+linear(p,f'cells.{j}.h',h)).chunk(4,-1)
            cell=torch.sigmoid(f)*cell+torch.sigmoid(i)*torch.tanh(g)
            h=torch.sigmoid(o)*torch.tanh(cell);states[j]=(h,cell)
            z=F.dropout(h,template.drop,training) if j==0 else h
        total=total+h
    return batched_head(p,torch.cat([h,total/L],-1),c)

def pinball(q,y):
    tau=torch.tensor(TAUS,device=q.device,dtype=torch.float32)
    e=y.float().unsqueeze(-1)-q.float()
    # Sum member objectives so increasing K does NOT shrink member gradients.
    return torch.maximum(tau*e,(tau-1)*e).mean(dim=(-1,-2)).sum()
