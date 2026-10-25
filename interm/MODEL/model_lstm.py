"""Two-layer LSTM cells avoid the unsupported fused cuDNN LSTM vmap rule."""
import torch
from torch import nn
from torch.nn import functional as F
from .model_patchtst import Head
class Cell(nn.Module):
    def __init__(self,indim,h):
        super().__init__();self.x=nn.Linear(indim,4*h);self.h=nn.Linear(h,4*h,bias=False)
    def forward(self,x,h,c):
        i,f,g,o=(self.x(x)+self.h(h)).chunk(4,-1)
        c=torch.sigmoid(f)*c+torch.sigmoid(i)*torch.tanh(g)
        return torch.sigmoid(o)*torch.tanh(c),c
class LSTM(nn.Module):
    def __init__(self,cfg,Fdim=11):
        super().__init__();self.cells=nn.ModuleList([Cell(Fdim,64),Cell(64,64)]);self.drop=cfg['dropout'];self.head=Head(128)
    def forward(self,x,cov):
        states=[(x.new_zeros(x.shape[0],64),x.new_zeros(x.shape[0],64)) for _ in range(2)]
        total=x.new_zeros(x.shape[0],64)
        # Loops are over recurrent TIME and layers, never ensemble members.
        for t in range(x.shape[1]):
            z=x[:,t]
            for j,cell in enumerate(self.cells):
                h,c=cell(z,*states[j]);states[j]=(h,c)
                z=F.dropout(h,self.drop,self.training) if j==0 else h
            total=total+h
        return self.head(torch.cat([h,total/x.shape[1]],-1),cov)
