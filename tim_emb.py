import math

import torch
from torch import nn
#目标 把形状为([batch_size])的时间步 embedding 为 [batch_size,emb_size]
class TimeEmbedding(nn.Module):
    def __init__(self,emb_size):
        super().__init__()
        self.half_size=emb_size//2
        half_emb=torch.exp(torch.arange(self.half_size)*(-1*math.log(10000)/(self.half_size-1)))
        self.register_buffer('half_emb1',half_emb)
    def forward(self,t):
        t=t.view(t.size(0),1)
        half_emb=self.half_emb1.unsqueeze(0).expand(t.size(0),self.half_size)
        half_emb_t=half_emb*t
        emb_t=torch.cat((half_emb_t.sin(),half_emb_t.cos()),dim=-1)
        return emb_t

