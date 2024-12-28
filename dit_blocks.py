import math

import torch
from torch import nn


class DITBLOCKS(nn.Module):
    def __init__(self, emb_size, heads):
        super().__init__()
        self.nhead = heads
        self.ln = nn.LayerNorm(emb_size)
        self.gamma1 = nn.Linear(emb_size, emb_size)
        self.beta1 = nn.Linear(emb_size, emb_size)
        self.alpha1 = nn.Linear(emb_size, emb_size)
        self.gamma2 = nn.Linear(emb_size, emb_size)
        self.beta2 = nn.Linear(emb_size, emb_size)
        self.alpha2 = nn.Linear(emb_size, emb_size)
        self.wq = nn.Linear(emb_size, heads * emb_size)
        self.wk = nn.Linear(emb_size, heads * emb_size)
        self.wv = nn.Linear(emb_size, heads * emb_size)
        self.linear1 = nn.Linear(heads * emb_size, emb_size)
        self.ln1 = nn.LayerNorm(emb_size)
        self.feedforward = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.ReLU(),
            nn.Linear(emb_size * 4, emb_size)
        )

    def forward(self, x, cond):  # (batch_size fenkuai*fenkuai emb_size) cond (batch_size emb_size)
        y=x
        x = self.ln(x)
        gm1 = self.gamma1(cond)
        gm2 = self.gamma2(cond)
        bt1 = self.beta1(cond)
        bt2 = self.beta2(cond)
        ap1 = self.alpha1(cond)
        ap2 = self.alpha2(cond)
        x = x * (1 + gm1.unsqueeze(1)) + bt1.unsqueeze(1)
        wq = self.wq(x)
        wk = self.wk(x)
        wv = self.wv(x)  # (batch_size fenkuai*fenkuai heads*emb_size)
        wq = wq.view(wq.size(0), wq.size(1), self.nhead, -1).permute(0, 2, 1, 3)  # (batch_size  heads fenkuai*fenkuai   emb_size)
        wk = wk.view(wk.size(0), wk.size(1), self.nhead, -1).permute(0, 2, 3, 1)  # (batch_size  heads  emb_size  fenkuai*fenkuai)
        wv = wv.view(wv.size(0), wv.size(1), self.nhead, -1).permute(0, 2, 1, 3)  # (batch_size  heads fenkuai*fenkuai   emb_size)
        attn = wq @ wk/math.sqrt(wq.size(2)) #(batch_size heads fenkuai*fenkuai fenkuai*fenkuai)
        attn = torch.softmax(attn,dim=-1)
        attn = attn @ wv #(batch_size  heads fenkuai*fenkuai   emb_size)
        attn  = attn.permute(0,2,1,3)
        attn = attn.reshape(attn.size(0),attn.size(1),attn.size(2)*attn.size(3)) #(batch_size fenkuai*fenkuai heads*emb_size)
        x = self.linear1(attn) #(batch_size fenkuai*fenkuai emb_size)
        x = x*ap1.unsqueeze(1)
        x = x + y
        z = x
        x = self.ln1(x)
        x = x * (1 + gm2.unsqueeze(1)) + bt2.unsqueeze(1)
        x = self.feedforward(x)
        x = x * ap2.unsqueeze(1)
        x = x + z
        return x


