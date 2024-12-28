import torch
from torch import nn

from dit_blocks import DITBLOCKS
from tim_emb import TimeEmbedding
class DIT(nn.Module):
    def __init__(self,img_size,channel,emb_size,heads,numm,patch_size,labels):
        super().__init__()
        self.patchsize=patch_size
        self.fenkuai=img_size//patch_size
        self.channel = channel
        self.conv=nn.Conv2d(in_channels=channel,out_channels=channel*patch_size**2,kernel_size=self.patchsize,padding=0,stride=patch_size)
        self.patch_emb=nn.Linear(in_features=self.patchsize**2,out_features=emb_size)
        self.patch_pos_emd=nn.Parameter(torch.rand(1,self.fenkuai**2,emb_size))
        self.label_emb=nn.Embedding(num_embeddings=labels,embedding_dim=emb_size)
        self.time_emd = nn.Sequential(
            TimeEmbedding(emb_size),
            nn.Linear(emb_size,emb_size),
            nn.ReLU(),
            nn.Linear(emb_size,emb_size),
        )
        self.dits=nn.ModuleList()
        for _ in range(numm):
            self.dits.append(DITBLOCKS(emb_size,heads))
        self.ln=nn.LayerNorm(emb_size)
        self.linear=nn.Linear(emb_size,self.patchsize**2*channel)
    def forward(self,x,t,y): #x:(batch_size channel img_size img_size)
        y_emb=self.label_emb(y)
        t_emb=self.time_emd(t)
        cond=y_emb+t_emb
        #print(x.size())
        x=self.conv(x) #(batch_size,patch_size**2*channel,fenkuai,fenkuai)
        #print(x.size())
        x=x.permute(0,2,3,1) #(batch_size fenkuai fenkuai patch_size**2*channel)
        x = x.view(x.size(0), x.size(1)*x.size(1), x.size(3))
        #print(x.size())
        x=self.patch_emb(x) #(batch_size fenkuai*fenkuai emb_size)
        x=x+self.patch_pos_emd
        for dit in self.dits:
            x=dit(x,cond)
        x=self.ln(x)
        #print(x.size())
        x=self.linear(x) #(batch_size fenkuai*fenkuai patch_size**2*channel)
        x=x.view(x.size(0),self.fenkuai,self.fenkuai,self.channel,self.patchsize,self.patchsize)  # (batch,fenkuai,fenkuai,channel,patch_size,patch_size)
        x=x.permute(0,3,1,2,4,5)    # (batch,channel,fenkuai(H),fenkuai(W),patch_size(H),patch_size(W))
        x=x.permute(0,1,2,4,3,5)    # (batch,channel,fenkuai(H),patch_size(H),fenkuai(W),patch_size(W))
        x=x.reshape(x.size(0),self.channel,self.fenkuai*self.patchsize,self.fenkuai*self.patchsize)   # (batch,channel,img_size,img_size)
        return x

if __name__ == '__main__':
    x= torch.rand(100,7,7,16)
    print(x.size())
    x = x.view(x.size(0), x.size(1) * x.size(1), x.size(3))
    print(x.size())