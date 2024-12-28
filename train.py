import os

import torch
from torch.utils.data import DataLoader
from torch import nn

from diffusion import forward_add_noise
from dit import DIT
from dateset import MNIST
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#print(DEVICE)
dataset=MNIST()
model =DIT(img_size=28,channel=1,emb_size=64,heads=4,numm=4,patch_size=4,labels=10).to(DEVICE)
try:
    model.load_state_dict(torch.load('model.pth'))
except:
    pass
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
lossfn=nn.L1Loss()
dataloader = DataLoader(dataset,batch_size=100,shuffle=True,num_workers=10,persistent_workers=True)
epochs = 500
T=1000
if __name__ == '__main__':
    model.train()
    iter_count=0
    for epoch in range(epochs):
        for img,label in dataloader:
            x=img*2-1
            y=label
            t=torch.randint(0,T,(img.size(0),))
            x,noise = forward_add_noise(x,t)
            predict_noise=model(x.to(DEVICE),t.to(DEVICE),y.to(DEVICE))
            loss=lossfn(noise.to(DEVICE),predict_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter_count %1000 ==0:
                print('echo:{} iter:{} loss:{}'.format(epoch,iter_count,loss))
                torch.save(model.state_dict(),'.model.pth')
                os.replace('.model.pth','model.pth')
            iter_count = iter_count+1


