import torch
T=1000
betas = torch.linspace(0.0001,0.02,T)
alphas = 1-betas
multi_alpha = torch.cumprod(alphas,dim=0)
multi_pre_alpha = torch.cat((torch.tensor([1.0]),multi_alpha[:-1]),dim=0)
var = (1-alphas)*(1-multi_pre_alpha)/(1-multi_alpha)

def forward_add_noise(x,t):
    batch_size=x.size(0)
    ex_multi_alpha=multi_alpha[t].view(batch_size,1,1,1)
    noise = torch.randn_like(x)
    x = torch.sqrt(ex_multi_alpha)*x+torch.sqrt(1-ex_multi_alpha)*noise
    return x,noise