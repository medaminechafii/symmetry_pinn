import torch
torch.set_default_dtype(torch.float64)
class PolyG(torch.nn.Module):
    """g(u) = sum_{k=1...5} c_k u^k g(0) = 0 by construction"""
    def __init__(self,k_max = 5):
        super().__init__()
        self.k_max = k_max
        self.c = torch.nn.Parameter(torch.zeros(k_max))
    def forward(self,u):
        u_pow = torch.stack([u**k for k in range(1,self.k_max+1)],dim=-1)
        g = (u_pow*self.c).sum(-1)
        return g
class MLPG(torch.nn.Module):
    """g(u) = net(u) - net(0),  net: 1 -> 32 -> 32 -> 1, tanh"""
    def __init__(self,width=32,depth=2):
        super().__init__()
        layers = []
        n_in = 1
        for i in range(depth):
            layers+=[torch.nn.Linear(n_in,width),torch.nn.Tanh()]
            n_in = width
        layers+=[torch.nn.Linear(width,1)]
        self.net = torch.nn.Sequential(*layers)
    
    def forward(self,u):
        out = torch.squeeze(self.net(u.unsqueeze(-1)),-1)
        diff = out - torch.squeeze(self.net(torch.zeros(1,1)),-1)
        return diff


