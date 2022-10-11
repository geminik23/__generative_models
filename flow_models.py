import torch
import torch.nn as nn

class RealNVP(nn.Module):
    def __init__(self, latent_dim, num_blocks, scale_net_func, transition_net_func):
        super().__init__()
        self.num_blocks = num_blocks
        self.latent_dim = latent_dim

        self.prior = torch.distributions.MultivariateNormal(torch.zeros(latent_dim), torch.eye(latent_dim))

        # transition networks
        self.t = nn.ModuleList([transition_net_func() for _ in range(num_blocks)])
        # scale networks
        self.s = nn.ModuleList([scale_net_func() for _ in range(num_blocks)])


    def coupling(self, x, i, forward=True):
        # devide into xa, xb
        (xa, xb) = torch.chunk(x, 2, 1)

        s = self.s[i](xa)
        t = self.t[i](xa)

        if forward: 
            yb = (xb - t) * torch.exp(-s)
        else: 
            yb = torch.exp(s) * xb + t

        # s for log determinant jacobian
        return torch.cat((xa, yb), 1), s

    def permute(self, x):
        # used simple operation : flip
        return x.flip(1) 

    def _f(self, x):
        """
        forward transformation.
        """
        
        # the log determinant jacobian can be calculated by summing the scale of x_a
        ldj = x.new_zeros(x.size(0)) # [B] -> single value
        
        for i in range(self.num_blocks):
            x, s = self.coupling(x, i, True)
            x = self.permute(x)
            ldj = ldj - s.sum(dim=1)
        return x, ldj

    def _inv_f(self, x):
        """
        inverse forward transformation.
        while sampling, it does not calculate the log determinant of jacobian
        """
        for i in reversed(range(self.num_blocks)):
            x = self.permute(x)
            x, _ = self.coupling(x, i, False)
        return x
    
    def forward(self, input):
        # dequantization
        device = input.device
        x = input + (1. - torch.rand(input.shape).to(device))/2.
        x, ldj = self._f(x)
        # when device is cuda:0
        return -(self.prior.log_prob(x.to('cpu')).to(device) + ldj).mean()
    
    def inference(self, batch_size, device):
        x = self.prior.sample((batch_size, )).to(device)
        return self._inv_f(x)
        
