from typing import overload
import torch
import torch.nn as nn

# as learning objective, I used the ELBO equation derivation directly.
# \w
# In Section 4.3 [1], for progressive generation, 

# decoder -> (negative) MSEloss
# other parts ? 



# T = 3
# decoder, net, net (T-2)
# x(zo) -> zs = [z1 -> z2 -> z3]
#
## -- reverse process
# l2(z3) l1(z2) decoder(z1)->x_0


# q diffusion process
# p reverse process
class DiffusionLatentModel(nn.Module):
    def __init__(self, dim, T, p_net_func, decoder_func, betas=(1e-4, 0.02)):
        super().__init__()
        # beta values from original paper (1e-4 to 0.02) 

        self.dim = dim
        self.T = T

        # network is only for reverse process
        self.nets = nn.ModuleList([p_net_func() for _ in range(T-1)]) # **p_networks are in reverse order**
        self.decoder = decoder_func()
        self.beta = torch.linspace(betas[0], betas[1], T)
        self.alpha = 1.0 - self.beta
        

    @staticmethod
    def _log_likelihood_of_norm(x, mu=torch.tensor([0.]), logvar=torch.tensor([0.])):
        var = torch.exp(logvar)
        log_p = -0.5 * torch.tensor(2*torch.pi).log_() - 1.0 / (2*var) * (x-mu)**2. - 0.5*torch.log(var) 
        return log_p

    @staticmethod
    def _reparameterize(mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + std*eps

    def _gaussian_diffusion(self, x, t):
        return x*torch.sqrt(self.alpha[t]) + torch.sqrt(self.beta[t])*torch.randn_like(x)

    # diffusion process
    def q_process(self, x):
        zs = []
        z = x
        for t in range(self.T):
            z = self._gaussian_diffusion(z, t)
            zs.append(z)
        return zs

    # reverse process
    def p_process(self, zs):
        mus = []
        log_vars = []
        for i, layer in enumerate(self.nets):
            h = layer(zs[self.T-1-i])
            mu_i, log_var_i = torch.chunk(h, 2, dim=1)
            mus.append(mu_i)
            log_vars.append(log_var_i)
        x_0 = self.decoder(zs[0])
        return x_0, list(reversed(mus)), list(reversed(log_vars))

    def forward(self, x):
        # forward process
        zs = self.q_process(x) # len(zs) = self.T

        device = x.device

        # backward process
        x_0, mus, log_vars = self.p_process(zs) # len(T-1)

        # for the convenience of computation
        mus.append(torch.full_like(mus[0], 0.0).to(device))
        log_vars.append(torch.full_like(log_vars[0], 0.0).to(device)) # exp(log_var=0) = 1

        ## ELBO
        L_0 = self._log_likelihood_of_norm(x-x_0, mus[-1], log_vars[-1]).sum(-1)
        
        # KL[q(z_T|z_(T-1)||p(z_T)]  + SIGMA_i KL[(1, T-1) q(z_i | z_(i-1)) || p(z_i|z_(i+1))]
        KLs = []
        for i in range(self.T-1, -1, -1):
            KLs.append((self._log_likelihood_of_norm(zs[i], torch.sqrt(self.alpha[i])*zs[i], torch.log(self.beta[i])) - self._log_likelihood_of_norm(zs[i], mus[i], log_vars[i])).sum(-1))
        KLs = torch.stack(KLs).sum()

        # negative log-likelihood
        loss = -(L_0 - KLs).mean()
        return loss


    def inference(self, batch_size, device):
        z = torch.randn([batch_size, self.dim]).to(device)
        for layer in self.nets:
            h = layer(z)
            mu_i, log_var_i = torch.chunk(h, 2, dim=1)
            z = self._reparameterize(torch.tanh(mu_i), log_var_i)
        return (self.decoder(z)+2.0)/2.0