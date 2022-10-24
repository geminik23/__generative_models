import unittest

import torch
import torch.nn as nn
from arm_model import CasualConv1d, AutoregressiveModel
from diffusion_model import DiffusionLatentModel


from functools import partial


class DiffTestCase(unittest.TestCase):

    def _q_net(self, dim):
        return nn.Sequential(nn.Linear(dim, dim*2)) # dim*2 -> mu, log_var

    def _decode_net(self, dim):
        return nn.Sequential(nn.Linear(dim, dim), nn.Tanh())

    
    def test_diffusion_model(self):
        batch_size = 1
        dim = 8

        x = torch.randn((batch_size, dim)).tanh_()
        model = DiffusionLatentModel(dim, 3, partial(self._q_net, dim), partial(self._decode_net, dim))

        # data
        y = model(x)
        

        # inference
        x_0 = model.inference(batch_size, 'cpu')
        self.assertEqual(torch.Size((batch_size, dim)), x_0.shape)


    def test_log_likelihood_of_norm(self):
        # as one-dim markov
        # log-likelihood function
        
        x = torch.randn((1, 1000))

        mu = 0.0
        var = torch.tensor([1.0])

        # full eq
        log_p = -0.5 * torch.tensor(2*torch.pi).log_() - 1.0/(2*var) * (x-torch.tensor([mu]))**2. - 0.5*torch.log(var) 
        # reduced for 0 mean and 1 var.
        log_p_reduced = -0.5 * torch.tensor(2*torch.pi).log_() - 0.5 * (x)**2. # - 0.5*torch.log(torch.tensor([var])) #=0
        self.assertTrue(torch.equal(log_p, log_p_reduced))


class ARMMTestCase(unittest.TestCase):
    def setUp(self):
        NUM_CLASS = 17
        HIDDEN_DIM = 512

        self.net = nn.Sequential(
            CasualConv1d(1, HIDDEN_DIM, dilation=1, kernel_size = 2, exclude_last=True, bias=True),
            nn.LeakyReLU(),
            CasualConv1d(HIDDEN_DIM, HIDDEN_DIM, dilation=1, kernel_size = 2, exclude_last=False, bias=True),
            nn.LeakyReLU(),
            CasualConv1d(HIDDEN_DIM, HIDDEN_DIM, dilation=1, kernel_size = 2, exclude_last=False, bias=True),
            nn.LeakyReLU(),    
            CasualConv1d(HIDDEN_DIM, NUM_CLASS, dilation=1, kernel_size = 2, exclude_last=False, bias=True)
        )

    def test_causalconv1d(self):
        DIM = 64
        NUM_CLASS = 17
        BATCH = 4
        ##
        ## FORWARD PASS
        # construct the network
        x = torch.randint(0, 15, (BATCH, DIM)).float()
        x = x.unsqueeze(1) # -> [B, C, D]
        y = self.net(x)
        self.assertEqual(y.shape, torch.Size([BATCH, NUM_CLASS, DIM]))

        ##
        ## SOFTMAX
        # apply the softmax
        y = y.permute(0,2,1) # -> [B, D, C]
        softmax = nn.Softmax(2)
        y = softmax(y)
        self.assertEqual(y.shape, torch.Size([BATCH, DIM, NUM_CLASS]))

        y = torch.argmax(y, dim=2)
        self.assertEqual(y.shape, torch.Size([BATCH, DIM]))

    def test_autoregressive_model(self):
        DIM = 64
        NUM_CLASS = 17
        BATCH = 4
        ##
        ## GET LOSS
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AutoregressiveModel(self.net, DIM, NUM_CLASS).to(device)
        
        x = torch.randint(0, 15, (BATCH, DIM), device=device).float()
        y = model(x)
        # y is scalar
        assert y.shape == torch.Size([])

        ##
        ## SAMPLING
        out = model.inference(1, device=device)
        assert out.shape == torch.Size([1, 64])


if __name__ == '__main__':
    unittest.main(verbosity=2)
