import torch
import torch.nn as nn
import torch.nn.functional as F


class CasualConv1d(nn.Module):
    """
    Casual 1D convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, exclude_last=False, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.E = exclude_last
        
        self.padding = (kernel_size - 1)*dilation + exclude_last*1
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=dilation, **kwargs)
        
    def forward(self, x):
        x = nn.functional.pad(x, (self.padding, 0))
        out = self.conv1d(x)
        if self.E:
            return out[:, :, :-1] # exclude last sample
        else:
            return out


EPS = 1.e-6
class AutoregressiveModel(nn.Module):
    def __init__(self, net, dim, num_classes):
        super().__init__()
        self.net = net
        self.dim = dim
        self.num_classes = num_classes

    def _f(self, x):
        # insert channel dim
        y = self.net(x.unsqueeze(1)) # [B, D] -> [B, C, D]
        return y
        
    def forward(self, x):
        # x : [B, D]
        # logits : [B, C, D]
        logits = self._f(x) # [B, D] -> [B, C, D]
        return F.cross_entropy(logits, x.long())
        
    def inference(self, batch_size, device):
        out = torch.zeros((batch_size, self.dim)).to(device)

        for dim in range(self.dim):
            d = self._f(out)
            d = F.softmax(d, dim=1)
            # new_out = torch.argmax(d[:, :, dim], dim=1) # all same images
            new_out = torch.multinomial(d[:, :, dim], num_samples=1)
            out[..., dim] = new_out[...,0]
        return out


def create_network(hidden_dim, out_dim, kernel_size, num_layers=3):
    dil = 1
    is_first = True
    net = []
    net.append(nn.Conv1d(1, hidden_dim, 1))
    net.append(nn.LeakyReLU())
    for l in range(num_layers):
        net.append(CasualConv1d(hidden_dim, hidden_dim, dilation=dil, kernel_size = kernel_size, exclude_last=is_first, bias=True))
        net.append(nn.LeakyReLU())
        dil *= 2
        is_first = False

    net.append(CasualConv1d(hidden_dim, out_dim, dilation=dil, kernel_size = kernel_size, exclude_last=is_first, bias=True))

    return nn.Sequential(*net)

