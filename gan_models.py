import torch.nn as nn
import numpy as np

def _create_layer(in_neurons, out_neurons, leak=0.1):
    return nn.Sequential(
        nn.Linear(in_neurons, out_neurons),
        nn.LeakyReLU(leak),
        nn.LayerNorm(out_neurons) # for this model, it is better than BatchNorm
    )

class View(nn.Module):
    """
    change the shape of tensor
    """
    def __init__(self, shape:tuple):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)

def get_gan_model(latent_size, neurons, out_shape, leak=0.1):
    G = nn.Sequential(
        _create_layer(latent_size, neurons, leak),
        _create_layer(neurons, neurons, leak),
        nn.Linear(neurons, abs(np.prod(out_shape))),
        View(out_shape),
        nn.Sigmoid(),
        )

    D = nn.Sequential(
        nn.Flatten(),
        _create_layer( abs(np.prod(out_shape)), neurons, leak),
        _create_layer(neurons, neurons, leak),
        nn.Linear(neurons, 1),
    )
    return G, D




