import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from gan_models import get_gan_model
from common import train_gan, sample_gan, train_wgan_gp, get_digit_dataset, get_fashion_dataset
from common import plot_digit_imgs

import os
result_dir = 'results'
if not(os.path.exists(result_dir)): os.mkdir(result_dir)



##
# HYPERPARAMETER
IMG_SIZE = 12
NUM_CLASS = 16 # value range 0-15
LEARNING_RATE = 1e-3
EPOCH = 500
LATENT_SIZE = 32
HIDDEN_DIM = 128
BATCH_SIZE = 128
OUT_SHAPE = (-1, 1, IMG_SIZE, IMG_SIZE)
 

##
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



##
# DATASET
def collate_fn(batch):
    return torch.stack(batch).view(-1, 1, IMG_SIZE, IMG_SIZE).float() / float(NUM_CLASS-1)

train_dataset = get_digit_dataset(IMG_SIZE, NUM_CLASS)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


## 
# MODEL
G, D = get_gan_model(LATENT_SIZE, HIDDEN_DIM, OUT_SHAPE)
optimizer_D = torch.optim.AdamW(D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))
optimizer_G = torch.optim.AdamW(G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))


##
# TRAIN
# d_losses, g_losses = train_gan(os.path.join(result_dir,'{}_gan.pt'), G, D, LATENT_SIZE, nn.BCEWithLogitsLoss(), optimizer_D, optimizer_G, train_loader, EPOCH, device )
d_losses, g_losses = train_wgan_gp(os.path.join(result_dir,'{}_wgan_gp.pt'), G, D, LATENT_SIZE, optimizer_D, optimizer_G, train_loader, EPOCH, device )


##
# Plot losses
plt.figure(figsize=(10,4))
plt.title('Generator and Discriminator losses ')
plt.plot(np.convolve(g_losses, np.ones((100,))/100, mode='valid'), label='G')
plt.plot(np.convolve(d_losses, np.ones((100,))/100, mode='valid'), label='D')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()



##
# Generate samples
# G = torch.load(os.path.join(result_dir, 'g_gan.pt'))
# D = torch.load(os.path.join(result_dir, 'd_gan.pt'))

G = torch.load(os.path.join(result_dir, 'g_wgan_gp.pt'))
D = torch.load(os.path.join(result_dir, 'd_wgan_gp.pt'))

imgs, _ = sample_gan(G, D, 16, LATENT_SIZE, device)
plot_digit_imgs(imgs, IMG_SIZE, (4,4))
