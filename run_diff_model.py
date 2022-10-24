import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import torchvision.transforms as T
from common import train_gen_network, get_digit_dataset, get_fashion_dataset
from diffusion_model import DiffusionLatentModel
from common import plot_digit_imgs


import os
result_dir = 'results'
if not(os.path.exists(result_dir)):
    os.mkdir(result_dir)


##
# HYPERPARAMTERS
IMG_SIZE = 12
DIM = IMG_SIZE*IMG_SIZE
NUM_STEPS = 7
HIDDEN_DIM = 512
LEARNING_RATE = 1e-3
EPOCH = 2000
BATCH_SIZE = 64
MAX_PATIENCE = 80
BETAS=(0.02, 0.9)

plot_func = lambda x: plot_digit_imgs(x, IMG_SIZE, (4, 4))

##
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

##
# DATASET
transform = T.Lambda(lambda x: torch.flatten(x*2.0-1.0)) # scaled [-1. ~ 1.]
dataset = get_digit_dataset(IMG_SIZE, None, transform)

val_len = int(len(dataset)*0.05)
train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


## 
# Model
def p_net():
    return nn.Sequential(
        nn.Linear(DIM, HIDDEN_DIM*2), nn.LeakyReLU(),
        nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM*2), nn.LeakyReLU(),
        nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM), nn.LeakyReLU(),
        nn.Linear(HIDDEN_DIM, DIM*2)
    )


def decode_net():
    return nn.Sequential(
        nn.Linear(DIM, HIDDEN_DIM*2),
        nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM*2),
        nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM),
        nn.Linear(HIDDEN_DIM, DIM), nn.Tanh(),
    )

model = DiffusionLatentModel(DIM, NUM_STEPS, p_net, decode_net, betas=BETAS)
optimizer = torch.optim.Adamax(model.parameters(), LEARNING_RATE)


## TRAIN DATA
results = train_gen_network(os.path.join(result_dir,"diff_e_{}.pt"), model, MAX_PATIENCE, train_loader, val_loader, EPOCH, device, optimizer=optimizer, plot_func=plot_func, plot_img_interval=25)
results = pd.DataFrame(results)



## plot the losses
sns.lineplot(x="epoch", y="val loss", data=results)
plt.show()



## LOAD DATA
model = torch.load(os.path.join(result_dir, 'diff_e_.pt'))


## GENERATE THE IMGS
model = model.eval()
imgs = model.inference(16, device).detach().cpu()
plot_digit_imgs(imgs, IMG_SIZE, (4,4))


