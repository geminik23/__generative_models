import torch
from torch.utils.data import DataLoader, random_split
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from common import train_gen_network, get_digit_dataset, get_fashion_dataset
from arm_model import AutoregressiveModel, create_network
from common import plot_digit_imgs


import os
result_dir = 'results'
if not(os.path.exists(result_dir)):
    os.mkdir(result_dir)



##
# HYPERPARAMTERS
IMG_SIZE = 12
DIM = IMG_SIZE*IMG_SIZE
NUM_CLASS = 16 # value range 0-15
HIDDEN_DIM = 256
LEARNING_RATE = 1e-3
EPOCH = 500
KERNEL_SIZE = IMG_SIZE-1 # almost width or height length
BATCH_SIZE = 64
MAX_PATIENCE = 20

plot_func = lambda x: plot_digit_imgs(x, IMG_SIZE, (4, 4))


##
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

##
# DATASET
dataset = get_digit_dataset(IMG_SIZE, NUM_CLASS)


val_len = int(len(dataset)*0.05)
train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


## 
# Model
model = AutoregressiveModel(create_network(HIDDEN_DIM, NUM_CLASS, KERNEL_SIZE, 3), DIM, NUM_CLASS).to(device)

optimizer = torch.optim.Adamax(model.parameters(), LEARNING_RATE)


## TRAIN DATA
results = train_gen_network(os.path.join(result_dir,"arm_e_{}.pt"), model, MAX_PATIENCE, train_loader, val_loader, EPOCH, device, optimizer=optimizer, plot_func=plot_func, plot_img_interval=10)
results = pd.DataFrame(results)





## plot the losses
sns.lineplot(x="epoch", y="val loss", data=results)
plt.show()



## LOAD DATA
model = torch.load(os.path.join(result_dir, 'arm_e_38.pt'))


## GENERATE THE IMGS
imgs = model.inference(16, device).detach().cpu()
plot_digit_imgs(imgs, IMG_SIZE, (4,4))


