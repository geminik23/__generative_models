from common import get_digit_dataset
from common import plot_digit_imgs

import torch
import numpy as np




def plot_image_from(dataset, img_size):
    imgs = np.stack([dataset[i] for i in range(16)])
    plot_digit_imgs(torch.tensor(imgs),img_size, (4,4))


if __name__ == '__main__':
    img_size=16
    dataset = get_digit_dataset(img_size, 16)
    plot_image_from(dataset, img_size)

