from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid


def load_data() -> DataLoader:
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(root='data\\Base Images\\',
                          transform=data_transform)
    data_loader = DataLoader(dataset,
                             batch_size=4, shuffle=True,
                             num_workers=4)
    return data_loader


def show_random_images(data_loader: DataLoader):
    # get some random images
    images, labels = iter(data_loader).next()

    # show images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = make_grid(images).numpy().transpose((1, 2, 0))
    np_img = np_img * std + mean
    plt.imshow(np_img)
    label = ', '.join('%5s' % data_loader.dataset.classes[labels[j]] for j in range(4))
    plt.xlabel(label)
    plt.show()
