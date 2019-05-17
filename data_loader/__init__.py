import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid


def load_data(train_size=0.7, valid_size=0.2, test_size=0.1) \
        -> (DataLoader, DataLoader, DataLoader):
    dataset = ImageFolder(root='data\\Base Images\\',
                          transform=transforms.Compose([
                              transforms.RandomResizedCrop(224, scale=(0.85, 1)),
                              transforms.RandomVerticalFlip(),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225]),
                          ]))

    train_indices, valid_indices, test_indices = (
        split_indices(len(dataset), train_size, valid_size, test_size)
    )

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=4,
                                               sampler=SubsetRandomSampler(train_indices))
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=4,
                                               sampler=SubsetRandomSampler(valid_indices))
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=4,
                                              sampler=SubsetRandomSampler(test_indices))
    return train_loader, valid_loader, test_loader


def split_indices(dataset_len, train_size, valid_size, test_size) \
        -> (torch.Tensor, torch.Tensor, torch.Tensor):
    if train_size + valid_size + test_size > 1:
        raise ValueError('Sum of train_size, valid_size, test_size must be lower than 1')

    indices = torch.randperm(dataset_len)

    train_indices = indices[:int(len(indices) * train_size)]
    idx = int(len(indices) * train_size)
    valid_indices = indices[idx: int(len(indices) * valid_size) + idx]
    idx += int(len(indices) * valid_size)
    test_indices = indices[idx: int(len(indices) * test_size) + idx]

    return train_indices, valid_indices, test_indices


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
