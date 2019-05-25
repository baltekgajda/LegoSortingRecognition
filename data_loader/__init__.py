import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid


def load_data(data_path, input_size, train_size=0.7, valid_size=0.2, test_size=0.1,
              batch_size=4, num_workers=4) \
        -> (DataLoader, DataLoader, DataLoader):
    dataset = ImageFolder(root=data_path,
                          transform=transforms.Compose([
                              transforms.RandomResizedCrop(input_size, scale=(0.85, 1)),
                              transforms.RandomVerticalFlip(),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225]),
                          ]))

    indices_dict = split_indices(len(dataset), train_size, valid_size, test_size)

    data_loaders_dict = {x: torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                                        sampler=SubsetRandomSampler(indices_dict[x]))
                         for x in ['train', 'val', 'test']}
    return data_loaders_dict


def split_indices(dataset_len, train_size, val_size, test_size) -> dict:
    if train_size + val_size + test_size > 1:
        raise ValueError('Sum of train_size, valid_size, test_size must be lower than 1')
    indices = torch.randperm(dataset_len)

    start = 0
    indices_dict = {}
    size_dict = {'train': train_size, 'val': val_size, 'test': test_size}
    for x in ['train', 'val', 'test']:
        end = int(len(indices) * size_dict[x]) + start
        indices_dict[x] = indices[start: end]
        start = end

    return indices_dict


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
