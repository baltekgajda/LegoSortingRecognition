from copy import copy

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid


def load_data(data_path, input_size, train_size=0.7, val_size=0.2,
              batch_size=4, num_workers=4, dataset_resize_factor=1) -> (DataLoader, DataLoader, DataLoader):
    # train_transforms = transforms.Compose([
    #     transforms.RandomResizedCrop(input_size, scale=(0.8, 1)),
    #     transforms.RandomVerticalFlip(),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])
    test_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_transforms = test_transforms
    dataset = ImageFolder(root=data_path, transform=test_transforms)

    if dataset_resize_factor != 1:
        dataset = reduce_dataset_size(dataset, dataset_resize_factor)

    dataset_dict = split_dataset_into_subsets_dict(dataset, train_size, val_size)
    # set different transformation in training set for augmentation
    dataset_dict['train'].dataset = copy(dataset)
    dataset_dict['train'].dataset.transform = train_transforms

    data_loaders_dict = {
        x: DataLoader(
            dataset=dataset_dict[x],
            batch_size=batch_size,
            num_workers=num_workers,
        )
        for x in ['train', 'val', 'test']
    }
    return data_loaders_dict


def reduce_dataset_size(dataset: Dataset, dataset_resize_factor: float) -> Dataset:
    if dataset_resize_factor > 1:
        raise ("dataset_resize_factor must be lower than 1, got: ", dataset_resize_factor)
    indices = torch.randperm(len(dataset)).tolist()
    new_size = int(len(dataset) * dataset_resize_factor)
    return Subset(dataset, indices[0:new_size])


def split_dataset_into_subsets_dict(dataset: Dataset, train_size: float, valid_size: float) -> dict:
    if train_size + valid_size >= 1:
        raise ("Sum of train and valid size factors must be lower than 1, got ", train_size + valid_size)
    lengths = [int(len(dataset) * x) for x in [train_size, valid_size]]
    lengths.append(len(dataset) - sum(lengths))  # add rest rows to testing set
    data_sets = random_split(dataset, lengths)
    return dict(zip(['train', 'val', 'test'], data_sets))


def show_random_images(data_loader: DataLoader):
    # get some random images
    images, labels = iter(data_loader).next()

    # show images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = make_grid(images).numpy().transpose((1, 2, 0))
    np_img = np_img * std + mean
    plt.imshow(np_img)
    if isinstance(data_loader.dataset, Subset):
        label = ', '.join(data_loader.dataset.dataset.classes[labels[j]] for j in range(4))
    else:
        label = ', '.join(data_loader.dataset.classes[labels[j]] for j in range(4))
    plt.xlabel(label)
    plt.show()
