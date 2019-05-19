from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from feature_extraction import initialize_model, train_model 
import VGGFactory

# model_name = "VGG"
# num_classes = 2
# data_dir = "./data/hymenoptera_data"
# num_of_epochs = 5
#
# model_ft, input_size = initialize_model(model_name, num_classes, use_pretrained=True)
#
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(input_size),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize(input_size),
#         transforms.CenterCrop(input_size),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }
#
# print("Initializing Datasets and Dataloaders...")
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
#
# dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
#
# # Detect if we have a GPU available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# params_to_update = model_ft.parameters()
# print("Params to learn:")
# params_to_update = []
# for name,param in model_ft.named_parameters():
#     if param.requires_grad:
#         params_to_update.append(param)
#         print("\t", name)
#
# optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
#
# # Setup the loss fxn
# criterion = nn.CrossEntropyLoss()
#
# print()
# for name,param in model_ft.named_parameters():
#     print("\t", name)

# Train and evaluate
# model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, num_epochs=num_of_epochs)

model = VGGFactory.create_model(1, 10)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

print()

simple_model = VGGFactory.simplify_model(model)
for name, param in simple_model.named_parameters():
    if param.requires_grad:
        print(name)
