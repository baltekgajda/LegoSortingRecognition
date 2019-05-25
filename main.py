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

from data_loader import load_data
from feature_extraction import initialize_model, train_model
import VGGFactory

model_name = "VGG"
data_dir = "./data/Base Images"
num_classes = 20
num_of_epochs = 50
input_size = 224

model = VGGFactory.create_model(1, num_classes)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

print()

simple_model = VGGFactory.simplify_model(model)
for name, param in simple_model.named_parameters():
    if param.requires_grad:
        print(name)

print("Initializing Datasets and Dataloaders...")
dataloaders_dict = load_data('./data/Base Images', input_size, 0.3, 0.1, 0.1)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

params_to_update = model.parameters()
print("Params to learn:")
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad:
        params_to_update.append(param)
        print("\t", name)

optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
torch.cuda.current_device()
model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer_ft, device, num_epochs=num_of_epochs)
