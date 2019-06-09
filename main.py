from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import load_data
from feature_extraction import train_model
import VGGFactory


def get_params_to_learn(m):
    params_to_learn = []
    for name, param in m.named_parameters():
        if param.requires_grad:
            params_to_learn.append(name)
    return params_to_learn


data_dir = "./images"
num_classes = 20
num_of_epochs = 50
input_size = 224

model = VGGFactory.create_model(3, num_classes)

print("Initializing Datasets and Dataloaders...")
dataloaders_dict = load_data(data_dir, input_size, batch_size=2)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

params_to_update = list(filter(lambda param: param.requires_grad, model.parameters()))
print("Params to learn:")
print(get_params_to_learn(model))

optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
# torch.cuda.current_device()
model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer_ft, device, num_epochs=num_of_epochs)

# Save trained model
# utils.save_model(model_ft, "./models")
