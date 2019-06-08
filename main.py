from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import utils

from data_loader import load_data
from feature_extraction import train_model
from net_test_and_metrics import test_network
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
dataloaders_dict = load_data('./data/Base Images', input_size, batch_size=4)

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

metrics = test_network(model, dataloaders_dict['test'], device)
print('Accuracy  {:4f}'.format(metrics['accuracy']))
print('Top 1 error {:4f}'.format(metrics['top_1_error']))

# Save trained model
utils.save_model(model_ft, "./models")
