from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy

import VGGFactory
import utils

# Setup
feature_extract = True

def train_model(model, dataLoaders, criterion, optimizer, device, num_epochs=4):
    since = time.time()
    
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model = model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
        
            running_loss = 0.0
            running_corrects = 0
        
            for inputs, labels in dataLoaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # Backward
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataLoaders[phase].sampler)
            epoch_acc = running_corrects.double() / len(dataLoaders[phase].sampler)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)


    print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def get_params_to_update(model):
    params_to_update = model.parameters()
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)

    return params_to_update


def train_classifier_only(dataloaders_dicts, models_folder, device, num_of_classes=20, num_of_epochs=10):
    return _train_model_for_scenario(1, "vgg-classifier-only", models_folder, dataloaders_dicts, device, num_of_classes, num_of_epochs)


def train_classifier_and_last_conv(dataloaders_dicts, models_folder, device, num_of_classes=20, num_of_epochs=10):
    return _train_model_for_scenario(2, "vgg-last_conv", models_folder, dataloaders_dicts, device, num_of_classes, num_of_epochs)


def train_full_net(dataloaders_dicts, models_folder, device, num_of_classes=20, num_of_epochs=10):
    return _train_model_for_scenario(3, "vgg-full-net", models_folder, dataloaders_dicts, device, num_of_classes, num_of_epochs)


def train_simplified_net(model, dataloaders_dicts, models_folder, device, num_of_epochs=10):
    simpler_model = VGGFactory.simplify_model(model)
    optimizer = optim.SGD(get_params_to_update(model), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model_ft, hist = train_model(simpler_model, dataloaders_dicts, criterion, optimizer, device, num_epochs=num_of_epochs)
    torch.save(model_ft, models_folder + "vgg-simplified" + ".pth")
    return model_ft, hist


def _train_model_for_scenario(scenario_id, model_name, models_folder,  dataloaders_dicts, device, num_of_classes=20, num_of_epochs=10):
    model = VGGFactory.create_model(scenario_id, num_of_classes)
    optimizer = optim.SGD(get_params_to_update(model), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model_ft, hist = train_model(model, dataloaders_dicts, criterion, optimizer, device, num_epochs=num_of_epochs)
    torch.save(model_ft, models_folder + model_name + ".pth")
    return model_ft, hist
