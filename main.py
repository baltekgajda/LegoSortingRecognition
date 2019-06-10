import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import load_data
from feature_extraction import train_model
from net_test_and_metrics import test_network
import VGGFactory
from svm_classification import train_model_with_svm


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
dataloaders_dict = load_data('./data/Base Images', input_size, batch_size=2)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

params_to_update = list(filter(lambda param: param.requires_grad, model.parameters()))
print("Params to learn:")
print(get_params_to_learn(model))

optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer_ft, device, num_epochs=num_of_epochs)

metrics = test_network(model_ft, dataloaders_dict['test'], num_classes, device=device)
model_classifier = train_model_with_svm(model, dataloaders_dict)

metrics = test_network(model_ft, dataloaders_dict['test'], num_classes, device=torch.device("cpu"),
                       svm_classifier=model_classifier)
print('Accuracy  {:4f}'.format(metrics['accuracy']))
print('Top 1 error {:4f}'.format(metrics['top_1_error']))

# Save trained model
# utils.save_model(model_ft, "./models")
