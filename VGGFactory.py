from __future__ import print_function
from __future__ import division
import torch.nn as nn
from torchvision import models


def create_model(model_type, num_of_classes):
    """Creates VGG network

    :param model_type:  1 - classification layers set to train,
                        2 - like 1st plus last conv layer set to train
                        3 - fully trained model with with all params set to train
    :param num_of_classes: Number of output classes
    :return: VGG network
    """
    model = models.vgg11(pretrained=True)

    # Set requires_grad to false for every layer. New layers requires grad by default
    set_requires_grad(model, False)

    if model_type == 1:
        adjust_classifier(model.classifier, num_of_classes)
    elif model_type == 2:
        adjust_classifier(model.classifier, num_of_classes)
        for name, param in model.features.named_parameters():
            if name == '18.weight' or name == '18.bias':
                param.requires_grad = True
    elif model_type == 3:
        adjust_classifier(model.classifier, num_of_classes)
        set_requires_grad(model, True)
    else:
        raise("Unsupported model_type. Expected 1 or 2 or 3, got: ", model_type)

    return model


def simplify_model(model):
    """ Simplifies VGG 11 model by removing last conv layer"""
    # (256, 8) means conv layer with 256 output channels , 'M' means max pool layer
    # Second number in each tuple means index of layer. In VGG each conv layer is followed by ReLu making indexing
    # a little bit harder
    model_cfg = [(64, 0), 'M', (128, 3), 'M', (256, 6), (256, 8), 'M', (512, 11), (512, 13), 'M']
    layers = []
    in_channels = 3
    for _, layer in enumerate(model_cfg):
        if layer == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            num_of_output_channels, index = layer
            conv2d = nn.Conv2d(in_channels, num_of_output_channels, kernel_size=3, padding=1)
            conv2d.weight.data = model.features[index].weight.data.clone()
            conv2d.bias.data = model.features[index].bias.data.clone()
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = num_of_output_channels

    # Adding one more max pool layer so that output of features layer would be 7 x 7
    # TODO: Verify how this impacts model.
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

    model.features = nn.Sequential(*layers)
    return model


def adjust_classifier(classifier, num_of_classes):
    set_requires_grad(classifier, True)
    classifier[6] = nn.Linear(4096, num_of_classes)


def set_requires_grad(model, val):
    for param in model.parameters():
        param.requires_grad = val


