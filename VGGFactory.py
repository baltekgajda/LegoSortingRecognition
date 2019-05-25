from __future__ import print_function
from __future__ import division
import torch.nn as nn
from torchvision import models

def create_model(model_type, num_of_clases):
    """Creates VGG network

    :param model_type:  1 - classification layers untrained,
                        2 - like 1st plus last conv layer untrained
                        3 - fully trained model with requires_grad set to True
    :param num_of_clases: Number of output classes
    :return: VGG network
    """
    model = models.vgg11(pretrained=True)

    # Set requires_grad to false for every layer. New layers requires grad by default
    set_requires_grad(model, False)

    if model_type == 1:
        model.classifier = create_classifier(num_of_clases)
    elif model_type == 2:
        model.classifier = create_classifier(num_of_clases)
        reset_last_conv_layer(model)
    elif model_type == 3:
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


def create_classifier(num_of_classes):
    return nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_of_classes),
    )


def reset_last_conv_layer(model):
    """ Resets last conv layer and last max pool layer"""
    # Relying on hardcoded index values is wrong
    # TODO: How to access and create last layer based only on what's inside ?
    model.features[11] = nn.Conv2d(512, 512, kernel_size=3, padding=1)

def set_requires_grad(model, val):
    for param in model.parameters():
        param.requires_grad = val


