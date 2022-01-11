"""
Utilities function
Created by Marco Mameli

"""

from torch import nn

def weights_init(layer):
    """
    Function for weight initialization of the layers of network
    :param layer: thenetwork layer
    :return: None it is inplace function
    """
    class_name = layer.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias.data, val=0.0)