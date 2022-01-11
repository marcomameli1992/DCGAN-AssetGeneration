"""
Training function
Created by Marco Mameli
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets as Dataset
from torch import nn
from torch import optim

def train(config, dataset:Dataset, generator_model:nn.Module, discriminator_model:nn.Module):
    # device definition
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # initialization loss
    criterion = nn.BCELoss()
    # creation of the noise input
    fixed_noise = torch.randn(config['transform']['image_size'])