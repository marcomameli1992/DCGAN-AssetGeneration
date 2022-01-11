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
    fixed_noise = torch.randn(config['transform']['image_size'], config['generator']['latent_vector'], )
    # Label convention
    real_label = 1.0
    fake_label = 0.0
    # Setup optimizer for the models
    discrminator_optimizer = optim.Adam(discriminator_model.parameters(), lr=config['training']['learning_rate'], betas=(config['training']['beta1'], config['training']['beta2']))
    generator_optimizer = optim.Adam(generator_model.parameters(), lr=config['training']['learning_rate'], betas=(config['training']['beta1'], config['training']['beta2']))
