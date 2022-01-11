"""
Training function
Created by Marco Mameli
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets as Dataset
from torch import nn
from torch import optim
from tqdm import tqdm, trange

def train(config, dataset:Dataset, generator_model:nn.Module, discriminator_model:nn.Module):
    # device definition
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # initialization loss
    criterion = nn.BCELoss()
    # creation of the noise input
    fixed_noise = torch.randn(config['transform']['image_size'], config['generator']['latent_vector'], 1, 1, device=device)
    # Label convention
    real_label = 1.0
    fake_label = 0.0
    # Setup optimizer for the models
    discriminator_optimizer = optim.Adam(discriminator_model.parameters(), lr=config['training']['learning_rate'], betas=(config['training']['beta1'], config['training']['beta2']))
    generator_optimizer = optim.Adam(generator_model.parameters(), lr=config['training']['learning_rate'], betas=(config['training']['beta1'], config['training']['beta2']))
    # create train loader
    train_loader = DataLoader(dataset=dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['n_workers'])
    image_list = []
    generator_losses = []
    discriminator_losses = []
    iters = 0
    for epoch in trange(config['training']['epochs'], desc="Epoch"):
        for i, image in enumerate(tqdm(iterable=train_loader, desc='Training')):
            ## Train with all-real batch
            discriminator_model.zero_grad()
            # Batch
            real_image = image[0].to(device)
            b_size = real_image.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # forward the discriminator
            output = discriminator_model(real_image).view(-1)
            # Compute the loss on all-real batch
            discriminator_error_real = criterion(output, label)
            # Compute gradients
            discriminator_error_real.backward()
            D_x = output.mean().item()
            ## Train with all-fake batch
            noise = torch.randn(b_size, config['generator']['latent_vector'], 1, 1, device=device)
            # forward the generator -> Generate the fake image batch with G
            fake = generator_model(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator_model(fake.detach()).view(-1)
            # Computer the discriminator loss for all-fake batch
            discriminator_error_fake = criterion(output, label)
            # Computer the gradients for the batch
            discriminator_error_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and real batch
            discrimator_error_total = discriminator_error_fake + discriminator_error_real
            # update the discriminator
            discriminator_optimizer.step()

            ## Update the Generator network

            generator_model.zero_grad()
            label.fill_(real_label) # fake label are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator_model(fake).view(-1)
            # Calculate the Generator loss
            generator_error = criterion(output, label)
            # Compute gradients
            generator_error.backward()
            D_G_z2 = output.mean().item()
            # Update generator
            generator_optimizer.step()
            