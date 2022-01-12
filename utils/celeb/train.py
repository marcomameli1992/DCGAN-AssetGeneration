"""
Training function
Created by Marco Mameli
"""
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets as Dataset
import torchvision.utils as vutils
from torch import nn
from torch import optim
from tqdm import tqdm, trange
import numpy as np
from neptune.new.types import File as nFile

def train(config, dataset:Dataset, generator_model:nn.Module, discriminator_model:nn.Module, tracking=None):
    # activate tracking for model files
    if tracking is not None and config['saving']['neptune']:
        tracking['train/models'].track_files(config['saving']['base_path'])
    # device definition
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # move model to device
    discriminator_model.to(device)
    generator_model.to(device)
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
            discriminator_error_total = discriminator_error_fake + discriminator_error_real
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

            # Memorize data for plotting if tracking is not active
            generator_losses.append(generator_error.item())
            discriminator_losses.append(discriminator_error_total.item())
            # Give information about the state of the training
            if i % config['training']['information_step'] == 0:
                print(f"Epoch:{epoch}/{config['training']['epochs']}\nIteration:{i}/{len(train_loader)}\nDiscriminator Loss:\t{discriminator_error_total.item()}\nGenerator Loss:\t\t{generator_error.item()}")
                print(f"Distribution: \tD(x): {D_x}\n\t\t\t\tD(G(z)): {D_G_z1}/{D_G_z2}")

            if tracking is not None:
                tracking["train/loss/generator"].log(generator_error.item())
                tracking["train/loss/discriminator/total"].log(discriminator_error_total.item())
                tracking["train/loss/discriminator/real"].log(discriminator_error_real.item())
                tracking["train/loss/discriminator/fake"].log(discriminator_error_fake.item())
                tracking["train/distribution/discriminator/D(x)"].log(D_x)
                tracking["train/distribution/discriminator/D(G(z))"].log(D_G_z1/D_G_z2)
            # Saving generated image
            if (iters % config['training']['image_step'] == 0) or ((epoch == config['training']['epochs'] - 1) and (i == len(train_loader) - 1)):
                with torch.no_grad():
                    fake = generator_model(fixed_noise).detach().cpu()
                image = np.transpose(vutils.make_grid(fake, padding=2, normalize=True).numpy(), (1,2,0))
                image_list.append(image)
                if tracking is not None:
                    tracking["train/generated_grid_image"].log(nFile.as_image(image))

                print("Saving Model...")
                generator_folder = os.path.join(config['saving']['base_path'], 'generator')
                discriminator_folder = os.path.join(config['saving']['base_path'], 'discriminator')
                os.makedirs(generator_folder, exist_ok=True)
                os.makedirs(discriminator_folder, exist_ok=True)
                generator_path = os.path.join(generator_folder, f"generator-{epoch}-{iters}.pt")
                discriminator_path = os.path.join(discriminator_folder, f"discriminator-{epoch}-{iters}.pt")
                torch.save(generator_model.state_dict(), generator_path)
                torch.save(discriminator_model.state_dict(), discriminator_path)

            iters += 1