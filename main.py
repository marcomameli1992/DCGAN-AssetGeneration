import os
import json
import argparse
from dataset import celeb
from model.base.discriminator import discriminator
from model.base.generator import generator
from utils.celeb.train import train
from utils.initilization import weights_init
parser = argparse.ArgumentParser()
parser.add_argument('--configuration_file', '-cf', required=True, help="Represent the path for configuration file.")
args = parser.parse_args()

with open(args.configuration_file, 'r') as config_file:
    config_parameter = json.load(config_file)

# Dataset opening
dataset_path = config_parameter['dataset']['path']
dataset = celeb.celeba(config=config_parameter, dataroot=dataset_path)

# Model creation
discriminator_parameter = config_parameter['discriminator']
discriminator_model = discriminator.Discriminator(input_channel=discriminator_parameter['input_channel'], feature_map=discriminator_parameter['feature_map'], output_channel=discriminator_parameter['output_channel'], multipliers=discriminator_parameter['multipliers'])
generator_parameter = config_parameter['generator']
generator_model = generator.Generator(input_channel=generator_parameter['latent_vector'], feature_map=generator_parameter['feature_map'], output_channel=generator_parameter['output_channel'], multipliers=generator_parameter['multipliers'])
# Init the model weight
generator_model.apply(weights_init)
discriminator_model.apply(weights_init)
# Training
train(config=config_parameter, dataset=dataset, generator_model=generator_model, discriminator_model=discriminator_model)