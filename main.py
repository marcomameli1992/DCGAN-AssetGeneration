import os
import json
import argparse
from dataset import celeb
from model.base.discriminator import discriminator
from model.base.generator import generator
from utils.celeb.train import train
from utils.initilization import weights_init

import neptune.new as neptune

parser = argparse.ArgumentParser()
parser.add_argument('--configuration_file', '-cf', required=True, help="Represent the path for configuration file.")
parser.add_argument('--neptune', '-n', required=False, action='store_true', help="The activation of the neptune storing experiment")
parser.add_argument('--neptune_config', '-nc', type=str, required=False, help="The path for the configuration file for the neptune using")
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

# Init neptune
run = None
if args.neptune:
    if args.neptune_config is not None:
        with open(args.neptune_config, 'r') as config_file:
            config_neptune = json.load(config_file)
        run = neptune.init(
            project=config_neptune['neptune']['project'],
            api_token=config_neptune['neptune']['token'],
        )
        run.sync()
        run['parameters/train'] = config_parameter['training']
        run['parameter/generator'] = config_parameter['generator']
        run['parameter/discriminator'] = config_parameter['discriminator']
        run['parameter/transform'] = config_parameter['transform']
    else:
        raise argparse.ArgumentError("Please provide the the path for the neptune configuration if it is active")



# Training
train(config=config_parameter, dataset=dataset, generator_model=generator_model, discriminator_model=discriminator_model, tracking=run)