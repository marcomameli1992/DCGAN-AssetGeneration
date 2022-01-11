"""
File for the celeb dataset
Created by Marco Mameli
"""

from torchvision import datasets as dataset
from torchvision import transforms as transforms

def celeba(config, dataroot):
    """
    Thi function opening the celeba dataset using the folder path.
    :param dataroot:
    :return: the dataset istance
    """
    dt = dataset.ImageFolder(root=dataroot, transform=transforms.Compose([
        transforms.Resize(config['transform']['image_size']),
                          transforms.CenterCrop(config['transform']['image_size']),
                          transforms.ToTensor(),
                          transforms.Normalize(
                              config['transform']['mean'],
                              config['transform']['std']
                          )
    ]))

    return dt
