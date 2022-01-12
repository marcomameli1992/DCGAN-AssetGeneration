"""
The discriminator network
Created by Marco Mameli

"""
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, input_channel:int, feature_map:int, output_channel:int, multipliers:list):
        super(Discriminator, self).__init__()
        # definition of the architecture
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=feature_map * multipliers[0], kernel_size=4, stride=2, padding=1, bias=False)
        self.activation1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(in_channels=feature_map * multipliers[0], out_channels=feature_map * multipliers[1], kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(num_features=feature_map * multipliers[1])
        self.activation2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.Conv2d(in_channels=feature_map * multipliers[1], out_channels=feature_map * multipliers[2],
                               kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(num_features=feature_map * multipliers[2])
        self.activation3 = nn.LeakyReLU(negative_slope=0.2)
        self.conv4 = nn.Conv2d(in_channels=feature_map * multipliers[2], out_channels=feature_map * multipliers[3],
                               kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm4 = nn.BatchNorm2d(num_features=feature_map * multipliers[3])
        self.activation4 = nn.LeakyReLU(negative_slope=0.2)
        self.conv5 = nn.Conv2d(in_channels=feature_map * multipliers[3], out_channels=output_channel,
                               kernel_size=4, stride=1, padding=0, bias=False)
        self.activation5 = nn.Sigmoid()

    def forward(self, input):
        x = self.conv1(input)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation2(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.activation3(x)
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.activation4(x)
        x = self.conv5(x)
        x = self.activation5(x)
        return x

if __name__ == "__main__":
    print("test")