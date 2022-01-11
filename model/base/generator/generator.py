"""
The generator network
Created by Marco Mameli

"""
from torch import nn

class Generator(nn.Module):
    def __init__(self, input_channel:int, feature_map:int, output_channel:int, multipliers:list):
        super.__init__(Generator, self).__init__()
        # Definition of the architecture
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels=input_channel, out_channels=feature_map * multipliers[0], kernel_size=4, stride=1, padding=0, bias=True)
        self.batch_norm1 = nn.BatchNorm2d(feature_map * multipliers[0])
        self.activation1 = nn.ReLU()
        self.conv_transpose2 = nn.ConvTranspose2d(in_channels=feature_map * multipliers[0], out_channels=feature_map * multipliers[1], kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(feature_map * multipliers[1])
        self.activation2 = nn.ReLU()
        self.conv_transpose3 = nn.ConvTranspose2d(in_channels=feature_map * multipliers[1],
                                                  out_channels=feature_map * multipliers[2], kernel_size=4, stride=2,
                                                  padding=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(feature_map * multipliers[2])
        self.activation3 = nn.ReLU()
        self.conv_transpose4 = nn.ConvTranspose2d(in_channels=feature_map * multipliers[2],
                                                  out_channels=feature_map * multipliers[3], kernel_size=4, stride=2,
                                                  padding=1, bias=False)
        self.batch_norm4 = nn.BatchNorm2d(feature_map * multipliers[3])
        self.activation4 = nn.ReLU()
        self.conv_transpose5 = nn.ConvTranspose2d(in_channels=feature_map * multipliers[3],
                                                  out_channels=output_channel, kernel_size=4, stride=2,
                                                  padding=1, bias=False)
        self.activation5 = nn.Tanh()

    def forward(self, input):
        x = self.conv_transpose1(input)
        x = self.batch_norm1(x)
        x = self.activation1(x)
        x = self.conv_transpose2(x)
        x = self.batch_norm2(x)
        x = self.activation2(x)
        x = self.conv_transpose3(x)
        x = self.batch_norm3(x)
        x = self.activation3(x)
        x = self.conv_transpose4(x)
        x = self.batch_norm4(x)
        x = self.activation4(x)
        x = self.conv_transpose5(x)
        x = self.activation5(x)
        return x

if __name__ == "__main__":
    print("test")