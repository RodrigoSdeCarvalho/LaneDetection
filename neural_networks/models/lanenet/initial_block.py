from torch import nn
import torch
from neural_networks.models.model import Model


class InitialBlock(Model):
    def __init__(self,
                 in_channels,
                 out_channels,
                 name: str = "initial_block",
                 bias=False,
                 relu=True):
        super().__init__(name)

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - 1,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias)

        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)

        self.batch_norm = nn.BatchNorm2d(out_channels)

        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        # Concatenate branches
        out = torch.cat((main, ext), 1)

        # Apply batch normalization
        out = self.batch_norm(out)

        return self.out_activation(out)
