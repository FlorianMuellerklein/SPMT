from xml.etree.ElementPath import xpath_tokenizer
import torch
import torch.nn as nn

from torchvision import models

def make_layer(
    in_channels: int,
    out_channels: int,
    blocks: int = 2,
    downsample: bool = False,
    residual: bool = True,
) -> torch.nn.Sequential:
    """
    Builder function for making Residual blocks

    Parameters
    ----------
    input_channels: int
        number of channels coming into the block
    out_channels: int
        number of channels for the block to produce on output
    blocks: int
        number of blocks to chain together between downsampling steps
    downsample: bool
        whether to downsample the spatial dimensions within the block
    residual: bool
        whether to use residual skip connections within block
    """

    if downsample:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                        stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    else:
        downsample = None

    layers = []
    layers.append(BasicBlock(in_channels, out_channels, downsample, residual=residual))
    for _ in range(1, blocks):
        layers.append(BasicBlock(out_channels, out_channels, residual=residual))

    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    """
    Tweaked from official pytorch implementation.

    Parameters
    ----------
    input_channels: int
        number of channels coming into the block
    out_channels: int
        number of channels for the block to produce on output
    downsample: bool
        whether to downsample the spatial dimensions within the block
    residual: bool
        whether to use residual skip connections within block
    """

    def __init__(self, in_channels, out_channels, downsample=None, residual=True):
        super(BasicBlock, self).__init__()
        self.residual = residual

        self.conv_a= nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2 if downsample is not None else 1,
            bias=False,
            padding=1
        )
        self.bn_a = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv_b = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1
        )
        self.bn_b = nn.BatchNorm2d(out_channels)
        self.downsample = downsample


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            residual = x

        out = self.conv_a(x)
        out = self.bn_a(out)
        out = self.relu(out)

        out = self.conv_b(out)
        out = self.bn_b(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Tweaked from official pytorch implementation.

    Parameters
    ----------
    input_channels: int
        number of image channels
    """

    def __init__(self, n):
        super(ResNet, self).__init__()
        self.n = n

        self.conv_a = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn_a = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = make_layer(16, 16, self.n)
        self.layer2 = make_layer(16, 32, self.n, downsample=True)
        self.layer3 = make_layer(32, 64, self.n, downsample=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(0.5)
        self.classification_out = torch.nn.Linear(64, 10)
        self.consistency_out = torch.nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv_a(x)
        x = self.bn_a(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x).squeeze()
        x = self.dropout(x)

        # decoupled classification and consistency
        classification_out = self.classification_out(x)
        consistency_out = self.consistency_out(x)

        return classification_out, consistency_out

    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
