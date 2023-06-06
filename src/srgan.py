import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels, out_channels,
            discriminator=False,
            use_activation=True,
            use_batch_norm=True,
            **kwargs
    ):
        super(ConvBlock).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_layer = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.out_channels,
            **kwargs, bias=not use_batch_norm
        )


class UpscaleBlock(nn.Module):
    def __init__(self):
        super(UpscaleBlock).__init__()


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock).__init__()


class Generator(nn.Module):
    def __init__(self):
        super(Generator).__init__()


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator).__init__()
