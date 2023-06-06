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

        self.discriminator = discriminator
        self.use_activation = use_activation
        self.use_batch_norm = use_batch_norm

        self.conv_layer = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.out_channels,
            **kwargs, bias=not self.use_batch_norm
        )
        self.batch_norm = nn.BatchNorm2d(self.out_channels) if self.use_batch_norm else nn.Identity()
        self.activation_func = nn.LeakyReLU(0.2, inplace=True) if self.discriminator else nn.PReLU(num_parameters=out_channels)

    def forward(self, x):
        out = self.conv_layer(x)
        out = self.batch_norm(out)
        if self.use_activation:
            out = self.activation_func(out)

        return out


class UpsampleBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            scale_factor
    ):
        super(UpsampleBlock).__init__()

        self.in_channels = in_channels
        self.scale_factor = scale_factor

        self.conv_layer = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.in_channels * scale_factor ** 2,
            kernel_size=3, stride=1, padding=1
        )
        self.pixel_shuffle_layer = nn.PixelShuffle(scale_factor)  # in_channels * 4, H, W --> in_channels, H * 2, W * 2
        self.activation_func = nn.PReLU(num_parameters=self.in_channels)

    def forward(self, x):
        out = self.conv_layer(x)
        out = self.pixel_shuffle_layer(out)
        out = self.activation_func(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock).__init__()

        self.block1 = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_activation=True,
            use_batch_norm=True
        )

        self.block2 = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_activation=False,
            use_batch_norm=True
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)

        return out + x


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, resnet_num_channels=64, resnet_num_blocks=16):
        super(Generator).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resnet_num_channels = resnet_num_channels
        self.resnet_num_blocks = resnet_num_blocks

        self.initial = ConvBlock(
            in_channels=self.in_channels,
            out_channels=self.resnet_num_channels,
            kernel_size=9, stride=1, padding=4,
            use_batch_norm=False
        )

        self.resnet = nn.Sequential(*[ResidualBlock(self.resnet_num_channels) for _ in range(self.resnet_num_blocks)])

        self.conv_block = ConvBlock(
            in_channels=self.resnet_num_channels,
            out_channels=self.resnet_num_channels,
            kernel_size=3, stride=1, padding=1,
            use_activation=False
        )

        self.upsamples = nn.Sequential(
            UpsampleBlock(self.resnet_num_channels, scale_factor=2),
            UpsampleBlock(self.resnet_num_channels, scale_factor=2)
        )

        self.final = nn.Conv2d(
            in_channels=self.resnet_num_channels,
            out_channels=self.out_channels,
            kernel_size=9, stride=1, padding=1
        )

        self.final_activation_func = nn.Tanh()

    def forward(self, x):
        initial = self.initial(x)
        out = self.resnet(initial)
        out = self.conv_block(out) + initial
        out = self.upsamples(out)
        out = self.final(out)
        out = self.final_activation_func(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=(64, 64, 128, 128, 256, 256, 512, 512)):
        super(Discriminator).__init__()

        self.in_channels = in_channels
        self.features = list(features)

        blocks = []

        for idx, feature in enumerate(self.features):
            blocks.append(
                ConvBlock(
                    in_channels=in_channels, out_channels=feature,
                    kernel_size=3, stride=1 + idx % 2, padding=1,
                    use_activation=True,
                    use_batch_norm=True if idx != 0 else False
                )
            )
            in_channels=feature

        self.blocks = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(in_features=512*6*6, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=1)
        )

    def forward(self, x):
        out = self.blocks(x)
        out = self.classifier(out)
        return out
