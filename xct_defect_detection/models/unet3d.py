"""
=============================================================================
3D U-Net Architecture
=============================================================================
Volumetric encoder-decoder U-Net (Ronneberger et al., 2015 adapted to 3D)
with configurable encoder depth, batch normalisation, dropout in the
bottleneck, and sigmoid output for binary defect/background segmentation.
=============================================================================
"""

import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ENCODER_CHANNELS, DROPOUT_RATE


class ConvBlock3D(nn.Module):
    """
    Double convolution block: Conv → BN → ReLU → Conv → BN → ReLU (3D).

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock3D(nn.Module):
    """
    Encoder stage: ConvBlock3D followed by 2×2×2 max pooling.

    Returns feature map before pooling (skip connection)
    and downsampled feature map.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        downsampled = self.pool(skip)
        return skip, downsampled


class DecoderBlock3D(nn.Module):
    """
    Decoder stage: transposed convolution upsampling followed by
    concatenation of skip connection and ConvBlock3D.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        # After concatenation with skip: in_channels = out_channels * 2
        self.conv = ConvBlock3D(out_channels * 2, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor
    ) -> torch.Tensor:
        x = self.upsample(x)

        # Handle odd spatial dimensions by centre-cropping skip
        if x.shape[2:] != skip.shape[2:]:
            skip = skip[
                :,
                :,
                :x.shape[2],
                :x.shape[3],
                :x.shape[4]
            ]

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric binary defect segmentation.

    Input:  (B, 1, D, H, W)
    Output: (B, 1, D, H, W)
    """

    def __init__(
        self,
        in_channels      : int   = 1,
        out_channels     : int   = 1,
        encoder_channels : list  = None,
        dropout_rate     : float = None
    ):
        super().__init__()

        channels     = encoder_channels or ENCODER_CHANNELS
        dropout_rate = dropout_rate     or DROPOUT_RATE

        # Encoder
        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(EncoderBlock3D(prev_ch, ch))
            prev_ch = ch

        # Bottleneck
        bottleneck_ch = channels[-1] * 2
        self.bottleneck = nn.Sequential(
            ConvBlock3D(channels[-1], bottleneck_ch),
            nn.Dropout3d(p=dropout_rate)
        )

        # Decoder
        self.decoders = nn.ModuleList()
        prev_ch = bottleneck_ch
        for ch in reversed(channels):
            self.decoders.append(DecoderBlock3D(prev_ch, ch))
            prev_ch = ch

        # Output
        self.output_conv = nn.Conv3d(channels[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        # Encoder
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.sigmoid(self.output_conv(x))


def get_model() -> UNet3D:
    """Instantiate and return a 3D U-Net with config defaults."""
    model = UNet3D(
        in_channels      = 1,
        out_channels     = 1,
        encoder_channels = ENCODER_CHANNELS,
        dropout_rate     = DROPOUT_RATE
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [Model] 3D U-Net — trainable parameters: {n_params:,}")
    return model
