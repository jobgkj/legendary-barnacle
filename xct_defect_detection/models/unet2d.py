"""
=============================================================================
2D U-Net Architecture
=============================================================================
Standard encoder-decoder U-Net (Ronneberger et al., 2015) with configurable
encoder depth, batch normalisation, dropout in the bottleneck, and sigmoid
output for binary defect/background segmentation.
=============================================================================
"""

import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ENCODER_CHANNELS, DROPOUT_RATE


class ConvBlock(nn.Module):
    """
    Double convolution block: Conv → BN → ReLU → Conv → BN → ReLU.

    This is the fundamental building block of the U-Net encoder
    and decoder stages.

    Parameters
    ----------
    in_channels  : int  — number of input feature channels
    out_channels : int  — number of output feature channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    Encoder stage: ConvBlock followed by 2×2 max pooling.

    Returns both the feature map before pooling (for skip connection)
    and the downsampled feature map.

    Parameters
    ----------
    in_channels  : int  — number of input feature channels
    out_channels : int  — number of output feature channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        skip      = self.conv(x)
        downsampled = self.pool(skip)
        return skip, downsampled


class DecoderBlock(nn.Module):
    """
    Decoder stage: transposed convolution upsampling followed by
    concatenation of the skip connection and a ConvBlock.

    Parameters
    ----------
    in_channels  : int  — number of input channels (from previous decoder)
    out_channels : int  — number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        # After concatenation with skip: in_channels = out_channels * 2
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(
        self,
        x:    torch.Tensor,
        skip: torch.Tensor
    ) -> torch.Tensor:
        x = self.upsample(x)

        # Handle odd spatial dimensions by centre-cropping skip connection
        if x.shape != skip.shape:
            skip = skip[
                :, :,
                :x.shape[2],
                :x.shape[3]
            ]

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """
    2D U-Net for binary defect segmentation in XCT slices.

    Architecture follows Ronneberger et al. (2015) with configurable
    encoder depth and dropout regularisation in the bottleneck.

    Input:  (B, 1, H, W)  — single-channel greyscale XCT patch
    Output: (B, 1, H, W)  — per-pixel defect probability in [0, 1]

    Parameters
    ----------
    in_channels      : int   — input channels (1 for greyscale XCT)
    out_channels     : int   — output channels (1 for binary segmentation)
    encoder_channels : list  — feature channels at each encoder depth
    dropout_rate     : float — dropout probability in bottleneck
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
            self.encoders.append(EncoderBlock(prev_ch, ch))
            prev_ch = ch

        # Bottleneck
        bottleneck_ch = channels[-1] * 2
        self.bottleneck = nn.Sequential(
            ConvBlock(channels[-1], bottleneck_ch),
            nn.Dropout2d(p=dropout_rate)
        )

        # Decoder (reverse encoder channels)
        self.decoders = nn.ModuleList()
        dec_channels  = list(reversed(channels))
        prev_ch       = bottleneck_ch
        for ch in dec_channels:
            self.decoders.append(DecoderBlock(prev_ch, ch))
            prev_ch = ch

        # Output layer
        self.output_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        self.sigmoid      = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder, bottleneck, and decoder.

        Parameters
        ----------
        x : torch.Tensor  — (B, 1, H, W)

        Returns
        -------
        torch.Tensor  — (B, 1, H, W), values in [0, 1]
        """
        skips = []

        # Encoder path
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.sigmoid(self.output_conv(x))


def get_model() -> UNet2D:
    """Instantiate and return a 2D U-Net with config defaults."""
    model = UNet2D(
        in_channels      = 1,
        out_channels     = 1,
        encoder_channels = ENCODER_CHANNELS,
        dropout_rate     = DROPOUT_RATE
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [Model] 2D U-Net — trainable parameters: {n_params:,}")
    return model
