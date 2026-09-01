"""
=============================================================================
2.5D U-Net Architecture
=============================================================================
Standard encoder-decoder U-Net (Ronneberger et al., 2015) with configurable
encoder depth, batch normalisation, dropout in the bottleneck, and sigmoid
output for binary defect/background segmentation.

"2.5D": the network is still built entirely from 2D convolutions (cheap,
same as a plain 2D U-Net), but each input carries UNET_INPUT_SLICES
adjacent XCT slices stacked as input channels instead of a single slice.
That gives the model some volumetric (through-slice) context — without
the memory/compute cost of true 3D convolutions — while it still predicts
a 2D mask for just the centre slice of that stack. See config.py's
UNET_INPUT_SLICES and data/dataset.py::XCTPatchDataset._get_slice_stack
for how the stack is assembled.
=============================================================================
"""

import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ENCODER_CHANNELS, DROPOUT_RATE, UNET_INPUT_SLICES


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
    2.5D U-Net for binary defect segmentation in XCT slices.

    Architecture follows Ronneberger et al. (2015) with configurable
    encoder depth and dropout regularisation in the bottleneck. Every
    conv is a plain 2D convolution — "2.5D" refers only to the input,
    which stacks several adjacent slices as channels (see module
    docstring); the network itself never sees a depth axis.

    Input:  (B, UNET_INPUT_SLICES, H, W)  — stacked adjacent XCT slices
    Output: (B, 1, H, W)  — per-pixel defect probability in [0, 1],
                            for the centre slice of the input stack only

    Parameters
    ----------
    in_channels      : int   — input channels (defaults to config.UNET_INPUT_SLICES;
                                pass 1 for the original single-slice 2D behaviour)
    out_channels     : int   — output channels (1 for binary segmentation)
    encoder_channels : list  — feature channels at each encoder depth
    dropout_rate     : float — dropout probability in bottleneck
    """

    def __init__(
        self,
        in_channels      : int   = None,
        out_channels     : int   = 1,
        encoder_channels : list  = None,
        dropout_rate     : float = None
    ):
        super().__init__()

        in_channels  = in_channels      or UNET_INPUT_SLICES
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
        x : torch.Tensor  — (B, UNET_INPUT_SLICES, H, W)

        Returns
        -------
        torch.Tensor  — (B, 1, H, W), values in [0, 1] — mask for the
                         centre slice of the input stack
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
    """Instantiate and return a 2.5D U-Net with config defaults."""
    model = UNet2D(
        in_channels      = UNET_INPUT_SLICES,
        out_channels     = 1,
        encoder_channels = ENCODER_CHANNELS,
        dropout_rate     = DROPOUT_RATE
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [Model] 2.5D U-Net (input slices={UNET_INPUT_SLICES}) — "
          f"trainable parameters: {n_params:,}")
    return model
