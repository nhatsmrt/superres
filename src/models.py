from torch import nn, Tensor
import torch.nn.functional as F
from nntoolbox.vision.components import PixelShuffleConvolutionLayer, ResNeXtBlock, ConvolutionalLayer
from typing import List, Optional
import numpy as np


class CustomResidualBlockPreActivation(ResNeXtBlock):
    def __init__(self, in_channels, activation=nn.ReLU, normalization=nn.BatchNorm2d):
        super(CustomResidualBlockPreActivation, self).__init__(
            branches=nn.ModuleList(
                [
                    nn.Sequential(
                        nn.ReplicationPad2d(1),
                        ConvolutionalLayer(
                            in_channels, in_channels, 3, padding=0,
                            activation=activation, normalization=normalization
                        ),
                        nn.ReplicationPad2d(1),
                        ConvolutionalLayer(
                            in_channels, in_channels, 3, padding=0,
                            activation=activation, normalization=normalization
                        )
                    )
                ]
            ),
            use_shake_shake=False
        )


class PixelShuffleUpsampler(nn.Sequential):
    def __init__(self):
        super(PixelShuffleUpsampler, self).__init__(
            PixelShuffleConvolutionLayer(
                in_channels=3, out_channels=16,
                normalization=nn.Identity, upscale_factor=2
            ),
            CustomResidualBlockPreActivation(in_channels=16, normalization=nn.Identity),
            PixelShuffleConvolutionLayer(
                in_channels=16, out_channels=32,
                normalization=nn.Identity, upscale_factor=2
            ),
            CustomResidualBlockPreActivation(in_channels=32, normalization=nn.Identity),
            PixelShuffleConvolutionLayer(
                in_channels=32, out_channels=3, activation=nn.Sigmoid,
                normalization=nn.Identity, upscale_factor=2
            )
        )


class DeepLaplacianPyramidNet(nn.Module):
    """
    References:

        Wei-Sheng Lai et al. "Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution"
        https://arxiv.org/pdf/1704.03915.pdf
    """
    def __init__(self, max_scale_factor: int):
        assert max_scale_factor >= 2
        assert 2 ** (int(np.log2(max_scale_factor))) == max_scale_factor

        super(DeepLaplacianPyramidNet, self).__init__()
        self.max_scale_factor = max_scale_factor

        scale = 2
        layers = []
        to_residuals = []
        in_channels = 3

        while scale <= self.max_scale_factor:
            out_channels = 16 if in_channels < 4 else in_channels * 2
            layers += [
                PixelShuffleConvolutionLayer(
                    in_channels=in_channels, out_channels=out_channels,
                    normalization=nn.Identity, upscale_factor=2
                ),
                CustomResidualBlockPreActivation(in_channels=out_channels, normalization=nn.Identity)
            ]
            to_residuals.append(
                ConvolutionalLayer(in_channels=out_channels, out_channels=3, activation=nn.Identity, padding=1)
            )
            in_channels = out_channels
            scale *= 2

        self.layers = nn.ModuleList(layers)
        self.to_residuals = nn.ModuleList(to_residuals)

    def forward(self, input: Tensor, upscale_factor: Optional[int]=None) -> Tensor:
        scale = 2
        feature = input
        output = input

        while scale <= self.max_scale_factor:
            upsampled = F.interpolate(output, scale_factor=2)
            for i in range((int(np.log2(scale)) - 1) * 2, int(np.log2(scale)) * 2):
                feature = self.layers[i](feature)

            residual = self.to_residuals[int(np.log2(scale)) - 1](feature)
            output = residual + upsampled
            scale *= 2

        return output

    def generate_pyramid(self, input: Tensor) -> List[Tensor]:
        scale = 2
        outputs = []

        feature = input
        output = input

        while scale <= self.max_scale_factor:
            upsampled = F.interpolate(output, scale_factor=2)
            for i in range((int(np.log2(scale)) - 1) * 2, int(np.log2(scale)) * 2):
                feature = self.layers[i](feature)

            residual = self.to_residuals[int(np.log2(scale)) - 1](feature)
            output = residual + upsampled
            outputs.append(output)
            scale *= 2

        return outputs
