from torch import nn, Tensor
import torch.nn.functional as F
from nntoolbox.vision.components import PixelShuffleConvolutionLayer, \
    ResNeXtBlock, ConvolutionalLayer, NeuralAbstractionPyramid
from nntoolbox.components import ScalingLayer
from typing import List, Optional
import numpy as np


def leaky_relu(): return nn.LeakyReLU(0.2, True)


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


class CustomResidualBlock(ResNeXtBlock):
    def __init__(self, in_channels, activation=nn.ReLU, normalization=nn.Identity):
        super(CustomResidualBlock, self).__init__(
            branches=nn.ModuleList(
                [
                    nn.Sequential(
                        nn.ReplicationPad2d(1),
                        ConvolutionalLayer(
                            in_channels, in_channels, 3, padding=0,
                            activation=activation, normalization=normalization
                        ),
                        nn.ReplicationPad2d(1),
                        nn.Conv2d(
                            in_channels, in_channels, 3, padding=0
                        ),
                        ScalingLayer()
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
        features_branch = []
        to_residuals = []
        upsample_layers = []
        in_channels = 3

        while scale <= self.max_scale_factor:
            out_channels = 16 if in_channels < 4 else in_channels * 2
            features_branch += [
                PixelShuffleConvolutionLayer(
                    in_channels=in_channels, out_channels=out_channels,
                    normalization=nn.Identity, upscale_factor=2
                ),
                CustomResidualBlock(in_channels=out_channels, normalization=nn.Identity)
            ]
            to_residuals.append(
                nn.Sequential(
                    ConvolutionalLayer(
                        in_channels=out_channels, out_channels=3, padding=1,
                        activation=nn.Identity, normalization=nn.Identity
                    ),
                    ScalingLayer()
                )
            )
            upsample_layers.append(
                PixelShuffleConvolutionLayer(
                    in_channels=3, out_channels=3, activation=nn.Identity,
                    normalization=nn.Identity, upscale_factor=2
                )
            )
            in_channels = out_channels
            scale *= 2

        self.features_branch = nn.ModuleList(features_branch)
        self.to_residuals = nn.ModuleList(to_residuals)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: Tensor, upscale_factor: Optional[int]=None) -> Tensor:
        scale = 2
        feature = input
        output = input

        while scale <= upscale_factor:
            # upsampled = F.interpolate(output, scale_factor=2)
            upsampled = self.upsample_layers[int(np.log2(scale)) - 1](output)
            for i in range((int(np.log2(scale)) - 1) * 2, int(np.log2(scale)) * 2):
                feature = self.features_branch[i](feature)

            residual = self.to_residuals[int(np.log2(scale)) - 1](feature)
            output = self.sigmoid(residual + upsampled)
            scale *= 2

        return output

    def generate_pyramid(self, input: Tensor) -> List[Tensor]:
        scale = 2
        outputs = []

        feature = input
        output = input

        while scale <= self.max_scale_factor:
            # upsampled = F.interpolate(output, scale_factor=2)
            upsampled = self.upsample_layers[int(np.log2(scale)) - 1](output)
            for i in range((int(np.log2(scale)) - 1) * 2, int(np.log2(scale)) * 2):
                feature = self.features_branch[i](feature)

            residual = self.to_residuals[int(np.log2(scale)) - 1](feature)
            output = self.sigmoid(residual + upsampled)
            outputs.append(output)
            scale *= 2

        return outputs


class RecursiveResidualBlock(nn.Module):
    def __init__(
            self, recursive_block: nn.Module, recursive_level: int, mode: str='ss'
    ):
        """
        :param recursive_block: recursive block
        :param recursive_level: number of times to repeat the recursive block
        :param mode: if 'ss', use single-source local residual learning (i.e skip connection from original input).
        Else if 'ds , use skip connection from previous recursion level output.
        """
        super(RecursiveResidualBlock, self).__init__()
        assert mode == 'ss' or mode == 'ds'
        self.recursive_block = recursive_block
        self.recursive_level = recursive_level
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        output = input
        for r in range(self.recursive_level):
            if self.mode == 'ss': output = input + self.recursive_block(output)
            else: output = output + self.recursive_block(output)
        return output


class DeepLaplacianPyramidNetV2(nn.Module):
    """
    Second version of deep laplacian pyramid net, which reuses the layers recursively

    References:

        Wei-Sheng Lai et al. "Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution"
        https://arxiv.org/pdf/1704.03915.pdf
    """
    def __init__(self, max_scale_factor: int):
        assert max_scale_factor >= 2
        assert 2 ** (int(np.log2(max_scale_factor))) == max_scale_factor

        super(DeepLaplacianPyramidNetV2, self).__init__()
        self.max_scale_factor = max_scale_factor

        self.conv_in = ConvolutionalLayer(
            in_channels=3, out_channels=64, padding=1,
            activation=leaky_relu, normalization=nn.Identity
        )
        self.feature_embedding = RecursiveResidualBlock(
            # CustomResidualBlock(in_channels=64, activation=leaky_relu, normalization=nn.Identity),
            nn.Sequential(
                ConvolutionalLayer(
                    in_channels=64, out_channels=64, activation=leaky_relu,
                    kernel_size=3, padding=1, normalization=nn.Identity
                ),
                ConvolutionalLayer(
                    in_channels=64, out_channels=64, activation=leaky_relu,
                    kernel_size=3, padding=1, normalization=nn.Identity
                )
            ),
            recursive_level=5
        )
        self.feature_upsampling = PixelShuffleConvolutionLayer(
            in_channels=64, out_channels=64, upscale_factor=2, activation=leaky_relu, normalization=nn.Identity
        )

        self.upsampling = PixelShuffleConvolutionLayer(
            in_channels=3, out_channels=3, upscale_factor=2, activation=nn.Identity, normalization=nn.Identity
        )
        self.conv_res = ConvolutionalLayer(
            in_channels=64, out_channels=3, padding=1,
            activation=nn.Identity, normalization=nn.Identity
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: Tensor, upscale_factor: Optional[int]=None) -> Tensor:
        scale = 2
        output = input
        feature = self.conv_in(input)

        while scale <= upscale_factor:
            upsampled = self.upsampling(output)
            feature = self.feature_upsampling(self.feature_embedding(feature))

            residual = self.conv_res(feature)
            output = self.sigmoid(residual + upsampled)
            scale *= 2

        return output

    def generate_pyramid(self, input: Tensor) -> List[Tensor]:
        scale = 2
        output = input
        feature = self.conv_in(input)
        outputs = []

        while scale <= self.max_scale_factor:
            upsampled = self.upsampling(output)
            feature = self.feature_upsampling(self.feature_embedding(feature))

            residual = self.conv_res(feature)
            output = self.sigmoid(residual + upsampled)
            scale *= 2
            outputs.append(output)

        return outputs


class NAPModel(nn.Module):
    def __init__(self, n_level: int=2):
        super().__init__()
        self.n_level = n_level
        lateral_connections = [nn.Identity() for _ in range(n_level + 1)]
        feature_embedding = RecursiveResidualBlock(
            # CustomResidualBlock(in_channels=64, activation=leaky_relu, normalization=nn.Identity),
            nn.Sequential(
                ConvolutionalLayer(
                    in_channels=64, out_channels=64, activation=leaky_relu,
                    kernel_size=3, padding=1, normalization=nn.Identity
                ),
                ConvolutionalLayer(
                    in_channels=64, out_channels=64, activation=leaky_relu,
                    kernel_size=3, padding=1, normalization=nn.Identity
                )
            ),
            recursive_level=2
        )
        feature_upsampling = PixelShuffleConvolutionLayer(
            in_channels=64, out_channels=64, upscale_factor=2, activation=leaky_relu, normalization=nn.Identity
        )
        forward_connections = [nn.Sequential(feature_embedding, feature_upsampling) for _ in range(n_level)]
        backward_connections = [
            ConvolutionalLayer(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2)
            for _ in range(n_level)
        ]

        self.nap = NeuralAbstractionPyramid(
            forward_connections=forward_connections, backward_connections=backward_connections,
            lateral_connections=lateral_connections, activation_function=nn.Identity(), normalization=nn.Identity(),
            duration=3
        )
        self.conv_in = ConvolutionalLayer(
            in_channels=3, out_channels=64, padding=1,
            activation=leaky_relu, normalization=nn.Identity
        )
        self.conv_op = ConvolutionalLayer(
            in_channels=64, out_channels=3, padding=1,
            activation=nn.Identity, normalization=nn.Identity
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: Tensor, upscale_factor: Optional[int] = None) -> Tensor:
        if upscale_factor is None: upscale_factor = 2 ** self.n_level
        assert 2 ** (int(np.log2(upscale_factor))) == upscale_factor
        input = self.conv_in(input)
        output = self.nap(input, return_all_states=False)[int(np.log2(upscale_factor)) - 1]
        return self.sigmoid(self.conv_op(output))

    def generate_pyramid(self, input: Tensor) -> List[Tensor]:
        input = self.conv_in(input)
        outputs = self.nap(input, return_all_states=False)
        print(len(outputs))
        print(outputs[0])
        return [self.sigmoid(self.conv_op(op)) for op in outputs]

