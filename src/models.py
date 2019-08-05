from torch import nn
from nntoolbox.vision.components import PixelShuffleConvolutionLayer, ResNeXtBlock, ConvolutionalLayer


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


class PixelShuffleDecoderV2(nn.Sequential):
    def __init__(self):
        super(PixelShuffleDecoderV2, self).__init__(
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
