import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, Resize, RandomCrop, ToTensor, RandomHorizontalFlip

from nntoolbox.vision.utils import pil_to_tensor, tensor_to_pil
from nntoolbox.vision.transforms import Identity, BatchVerticalFlip, BatchRotation90, \
    BatchRotation180, BatchRotation270, BatchCompose
from torch import Tensor
from PIL import Image


class SuperResolutionizer:
    """
    Super resolutionizer. Implement geometric self-ensembling trick.

    References:

        Radu Timofte, Rasmus Rothe, Luc Van Gool. "Seven ways to improve example-based single image super resolution."
        https://arxiv.org/pdf/1511.02228.pdf
    """
    def __init__(self, model, use_geometric_ensemble: bool = True):
        self.model = model
        self.model.eval()

        self.use_geometric_ensemble = use_geometric_ensemble
        if use_geometric_ensemble:
            self.geometric_transforms = [
                Identity(), BatchVerticalFlip(), BatchRotation90(), BatchRotation180(), BatchRotation270(),
                BatchCompose([BatchRotation90(), BatchVerticalFlip()]),
                BatchCompose([BatchRotation180(), BatchVerticalFlip()]),
                BatchCompose([BatchRotation270(), BatchVerticalFlip()])
            ]

            self.inverse_transforms = [
                Identity(), BatchVerticalFlip(), BatchRotation270(), BatchRotation180(), BatchRotation90(),
                BatchCompose([BatchVerticalFlip(), BatchRotation270()]),
                BatchCompose([BatchVerticalFlip(), BatchRotation180()]),
                BatchCompose([BatchVerticalFlip(), BatchRotation90()])
            ]

    @torch.no_grad()
    def upscale(self, image: Image, upscale_factor: int) -> Image:
        return tensor_to_pil(self.upscale_tensor(pil_to_tensor(image), upscale_factor))

    @torch.no_grad()
    def upscale_tensor(self, low_res: Tensor, upscale_factor: int) -> Tensor:
        if self.use_geometric_ensemble:
            return torch.mean(torch.stack(
                [
                    self.inverse_transforms[i](
                        self.model(self.geometric_transforms[i](low_res), upscale_factor=upscale_factor)
                    )
                    for i in range(len(self.geometric_transforms))
                    ], -1
            ), -1)
        else:
            return self.model(low_res, upscale_factor=upscale_factor)
