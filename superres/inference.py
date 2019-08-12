from torch import nn


class MSSuperResolutionizer:
    def __init__(self, model: nn.Module, max_upscale: int):
        self.model = model
        self.max_upscale = max_upscale