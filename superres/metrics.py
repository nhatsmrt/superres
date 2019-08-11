from nntoolbox.metrics import Metric
from typing import Dict, Any
import torch
import math
from torch.nn.functional import mse_loss


class PSNR(Metric):
    """
    Peak Signal-to-Noise Ratio:

    PSNR = 10 log_10( MAX^2_I / MSE) = 20 log_10(MAX_I) - 10 log_10 (MSE)

    References:

        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    def __init__(self):
        self._best = float('-inf')

    def __call__(self, logs: Dict[str, Any]) -> float:
        outputs = (logs['outputs'] * 255).to(torch.uint8).to(torch.float32)
        labels = (logs['labels'] * 255).to(torch.uint8).to(torch.float32)
        mse = mse_loss(outputs, labels).cpu().detach().numpy()
        psnr = 20 * math.log10(255) - 10 * math.log10(mse)

        if psnr > self._best:
            self._best = psnr

        return psnr
