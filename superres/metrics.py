from nntoolbox.metrics import Metric
from typing import Dict, Any
import torch
import math
from torch.nn.functional import mse_loss
import numpy as np
from nntoolbox.utils import compute_num_batch
from typing import Optional


class PSNR(Metric):
    """
    Peak Signal-to-Noise Ratio:

    PSNR = 10 log_10( MAX^2_I / MSE) = 20 log_10(MAX_I) - 10 log_10 (MSE)

    References:

        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    def __init__(self, batch_size: Optional[int]=None):
        self._best = float('-inf')
        self._batch_size = batch_size

    def __call__(self, logs: Dict[str, Any]) -> float:
        batch_size = len(logs['outputs']) if self._batch_size is None else self._batch_size
        n_batch = compute_num_batch(len(logs['outputs']), batch_size)
        psnrs = []

        for batch_idx in range(n_batch):
            outputs = (
                logs['outputs'][batch_idx * batch_size:(batch_idx + 1) * batch_size] * 255
            ).to(torch.uint8).to(torch.float32)
            labels = (
                logs['labels'][batch_idx * batch_size:(batch_idx + 1) * batch_size] * 255
            ).to(torch.uint8).to(torch.float32)
            mse = mse_loss(outputs, labels).cpu().detach().numpy()
            psnr = 20 * math.log10(255) - 10 * math.log10(mse)
            psnrs.append(psnr)

        psnr = np.mean(psnrs)
        if psnr > self._best:
            self._best = psnr

        return psnr
