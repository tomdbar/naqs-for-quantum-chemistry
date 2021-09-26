import torch
from torch import nn
import torch.nn.functional as F

import math

from src.naqs.network.base import AmplitudeEncoding

class LogSigmoid(nn.Module):

    def forward(self, x):
        return F.sigmoid(x).log()

class _MaskedSoftmaxBase(nn.Module):

    def mask_input(self, x, mask, val):
        if mask is not None:
            m = mask.clone() # Don't alter original
            if m.dtype == torch.bool:
                x_ = x.masked_fill(~m.to(x.device), val)
            else:
                x_ = x.masked_fill((1 - m.to(x.device)).bool(), val)

            # x_ = x + (1 - mask.to(x.device)).float() * val
            # x_ = x

            # _det_mask = (mask.sum(1) >= mask.shape[-1]-1).bool()
            # mask_set = mask.clone()
            # mask_set[_det_mask] = 1 - mask_set[_det_mask]
            # mask_set[~_det_mask] *= 0
            #
            # x_ = x_.masked_fill(mask_set.to(x.device), 1)

        else:
            x_ = x
        if x_.dim() < 2:
            x_.unsqueeze_(0)
        return x_

class SoftmaxLogProbAmps(_MaskedSoftmaxBase):
    amplitude_encoding = AmplitudeEncoding.LOG_AMP
    masked_val = float('-inf')

    def forward(self, x, mask=None, dim=1):
        x_ = self.mask_input(2*x, mask, self.masked_val)
        return 0.5 * F.log_softmax(x_, dim=dim)

class SoftmaxLogProbs(_MaskedSoftmaxBase):
    amplitude_encoding = AmplitudeEncoding.LOG_AMP
    masked_val = float('-inf')

    def forward(self, x, mask=None, dim=1):
        x_ = self.mask_input(x, mask, self.masked_val)
        return 0.5 * F.log_softmax(x_, dim=dim)

class SoftmaxProbAmps(_MaskedSoftmaxBase):
    amplitude_encoding = AmplitudeEncoding.AMP
    masked_val = float('-inf')

    def forward(self, x, mask=None, dim=1):
        x_ = self.mask_input(x.pow(2), mask, self.masked_val)
        return F.softmax(x_, dim=dim).pow(0.5)

class SoftmaxProbAmpsToLogProbAmps(_MaskedSoftmaxBase):
    amplitude_encoding = AmplitudeEncoding.LOG_AMP
    masked_val = float('-inf')

    def forward(self, x, mask=None, dim=1):
        x_ = self.mask_input(x.pow(2), mask, self.masked_val)
        return F.softmax(x_, dim=dim).pow(0.5).log()

class SoftmaxProbsToLogProbAmps(_MaskedSoftmaxBase):
    amplitude_encoding = AmplitudeEncoding.LOG_AMP
    masked_val = float('-inf')

    def forward(self, x, mask=None, dim=1):
        x_ = self.mask_input(x, mask, self.masked_val)
        return F.softmax(x_, dim=dim).pow(0.5).log()


class _MaskedScaledBase(nn.Module):
    masked_val = 0

    def __init__(self, scale=math.pi):
        super().__init__()
        self.scale = scale

    def mask_input(self, x, mask, val):
        if mask is not None:
            m = mask.clone() # Don't alter original
            fixed_output = (m.sum(-1) == 1)
            m[~fixed_output] = 0
            if m.dtype == torch.bool:
                x_ = x.masked_fill(m.to(x.device), val)
            else:
                x_ = x.masked_fill(m.to(x.device).bool(), val)
        else:
            x_ = x
        if x_.dim() < 2:
            x_.unsqueeze_(0)
        return x_

class ScaledSoftSign(_MaskedScaledBase):

    def forward(self, x, mask=None):
        x = self.mask_input(x, mask, self.masked_val)
        return self.scale * F.softsign(x)

class ScaledTanh(_MaskedScaledBase):

    def forward(self, x, mask=None):
        x = self.mask_input(x, mask, self.masked_val)
        return self.scale * F.tanh(x)

class ScaledHardTanh(_MaskedScaledBase):

    def __init__(self, scale=math.pi, min=-1, max=1):
        super().__init__(scale)
        self.min, self.max = min, max

    def forward(self, x, mask=None):
        x = self.mask_input(x, mask, self.masked_val)
        return self.scale * F.hardtanh(x, min_val=self.min, max_val=self.max)

class ScaledStep(_MaskedScaledBase):

    def __init__(self, scale=math.pi, threshold=0):
        super().__init__(scale)
        self.threshold = threshold

    def forward(self, x, mask=None):
        x = self.mask_input(x, mask, self.masked_val)
        return self.scale * F.relu(torch.sign(x - self.threshold))

class ScaledSin(_MaskedScaledBase):

    def __init__(self, scale=math.pi, pow=2):
        super().__init__(scale)
        self.pow = pow

    def forward(self, x, mask=None):
        x = self.mask_input(x, mask, self.masked_val)
        return self.scale * x.sin().pow(self.pow)

class ScaledSigmoid(_MaskedScaledBase):

    def forward(self, x, mask=None):
        x = self.mask_input(x, mask, self.masked_val)
        return self.scale * F.sigmoid(x)