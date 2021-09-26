import torch
from torch import nn
from abc import ABC, abstractmethod
from enum import Enum

class ComplexCoord(Enum):
    RECT = 0
    POLAR = 1

class AmplitudeEncoding(Enum):
    AMP = 0
    LOG_AMP = 1
    PROB = 2
    LOG_PROB = 3

class InputEncoding(Enum):
    BINARY = 0
    INTEGER = 1

class NadeMasking(Enum):
    NONE = 0
    PARTIAL = 1
    FULL = 2

class ComplexAutoregressiveMachine_Base(nn.Module, ABC):

    def __init__(self,
                 device=None,
                 out_device="cpu",
                 output_coords=ComplexCoord.POLAR,
                 amplitude_encoding=None):
        super().__init__()

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            # I trust you know what you are doing!
            self.device = device
        self.out_device = out_device

        print(f"model device : {self.device}, out device : {self.out_device}")

        self.output_coords = output_coords
        self.amplitude_encoding = amplitude_encoding

    @abstractmethod
    def sample(self, mode=True):
        raise NotImplementedError()

    @abstractmethod
    def predict(self):
        raise NotImplementedError()