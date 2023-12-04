from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.nn import Module
import os
import numpy as np
from torch.nn.modules.module import T
from utils.path import Path
from abc import ABC, abstractmethod


class Model(Module, ABC):
    def __init__(self, name: str):
        super().__init__()
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def forward(self, x):
        pass

    def print_params(self):
        for name, param in self.named_parameters():
            print(name, param.data)

    def save(self):
        torch.save(self.state_dict(), Path().get_model(self.name))

    def load(self) -> bool:
        if os.path.exists(Path().get_model(self.name)):
            self.load_state_dict(torch.load(Path().get_model(self.name)))
            return True
        return False
