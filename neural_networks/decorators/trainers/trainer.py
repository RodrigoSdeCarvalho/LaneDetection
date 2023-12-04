import os
import torch
import time
from abc import ABC, abstractmethod
from neural_networks.decorators.model_decorator import ModelDecorator


class Trainer(ModelDecorator, ABC):
    def __init__(self, model, name,
                 dataset, optimizer,
                 lr_scheduler=None, criterion=None):
        super().__init__(model, name)
        self._dataset = dataset

        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._criterion = criterion

    @abstractmethod
    def train(self):
        pass
