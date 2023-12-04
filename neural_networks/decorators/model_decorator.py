import torch
from utils.config import DecoratorConfig
from abc import ABC, abstractmethod


class ModelDecorator(ABC):
    def __init__(self, model, name):
        if torch.backends.cudnn.version() < 8000:
            torch.backends.cudnn.benchmark = True
        self._name = name
        self._model = model

        if DecoratorConfig().device == 'cuda':
            self.send_to_cuda()

    @property
    def name(self):
        return self._name

    def send_to_cuda(self):
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        self._model.to(device)

    def save(self):
        self._model.save()

    def load(self):
        self._model.load()
