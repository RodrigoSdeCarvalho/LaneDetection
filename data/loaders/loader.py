from abc import ABC, abstractmethod


class Loader(ABC):
    def __init__(self, config, src_path=None):
        self._config = config
        self._src_path = src_path

    @abstractmethod
    def get(self):
        pass

    @property
    @abstractmethod
    def src_path(self):
        pass
