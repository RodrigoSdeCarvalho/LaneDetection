import os
from os import path as syspath
from utils.singleton import Singleton


class Path(Singleton):
    def __init__(self):
        if not super().created:
            self._root = syspath.join(os.getcwd())

    @property
    def root(self):
        return self._root

    @property
    def assets(self):
        return syspath.join(self._root, 'assets')

    @property
    def data(self):
        return syspath.join(self.assets, 'data')

    @property
    def video(self):
        return syspath.join(self.data, 'video')

    def get_video(self, video_name):
        return syspath.join(self.video, video_name)

    @property
    def train_data(self):
        return syspath.join(self.data, 'train')

    def train_dataset(self, dataset_name):
        return syspath.join(self.train_data, dataset_name)

    @property
    def test_data(self):
        return syspath.join(self.data, 'test')

    def test_dataset(self, dataset_name):
        return syspath.join(self.test_data, dataset_name)

    @property
    def models(self):
        return syspath.join(self._root, 'assets', 'models')

    @property
    def outputs(self):
        return syspath.join(self._root, 'outputs')

    def get_output(self, output_name):
        return syspath.join(self.outputs, output_name)

    def get_model(self, model_name):
        model_name = model_name + ".pt"
        return syspath.join(self.models, model_name)

    def __call__(self, *args, **kwargs):
        return syspath.join(self._root, *args)
