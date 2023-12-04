from models.model import Model
from decorators.model_decorator import ModelDecorator
from utils.singleton import Singleton


class Factory(Singleton):
    def __init__(self):
        if not super().created:
            self._read_config()

    def _read_config(self):
        pass

    def __call__(self, model_to_be_wrapped: Model, *args, **kwargs) -> ModelDecorator:
        return self.get_model(*args, **kwargs)

    def get_model(model_to_be_wrapped: Model, *args, **kwargs) -> ModelDecorator:
        model = Model(*args, **kwargs)
        model = ModelDecorator(model)

        return model
