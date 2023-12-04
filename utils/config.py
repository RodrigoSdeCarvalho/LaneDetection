from utils.singleton import Singleton
import json, os


class Config(Singleton):
    def __init__(self):
        if not super().created:
            self._config_json = self._load_config()

    def _load_config(self) -> dict:
        """Loads the config file"""
        with open(os.path.join(os.getcwd(), 'config.json')) as file:
            return json.load(file)['config']

    @property
    def log(self) -> bool:
        """Returns whether the log is enabled or not"""
        return self._config_json['log']

    @log.setter
    def log(self, value: bool):
        """Sets the log value"""
        self._config_json['log'] = value

    @property
    def save(self) -> bool:
        """Returns whether the save is enabled or not"""
        return self._config_json['save']

    def log_type(self, type: str) -> dict:
        """Returns the log type"""
        return self._config_json[type]

    @property
    def profile(self) -> str:
        """Returns the profile"""
        return self._config_json['profile']


class ModelConfig(Config):
    def __init__(self):
        super().__init__()

    def _load_config(self) -> dict:
        """Loads the config file"""
        with open(os.path.join(os.getcwd(), 'config.json')) as file:
            return json.load(file)['config']['model']

    @property
    def name(self) -> str:
        """Returns the name of the model"""
        return self._config_json['name']

    @property
    def batch_size(self) -> int:
        """Returns the batch size"""
        return self._config_json['batch_size']

    @property
    def epochs(self) -> int:
        """Returns the number of epochs"""
        return self._config_json['epochs']


class DecoratorConfig(Config):
    def __init__(self):
        super().__init__()

    def _load_config(self) -> dict:
        """Loads the config file"""
        with open(os.path.join(os.getcwd(), 'config.json')) as file:
            return json.load(file)['config']['decorator']

    @property
    def name(self) -> str:
        """Returns the name of the decorator"""
        return self._config_json['name']

    @property
    def device(self) -> str:
        """Returns the device"""
        return self._config_json['device']

    @property
    def max_distance(self) -> int:
        """Returns the max distance"""
        return self._config_json['max_distance']

    @property
    def frames_to_remember(self) -> int:
        """Returns the number of frames to remember"""
        return self._config_json['frames_to_remember']


class DatasetConfig(Config):
    def __init__(self):
        super().__init__()

    def _load_config(self) -> dict:
        """Loads the config file"""
        with open(os.path.join(os.getcwd(), 'config.json')) as file:
            return json.load(file)['config']['dataset']

    @property
    def name(self) -> str:
        """Returns the name of the dataset"""
        return self._config_json['name']


class ImageConfig(Config):
    def __init__(self):
        super().__init__()

    def _load_config(self) -> dict:
        """Loads the config file"""
        with open(os.path.join(os.getcwd(), 'config.json')) as file:
            return json.load(file)['config']['image']

    @property
    def width(self) -> int:
        """Returns the width of the image"""
        return self._config_json['width']

    @property
    def height(self) -> int:
        """Returns the height of the image"""
        return self._config_json['height']

    @property
    def preprocess_width(self) -> int:
        """Returns the preprocess width of the image"""
        return self._config_json['preprocess_width']

    @property
    def preprocess_height(self) -> int:
        """Returns the preprocess height of the image"""
        return self._config_json['preprocess_height']

    @property
    def channels(self) -> int:
        """Returns the number of channels of the image"""
        return self._config_json['channels']

    @property
    def size(self) -> tuple:
        """Returns the size of the image"""
        return self.width, self.height

    @property
    def shape(self) -> tuple:
        """Returns the shape of the image"""
        return self.width, self.height, self.channels

    @property
    def bev(self) -> dict[str, int]:
        return self._config_json['bev']

    @property
    def calibration(self) -> dict[str, int]:
        return self._config_json['calibration']
