from utils.singleton import Singleton
import os
from enum import Enum
import datetime
from utils.config import Config


class Logger(Singleton):
    class Type(Enum):
        INFO = 'INFO'
        ERROR = 'ERROR'
        TRACE = 'TRACE'
        WARNING = 'WARNING'

    log_is_on: bool = None

    log_type_is_on: dict = {
        "info": None,
        "error": None,
        "trace": None,
        "warning": None
    }

    @staticmethod
    def _should_log():
        if Logger.log_is_on is None:
            Logger._check_log_config()
        return Logger.log_is_on

    @classmethod
    def _check_log_config(cls):
        cls.log_is_on = Config().log

    @staticmethod
    def _should_log_type(type: str):
        if Logger.log_type_is_on[type] is None:
            Logger._check_log_type_config(type)
        return Logger.log_type_is_on[type]

    @classmethod
    def _check_log_type_config(cls, type: str):
        cls.log_type_is_on[type] = Config().log_type(type)

    @staticmethod
    def info(message: str, show=True, save=False):
        if not save:
            save = Config().save
        if Logger._should_log():
            if Logger._should_log_type("info"):
                Logger().log(message, Logger.Type.INFO, show,save)

    @staticmethod
    def error(message: str, show=True, save=False):
        if not save:
            save = Config().save
        if Logger._should_log():
            if Logger._should_log_type("error"):
                Logger().log(message, Logger.Type.ERROR, show,save)

    @staticmethod
    def trace(message: str, show=True, save=False):
        if not save:
            save = Config().save
        if Logger._should_log():
            if Logger._should_log_type("trace"):
                Logger().log(message, Logger.Type.TRACE, show,save)

    @staticmethod
    def warning(message: str, show=True, save=False):
        if not save:
            save = Config().save
        if Logger._should_log():
            if Logger._should_log_type("warning"):
                Logger().log(message, Logger.Type.WARNING, show,save)

    def __init__(self):
        if not super().created:
            self._filename = f"log_{datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.txt"
            if Config().save:
                self._create_log_file()

    def _create_log_file(self):
        """Creates a log file if it doesn't exist"""
        path = self._path()
        if not os.path.exists(path):
            with open(path, 'w') as file:
                file.write('Lane Detection Logs\n')

    def _path(self):
        """Returns the path to the log file"""
        return os.path.join(os.getcwd(), 'logs', self._filename)

    def log(self, message: str, type: Type = Type.TRACE, show: bool=False, save: bool=False):
        """Logs a message to a file"""
        self._add_message(type, message, save)
        if show:
            print(self._build_message(type, message))

    def _add_message(self, type: Type, message: str, save: bool):
        """Adds a message to the log file"""
        message = self._build_message(type, message)
        if save:
            with open(self._path(), 'a') as file:
                file.write(message + '\n')

    def _build_message(self, type: Type, message: str) -> str:
        return f"[{type.value}] - {datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}: {message}"