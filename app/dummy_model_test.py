from essentials.add_module import set_working_directory
set_working_directory()

from utils.logger import Logger
from neural_networks.models.lanenet.enet import ENet


def dummy_main():
    logger = Logger()
    model = ENet(2, 4)
    model.save()
    model.load()
    logger.log("Model saved and loaded successfully.")


if __name__ == '__main__':
    dummy_main()
