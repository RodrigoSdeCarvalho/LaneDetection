from essentials.add_module import set_working_directory
set_working_directory()

from utils.logger import Logger
from neural_networks.decorators.trainers.enet_trainer import EnetTrainer


def train_enet():
    trainer = EnetTrainer()
    trainer.train()


if __name__ == '__main__':
    train_enet()
