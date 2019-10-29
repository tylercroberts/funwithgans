import torch
import random
from pathlib import Path

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract Base Class for all models inside project folders."""

    def __init__(self, args, train=True, logger=None):
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
        self.train = train
        self.storage_dir = args.storage_dir
        self.model_dir = args.model_dir
        self.image_dir = args.image_dir
        self.data_dir = args.data_dir
        self.log_dir = args.log_dir
        self.logger = logger
        self.random_seed = 903 if bool(args.reproducible) is True else random.randint(1, 100000)
        self.img_dim = args.img_dim
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.epochs = args.epochs
        self.beta1 = args.beta
        self.ngpu = args.ngpu
        self.models = list()
        self.model_name = ''

        logger.info(f"Random seed is {self.random_seed}")
        random.seed(self.random_seed)

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def update_parameters(self):
        pass

    @abstractmethod
    def load_networks(self):
        pass

    def save_networks(self):
        for i in self.models:
            torch.save(self.__getattribute__(i), Path(self.model_dir) / f'{self.model_name}_{i}.pt')

    @abstractmethod
    def update_learning_rate(self):
        """Needed if using a scheduler"""
        pass

    @abstractmethod
    def predict_test(self):
        with torch.no_grad():
            self.forward()

