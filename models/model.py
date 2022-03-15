from abc import abstractmethod, ABC
import numpy as np


class Model(ABC):
    def __init__(
        self, save: bool, file_path: str, epoch: int = 10, batch_size: int = 32
    ):
        self.save = save
        self.models = self.build_model()
        self.file_path = file_path
        self.epoch = epoch
        self.batch_size = batch_size

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray, verbose: bool):
        pass
