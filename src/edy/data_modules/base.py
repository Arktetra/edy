from abc import abstractmethod
from torch.utils.data import DataLoader


class DataModule:
    def __init__(self, batch_size: int = 1, shuffle: bool = True, num_workers: int = 1, on_gpu: bool = False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.on_gpu = on_gpu

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    @abstractmethod
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    @abstractmethod
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )
