from edy.data_modules.base import DataModule
from edy.datasets.edy_dataset import EdyDataset
# from edy.metadata.edy import PROCESSED_DATA_DIR

import huggingface_hub
import torch
import zipfile

from pathlib import Path

from torch.utils.data import random_split


class EdyDataModule(DataModule):
    def __init__(
        self, root: str, batch_size: int = 1, shuffle: bool = True, num_workers: int = 1, on_gpu: bool = False
    ):
        super().__init__(batch_size, shuffle, num_workers, on_gpu)
        self.data_dir = Path(root) / "data" / "processed" / "EDY"
        self.root = root

    def prepare_data(self):
        if self.data_dir.exists():
            print("Data directory already exists.")
            return

        huggingface_hub.snapshot_download(
            repo_id="Darktetra/edy-dataset-final",
            repo_type="dataset",
            local_dir=self.root,
        )

        with zipfile.ZipFile(f"{self.root}/edy.zip") as ref:
            ref.extractall()

    def setup(self):
        dataset = EdyDataset(self.data_dir)
        generator = torch.Generator().manual_seed(41)
        self.train_dataset, self.val_dataset = random_split(dataset, [0.9, 0.1], generator=generator)
        self.val_dataset, self.test_dataset = random_split(self.val_dataset, [0.8, 0.2], generator=generator)
