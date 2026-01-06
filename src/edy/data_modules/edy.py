from edy.data_modules.base import DataModule
from edy.datasets.edy_dataset import EdyDataset
from edy.metadata.edy import PROCESSED_DATA_DIR
from edy.metadata.common import ROOT

import huggingface_hub
import zipfile


class EdyDataModule(DataModule):
    def __init__(self, batch_size: int = 1, shuffle: bool = True, num_workers: int = 1, on_gpu: bool = False):
        super().__init__(batch_size, shuffle, num_workers, on_gpu)
        self.data_dir = PROCESSED_DATA_DIR

    def prepare_data(self):
        if self.data_dir.exists():
            print("Data directory already exists.")
            return

        huggingface_hub.snapshot_download(
            repo_id="Arktetra/edy-dataset-final",
            repo_type="dataset",
            local_dir=ROOT,
        )

        with zipfile.ZipFile(ROOT / "edy.zip") as ref:
            ref.extractall()

    def setup(self):
        self.train_dataset = EdyDataset(self.data_dir)
