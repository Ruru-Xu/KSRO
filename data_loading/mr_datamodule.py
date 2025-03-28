import argparse
from pathlib import Path
import joblib
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .acdc_data import ACDCDataset

class MRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: argparse.ArgumentParser,
    ):
        super().__init__()
        self.config = config

        if self.config.use_weighted_sampler:
            assert Path(self.config.train_sampler_filename).is_file()
            self.train_sampler_weights = joblib.load(self.config.train_sampler_filename)
        else:
            print(
                "***" * 10,
                #"Not using Weighted Random Sampler For Training RM",
                "***" * 10,
            )
            self.train_sampler_weights = None

        self.setup()

    def setup(self, stage=None):
        self.train_dataset = ACDCDataset(config=self.config, mode="train_recon")
        self.val_dataset = ACDCDataset(config=self.config, mode="val_recon")

    def train_dataloader(self):
        sampling_dict = dict(shuffle=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            drop_last=True,
            pin_memory=True,
            **sampling_dict,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            num_workers=self.config.num_workers,
            shuffle=self.config.val_shuffle,
        )
