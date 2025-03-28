import argparse
import h5py
import numpy as np
from .mr_dataset import MRDataset

class ACDCDataset(MRDataset):
    def __init__(
        self,
        config: argparse.ArgumentParser,
        mode: str,
    ):
        super().__init__(config=config, mode=mode)

    def __getitem__(self, index):
        assert self.mode in self.metadata_by_mode
        loc = self.get_metadata_value(index, "location")
        kspace_key, target_key = "sc_kspace", "target"
        with h5py.File(loc) as f:
            kspace_arr = f[kspace_key][:]
            target_arr = f[target_key][:]

        info = self.metadata_by_mode[self.mode].iloc[index]
        kspace_arr = np.expand_dims(kspace_arr, axis=0)
        
        parameters = {
            kspace_key: kspace_arr,
            target_key: target_arr,
            "volume_id": info.volume_id,
            "slice_id": info.slice_id,
            "data_split": info.data_split,
            "location": info.location,
            'annotation': info.annotation,
        }

        return parameters
