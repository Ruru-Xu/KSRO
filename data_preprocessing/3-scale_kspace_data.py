import torch
import numpy as np
import pandas as pd
import h5py
import tqdm
from typing import Tuple

import fastmri
from fastmri.data import transforms as T


def center_crop(data, shape: Tuple[int, int]):
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def process_file(file_path, mean, Rrr, Rii, Rri, output_key="sc_kspace_scaled"):
    # Load and process k-space data for a single file
    with h5py.File(file_path, 'r+') as file:
        if output_key in file:
            return  # Skip if already processed

        kspace_data = file['sc_kspace'][()]
        data_fft = fastmri.ifft2c(T.to_tensor(kspace_data).float())
        data_fft = torch.complex(data_fft[..., 0], data_fft[..., 1]).type(torch.complex64)

        # Scale data
        slice_out = data_fft - mean
        scaled_data = (Rrr[None, None] * slice_out.real + Rri[None, None] * slice_out.imag).type(torch.complex64) \
                      + 1j * (Rii[None, None] * slice_out.imag + Rri[None, None] * slice_out.real).type(torch.complex64)

        # Write scaled data to file
        file.create_dataset(output_key, data=scaled_data.numpy())


def compute_mean_cov(data_paths):
    # Compute mean and covariance
    mean_real, mean_imag, numel = 0, 0, 0
    sum_real, sum_imag = 0, 0
    with torch.no_grad():
        for file_path in tqdm.tqdm(data_paths, desc="Computing mean and covariance"):
            with h5py.File(file_path, 'r') as file:
                kspace_data = file['sc_kspace'][()]
                data_fft = fastmri.ifft2c(T.to_tensor(kspace_data).float())
                data_fft = torch.complex(data_fft[..., 0], data_fft[..., 1]).type(torch.complex64)

                sum_real += data_fft.real.sum()
                sum_imag += data_fft.imag.sum()
                numel += data_fft.numel()

        # Calculate mean
        mean_real = sum_real / numel
        mean_imag = sum_imag / numel
        mean = torch.complex(mean_real, mean_imag)

        # Compute covariance
        Crr, Cii, Cri = 0, 0, 0
        for file_path in tqdm.tqdm(data_paths, desc="Computing covariance matrix"):
            with h5py.File(file_path, 'r') as file:
                kspace_data = file['sc_kspace'][()]
                data_fft = fastmri.ifft2c(T.to_tensor(kspace_data).float())
                data_fft = torch.complex(data_fft[..., 0], data_fft[..., 1]).type(torch.complex64)

                centered_data = data_fft - mean
                Crr += centered_data.real.pow(2).sum()
                Cii += centered_data.imag.pow(2).sum()
                Cri += (centered_data.real * centered_data.imag).sum()

        # Normalize covariance
        Crr /= numel
        Cii /= numel
        Cri /= numel
        eps = 1e-8  # Avoid division by zero

        # Calculate the inverse square root of the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det + eps)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t + eps)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

    return mean, Rrr, Rii, Rri


if __name__ == '__main__':
    csv_file = "*****"
    df = pd.read_csv(csv_file)
    df_train = df[df.data_split == "train_recon"]
    df_val = df[df.data_split == "val_recon"]
    df_test = df[df.data_split == "test_recon"]
    print("Training examples:", len(df_train), "Validation examples:", len(df_val), "Test examples:", len(df_test))

    # Get paths for training data
    list_train_paths = list(df_train.location)

    # Compute mean and covariance from training data
    mean, Rrr, Rii, Rri = compute_mean_cov(list_train_paths)

    # Process training, validation, and test sets
    print("Processing training data...")
    for file_path in tqdm.tqdm(list_train_paths, desc="Scaling training data"):
        process_file(file_path, mean, Rrr, Rii, Rri)

    for df_part, label in zip([df_val, df_test], ["validation", "test"]):
        print(f"Processing {label} data...")
        for file_path in tqdm.tqdm(df_part.location, desc=f"Scaling {label} data"):
            process_file(file_path, mean, Rrr, Rii, Rri)
