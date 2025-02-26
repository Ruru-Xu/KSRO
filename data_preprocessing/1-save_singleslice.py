import os
from typing import Dict, Tuple

import torch
from tqdm import tqdm

import h5py
from joblib import load, dump

import pandas as pd
import numpy as np
import fastmri
import nibabel as nib

import nibabel as nib
import numpy as np


def center_crop_img(file_name, crop_size=256, pad_size=512):
    image = nib.load(file_name).get_fdata()
    H, W, S = image.shape 

    background = np.zeros((pad_size, pad_size, S), dtype=image.dtype)

    start_h = (pad_size - H) // 2
    start_w = (pad_size - W) // 2
    background[start_h:start_h + H, start_w:start_w + W, :] = image

    final_start_h = (pad_size - crop_size) // 2
    final_start_w = (pad_size - crop_size) // 2
    final_image = background[final_start_h:final_start_h + crop_size, final_start_w:final_start_w + crop_size, :]

    return final_image, (start_h, start_w, final_start_h, final_start_w)


def norm_img(data_img):

    slice_min = data_img.min(axis=(0, 1), keepdims=True)  # shape: (1,1,Slices)
    slice_max = data_img.max(axis=(0, 1), keepdims=True)  # shape: (1,1,Slices)

    denominator = slice_max - slice_min
    denominator[denominator == 0] = 1

    data_img_norm = (data_img - slice_min) / denominator 
    return data_img_norm
def img2kspace(data_img_norm):

    H, W, S = data_img_norm.shape
    kspace = np.zeros((H, W, S), dtype=np.complex64)
    for i in range(S):
        kspace_slice = np.fft.fft2(data_img_norm[:, :, i])
        kspace[:, :, i] = kspace_slice
    return dict(kspace=kspace)

'''
# check id kspace correct:
import matplotlib.pyplot as plt
reconstructed = np.fft.ifft2(kspace_slice)
plt.imshow(torch.abs(reconstructed), cmap='bone')
plt.show()
'''

def processed_kspace(file_name):
    img_crop, offsets = center_crop_img(file_name)
    target = norm_img(img_crop)
    data_dict = img2kspace(target)
    kspace = torch.tensor(data_dict['kspace'])
    data_dict['kspace'] = kspace
    data_dict['target'] = target
    data_dict['offsets'] = offsets 
    return data_dict


def save_single_slice(file_name: str,
                     annotations: pd.DataFrame,
                     save_root: str,
                     meta_data: Dict):
    data_dict = processed_kspace(file_name)

    if data_dict is None:
        return None

    volume_id = file_name.split("/")[-1].replace(".nii.gz", "")
    volume_path = os.path.join(save_root, volume_id)
    os.makedirs(volume_path)


    start_h, start_w, final_start_h, final_start_w = data_dict['offsets']

 
    df = annotations[annotations.file == volume_id].copy()

    df['x'] = df['x'] + (start_w - final_start_w)
    df['y'] = df['y'] + (start_h - final_start_h)

    annotations.loc[df.index, ['x', 'y']] = df[['x', 'y']]


    num_slice = data_dict['kspace'].shape[2]
    for slice_id in range(num_slice):

        # make slice path
        slice_filename = f'{volume_id}_{slice_id}.h5'

        slice_path = os.path.join(volume_path, slice_filename)

        hf = h5py.File(slice_path, 'w')
        hf.create_dataset('sc_kspace', data=data_dict['kspace'][:, :, slice_id])
        hf.create_dataset('target', data=data_dict['target'][:, :, slice_id])
        hf.close()

        meta_data[f'knee_{volume_id}_{slice_id}'] = dict(volume_id=volume_id,
                                                         slice_id=slice_id,
                                                         shape=data_dict['kspace'].shape,
                                                         dataset='acdc')

def list_all_files(root_dir):
    files_paths = []
    for patient in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient)
        if os.path.isdir(patient_path):
  
            for f in os.listdir(patient_path): 
                if "frame" in f and f.endswith(".nii.gz") and "_gt.nii.gz" not in f:
                    files_paths.append(os.path.join(patient_path, f))
    return files_paths

def main():
    annotations = pd.read_csv('*****')
    train_path = '*****'
    val_path = '*****'
    test_path = '*****'
    save_generated_files = '*****'
    save_meta = '*****'

    train_paths = list_all_files(train_path)
    val_paths = list_all_files(val_path)
    test_paths = list_all_files(test_path)

    paths = train_paths + val_paths + test_paths
    meta_data = {}
    for i, file_name in enumerate(tqdm(paths)):
        try: 
            save_single_slice(file_name=file_name,
                             annotations=annotations,
                             save_root=save_generated_files,
                             meta_data=meta_data)
        except OSError:
            pass

    meta_data = pd.DataFrame(meta_data).T
    dump(meta_data, os.path.join(save_meta, 'meta_data.p'))
    annotations.to_csv('*****', index=False)


if __name__ == '__main__':
    main()
