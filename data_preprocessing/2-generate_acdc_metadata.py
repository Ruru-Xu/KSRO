import os
import h5py
import tqdm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch


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
    train_path = '*****'
    val_path = '*****'
    test_path = '*****'
    df_acdc = pd.read_csv('*****')

    train_paths = list_all_files(train_path)
    val_paths = list_all_files(val_path)
    test_paths = list_all_files(test_path)
    train_vols, val_vols, test_vols = train_paths, val_paths, test_paths # Total number of volumes
    train_recon = [vol.split('/')[-1].replace('.nii.gz', '') for vol in train_vols]
    val_recon = [vol.split('/')[-1].replace('.nii.gz', '') for vol in val_vols]
    test_recon = [vol.split('/')[-1].replace('.nii.gz', '') for vol in test_vols]

    def get_data_split(file_name):
        if file_name in set(train_recon):
            return "train_recon"
        elif file_name in set(val_recon):
            return "val_recon"
        elif file_name in set(test_recon):
            return "test_recon"
        return None

    # Group by file and slice, then collect lists of bounding box coordinates for each slice
    df_acdc_file_slice_level = df_acdc.groupby(['file', 'slice']).apply(
        lambda x: x[['x', 'y', 'width', 'height']].to_dict(orient='list')
    ).reset_index()

    # Create a unique key for each file and slice combination
    df_acdc_file_slice_level['key'] = df_acdc_file_slice_level['file'] + "_" + df_acdc_file_slice_level['slice'].astype(str)

    # Create a dictionary to store bounding boxes as lists of (x, y, width, height) tuples
    file_box_dict = {
        row['key']: list(zip(row[0]['x'], row[0]['y'], row[0]['width'], row[0]['height']))
        for _, row in df_acdc_file_slice_level.iterrows()
    }

    def get_box(file_name):
        """Retrieve all bounding boxes for a given file slice combination."""
        return file_box_dict.get(file_name, [None])

    index, volume_ids, slice_ids, shapes, box = [], [], [], [], []
    data_split, locations = [], []

    root_path = '*****'
    folder_list = os.listdir(root_path)
    for folder in tqdm.tqdm(folder_list):
        try:
            files_path = os.path.join(root_path, folder)
            files_in_folder = os.listdir(files_path)

            for file_in_folder in files_in_folder:
                # file_in_folder contains .h5
                file_path = os.path.join(files_path, file_in_folder)
                file = h5py.File(file_path)
                kspace = file['sc_kspace']

                vol_name = folder
                slice_id = file_in_folder.split('.')[0].split('_')[-1]
                shape = kspace.shape

                data_split_val = get_data_split(vol_name)
                box_val = get_box(f"{vol_name}_{slice_id}")

                index_str = f'knee_{file_in_folder.split(".")[0]}'

                index.append(index_str)
                volume_ids.append(vol_name)
                slice_ids.append(slice_id)
                box.append(box_val)
                shapes.append(shape)
                data_split.append(data_split_val)
                locations.append(file_path)
        except OSError:
            pass

    metadata_df = pd.DataFrame({
        'index': index,
        'volume_id': volume_ids,
        'slice_id': slice_ids,
        'annotation': box,
        'shape': shapes,
        'data_split': data_split,
        'location': locations
    })

    metadata_df.to_csv('*****' + 'metadata_acdc.csv')


if __name__ == '__main__':
    main()
