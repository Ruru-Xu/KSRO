import os
import csv
import pathlib
import nibabel as nib
import numpy as np


def get_bounding_box(mask_2d):
    y_indices, x_indices = np.nonzero(mask_2d)
    if len(y_indices) == 0:
        return None
    min_y, max_y = np.min(y_indices), np.max(y_indices)
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    return (min_x, min_y, width, height)


def process_directory(path_dir, writer):
    patients_folder = sorted([f for f in path_dir.iterdir() if f.is_dir()])
    for folder in patients_folder:
        for file in sorted(os.listdir(folder)):
            if file.endswith('.nii.gz') and '_gt' not in file and '4d' not in file:
                volume_id = file[:file.index('.nii.gz')] 
                path_img = os.path.join(folder, file)
                gt_file = volume_id + '_gt.nii.gz'
                path_label = os.path.join(folder, gt_file)

                if not os.path.exists(path_label):
                    continue

                data_img = nib.load(path_img).get_fdata()
                data_label = nib.load(path_label).get_fdata()

                data_label_binary = (data_label > 0).astype(np.uint8)

                num_slices = data_label_binary.shape[-1]
                for slice_id in range(num_slices):
                    slice_mask = data_label_binary[:, :, slice_id]
                    if np.any(slice_mask):
                        bbox = get_bounding_box(slice_mask)
                        if bbox is not None:
                            x, y, width, height = bbox
                            writer.writerow([volume_id, slice_id, x, y, width, height])


if __name__ == '__main__':
    path_train = pathlib.Path('************')
    path_val = pathlib.Path('************')
    path_test = pathlib.Path('************')
    save_root = '************'


    output_csv = os.path.join('************', 'acdc.csv')
    os.makedirs(save_root, exist_ok=True)

    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
          writer.writerow(["file", "slice", "x", "y", "width", "height"])

        process_directory(path_train, writer)
        process_directory(path_val, writer)
        process_directory(path_test, writer)

    print("save CSV:", output_csv)
