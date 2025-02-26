import torch
from pytorch_msssim import ssim as py_msssim
import ast
import fastmri

def calculate_lesion_ssim(recons, target, annotations): 
    recons_abs = fastmri.complex_abs(recons)
    recons_norm = (recons_abs - recons_abs.amin(dim=(-1, -2), keepdim=True)) / (recons_abs.amax(dim=(-1, -2), keepdim=True) - recons_abs.amin(dim=(-1, -2), keepdim=True))

    recons = recons[..., 0] + 1j * recons[..., 1]
    total_ssims = []
    total_mses = []

    for i in range(recons.shape[0]):  # Iterate over each slice
        lesion_ssims = []
        lesion_mses = []

        try:
            bboxes = ast.literal_eval(annotations[i])  # Convert the string to a list of tuples
        except (ValueError, SyntaxError):
            print(f"Skipping invalid annotation for slice {i}: {annotations[i]}")
            total_ssims.append(0.0)
            continue

        for bbox in bboxes:
            x0, y0, w, h = bbox
            # Extract regions and calculate SSIM
            target_region = target[i, :, y0:y0+h, x0:x0+w]
            recons_region = recons[i, :, y0:y0+h, x0:x0+w]
            recons_norm_region = recons_norm[i, :, y0:y0+h, x0:x0+w]
            lesion_ssim = py_msssim(target_region.unsqueeze(0), recons_norm_region.unsqueeze(0), data_range=1.0, win_size=5)
            lesion_ssims.append(lesion_ssim.item())

            target_region_kspace = fastmri.fft2c(torch.stack((target_region, torch.zeros_like(target_region)), dim=-1))
            recons_region_kspace = fastmri.fft2c(torch.stack((recons_region.real, recons_region.imag), dim=-1))

            # Compute normalized MSE
            data_range_real = target_region_kspace[..., 0].max() 
            data_range_imag = target_region_kspace[..., 1].max()

            mse_real = torch.mean((target_region_kspace[..., 0] - recons_region_kspace[..., 0]) ** 2).item()
            mse_imag = torch.mean((target_region_kspace[..., 1] - recons_region_kspace[..., 1]) ** 2).item()

            mse_real_normalized = mse_real / (data_range_real ** 2)
            mse_imag_normalized = mse_imag / (data_range_imag ** 2)

            combined_mse = (mse_real_normalized + mse_imag_normalized) / 2
            lesion_mses.append(combined_mse)

        slice_ssim = torch.mean(torch.tensor(lesion_ssims, dtype=torch.float32))
        total_ssims.append(slice_ssim.item())

        slice_mse = torch.mean(torch.tensor(lesion_mses, dtype=torch.float32))
        total_mses.append(slice_mse.item())

    return torch.tensor(total_ssims, device='cuda'), torch.tensor(total_mses, device='cuda')



def compute_lesion_priority_kspace(target, annotation):
    """
    Compute a lesion priority map based on annotations.
    """
    temp = target.clone()
    lesion_priority_map = torch.zeros_like(temp)

    for i in range(temp.shape[0]):  # Iterate over the batch dimension
        # Parse the bounding boxes from the annotations
        bboxes = ast.literal_eval(annotation[i])

        # Reverse the y-coordinates of bounding boxes to match the original target orientation
        for bbox in bboxes:
            x0, y0, w, h = bbox
            target_region = temp[i, :, y0:y0+h, x0:x0+w]
            lesion_priority_map[i, :, y0:y0+h, x0:x0+w] = target_region
    lesion_kspace = fastmri.fft2c(torch.stack((lesion_priority_map, torch.zeros_like(lesion_priority_map)), dim=-1))
    lesion_kspace = lesion_kspace[..., 0] + 1j * lesion_kspace[..., 1]
    return lesion_kspace
