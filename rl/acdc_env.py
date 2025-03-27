import fastmri
import torch
import numpy as np
from pytorch_msssim import ssim as py_msssim
from torch.utils.data import DataLoader
import ast
from rl.calculate_lesion_ssim import calculate_lesion_ssim, compute_lesion_priority_kspace
import matplotlib.pyplot as plt

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class ACDC_Env:

    def __init__(self, data_loader, observation_space=(1, 256, 256), device='cuda',
                 k_fraction=0.1, eval=False, fixed_budget=False, sampled_indices=[0, 255],
                 scale_reward=True, reward_mode=10, srange=[0, 255], delay_step=2,
                 evaluation_only=False, budget=32):
        # Environment properties and initialization
        self.state = 0
        self.done = False
        self.data_loader = data_loader
        self.data_loader_iter = iter(self.data_loader)
        self.sampled_indices = sampled_indices
        self.observation_space = observation_space
        self.action_space = Namespace(n=sampled_indices[1] - sampled_indices[0] + 1)
        self.act_dim = self.action_space.n
        self.num_envs = data_loader.batch_size
        self.device = device
        self.k_fraction = k_fraction
        self.eval = eval
        self.fixed_budget = fixed_budget
        self.scale_reward = scale_reward
        self.reward_mode = reward_mode
        self.previous_ssim_lesion = None
        self.previous_mse_lesion = None
        self.previous_ssim_global = None
        self.delay_step = delay_step
        self.srange = srange
        self.evaluation_only = evaluation_only
        self.budget = budget

    def get_remain_epi_lines(self):
        return self.budget - self.counter

    def get_cur_mask_2d(self):
        cur_mask = ~self.accumulated_mask.bool()
        cur_mask = cur_mask.squeeze()
        return cur_mask[:, self.sampled_indices[0]:self.sampled_indices[1] + 1]

    def factory_reset(self):
        self.data_loader_iter = iter(self.data_loader)

    def reach_budget(self):
        return self.counter >= self.budget

    def calculate_alpha(self, ssim_global, ssim_lesion):
        diff_mean = abs(ssim_global - ssim_lesion).mean()
        if diff_mean < 0.015:
            return 0.5  
        elif ssim_global.mean() > ssim_lesion.mean():
            return min(0.1 + 0.9 * (self.counter / self.budget), 1.0)  
        else:
            return max(0.1, 0.9 - 0.9 * (self.counter / self.budget))  

    def step(self, action, training=False):
        info = {}

        # Modify action to fit the range of sampled indices
        action = action + self.sampled_indices[0]
        action = torch.Tensor(action)
        action = torch.nn.functional.one_hot(action.long(), self.num_cols).unsqueeze(1).unsqueeze(1)

        # Update accumulated mask
        self.accumulated_mask = torch.max(self.accumulated_mask, action)
        self.counter += 1

        # Get observation and reward
        observation = self.state['sc_kspace'] * self.accumulated_mask
        recons = fastmri.ifft2c(torch.stack((observation.real, observation.imag), dim=-1))
        recons_abs = fastmri.complex_abs(recons)
        recons_norm = (recons_abs - recons_abs.amin(dim=(-1, -2), keepdim=True)) / (recons_abs.amax(dim=(-1, -2), keepdim=True) - recons_abs.amin(dim=(-1, -2), keepdim=True))
        ssim_global = py_msssim(self.state['target'], recons_norm, data_range=1.0, size_average=False)
        ssim_lesion, mse_lesion = calculate_lesion_ssim(recons, self.state['target'], self.state['annotation'])
        alpha, beta = calculate_dynamic_weight(ssim_global.mean().item(), ssim_lesion.mean().item(), self.counter)
        reward_global = ssim_global - self.previous_ssim_global
        reward_lesion = ssim_lesion - self.previous_ssim_lesion

        reward = 0.6 * (alpha * reward_global + beta * reward_lesion) - 0.4 * (mse_lesion - self.previous_mse_lesion)
        self.previous_ssim_lesion = ssim_lesion
        self.previous_mse_lesion = mse_lesion
        self.previous_ssim_global = ssim_global

        if training:
            observation = lesion_kspace_DC(self.state['target'], observation, self.state['annotation'])

        if self.reach_budget():
            done = torch.ones(1)  # Single done flag as budget is reached
            info['ssim_score'] = self.previous_ssim_lesion  # Optionally log SSIM score
            info['final_mask'] = self.accumulated_mask.clone().cpu().numpy()  # Optionally log mask
            observation = self.reset()
        else:
            done = torch.zeros(1)  # Single done flag as budget not yet reached

        return observation, reward, done, info

    def reset(self):
        # Reset data iterator if needed
        try:
            batch = next(self.data_loader_iter)
        except StopIteration:
            self.data_loader_iter = iter(self.data_loader)
            batch = next(self.data_loader_iter)

        # Move batch data to the designated device
        batch["sc_kspace"] = batch["sc_kspace"].to(self.device)
        batch["target"] = batch["target"].float().to(self.device).unsqueeze(1)
        if len(batch["sc_kspace"].shape) == 3:
            batch["sc_kspace"] = batch["sc_kspace"].unsqueeze(1)
        self.state = batch

        # Set up initial k-space and accumulated mask
        kspace = batch["sc_kspace"]
        batch_size = kspace.shape[0]
        num_cols = kspace.shape[-1]
        self.num_cols = num_cols
        mask = torch.zeros(batch_size, 1, 1, num_cols).to(self.device)
        mask[..., num_cols // 2 - 16:num_cols // 2 + 16] = 1
        self.accumulated_mask = mask
        self.counter = 0

        # Return the masked k-space as initial observation
        s0 = kspace * self.accumulated_mask
        self.done = torch.zeros(batch_size)

        initial_recons = fastmri.ifft2c(torch.stack((s0.real, s0.imag), dim=-1))
        initial_recons_abs = fastmri.complex_abs(initial_recons)
        initial_recons_norm = (initial_recons_abs - initial_recons_abs.amin(dim=(-1, -2), keepdim=True)) / (
                    initial_recons_abs.amax(dim=(-1, -2), keepdim=True) - initial_recons_abs.amin(dim=(-1, -2), keepdim=True))

        self.previous_ssim_global = py_msssim(self.state['target'], initial_recons_norm, data_range=1.0, size_average=False)
        self.previous_ssim_lesion, self.previous_mse_lesion = calculate_lesion_ssim(initial_recons, self.state['target'], self.state['annotation'])
        return s0


def calculate_dynamic_weight(ssim_global, ssim_lesion, step, total_steps=32):
    diff = abs(ssim_global - ssim_lesion)
    progress = step / total_steps 

    if diff < 0.01:
        alpha = 0.4 
        beta = 0.6
    else:
        alpha = max(0.1, 0.3 - 0.2 * progress)
        beta = 1.0 - alpha

    return alpha, beta

def lesion_kspace_DC(target, observation, annotation):
    lesion_kspace_priority = compute_lesion_priority_kspace(target, annotation)

    # Create a mask for the observed k-space (non-zero values in new_masked_kspace)
    observed_mask = observation != 0  # Shape: [batch, 1, 320, 320]

    # Combine new_masked_kspace and recons_kspace: retain values from new_masked_kspace where non-zero
    data_consistent_kspace = torch.where(observed_mask, observation, lesion_kspace_priority)

    return data_consistent_kspace

if __name__ == "__main__":
    pass



