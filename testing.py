import random
import time

import hydra
import numpy as np
import torch
from data_loading import MRDataModule
from actor_critic_modules.acdc_env import ACDC_Env
from omegaconf import OmegaConf
from collections import deque
from utils.utils import eval_mode, set_seed_everywhere
from hydra.core.hydra_config import HydraConfig
import os
import joblib
import logging
from pathlib import Path
from promptmr.promptmr import PromptMR
from segment.seg_test import load_5segmodels


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_data(config):
    return MRDataModule(config=config, dev_mode=config.dev_mode)

# Randomize seeds
def randomize_seed():
    seed = int(time.time()) % (2**32 - 1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_reward_model(cfg):
    reward_model = PromptMR(
        num_cascades=12,  # number of unrolled iterations
        num_adj_slices=5,  # number of adjacent slices

        n_feat0=48,  # number of top-level channels for PromptUnet
        feature_dim=[72, 96, 120],
        prompt_dim=[24, 48, 72],

        sens_n_feat0=24,
        sens_feature_dim=[36, 48, 60],
        sens_prompt_dim=[12, 24, 36],

        no_use_ca=False,
    )
    state_dict = torch.load(cfg.env.reward_model_ckpt)['state_dict']
    state_dict.pop('loss.w')
    state_dict = {k.replace('promptmr.', ''): v for k, v in state_dict.items()}
    reward_model.load_state_dict(state_dict)
    reward_model = reward_model.eval().to(cfg.device)
    return reward_model

@hydra.main(version_base=None, config_path='configs', config_name='eval')
def main(cfg):
    print(cfg)
    print(f"Current working directory : {os.getcwd()}")
    print(f"hydra path:{HydraConfig.get().run.dir}")
    run_dir = Path(HydraConfig.get().run.dir)
    data_module = get_data(cfg.env)

    if cfg.eval_data == 'val':
        val_loader = data_module.val_dataloader()
    else:
        val_loader = data_module.test_dataloader()

    print(f"-----length of eval_dataloader:{len(val_loader)}-----")

    set_seed_everywhere(cfg.seed)

    print(cfg.env)

    eval_envs = prepare_evaluate_envs(cfg, val_loader)

    
    ac = hydra.utils.instantiate(
        cfg.model, action_space=eval_envs.action_space)

    ac.to(cfg.device)

    global global_step
    global_step = 0

    load_snapshot(ac, cfg.load_from_snapshot_base_dir)
    eval_envs.reward_model = get_reward_model(cfg)

    eval_stats = {}
    start = cfg.eval_range[0]
    end = cfg.eval_range[1]
    # randomize_seed()
    for num_line in range(start, end+1):
        logging.info(
            f"=================Eval at budget of {num_line} line=================")
        eval_envs.set_budget(num_line)
        eval_stats[num_line] = evaluate(ac, eval_envs, num_line=num_line)

    # dump the evaluation stats
    ckpt_path = Path(cfg.load_from_snapshot_base_dir)
    if cfg.eval_data == 'val':
        joblib.dump(eval_stats, ckpt_path /
                    f"ppo_ts_val_stats_{start}_{end}.pkl")
    else:
        joblib.dump(eval_stats, ckpt_path /
                    f"ppo_ts_test_stats_{start}_{end}.pkl")


def load_snapshot(model, load_from_snapshot_base_dir):

    snapshot_base_dir = Path(load_from_snapshot_base_dir)
    snapshot = snapshot_base_dir / f'best_model.pt'
    if not snapshot.exists():
        logging.info(
            f"---WARNING---[Train.py] snapshot:{snapshot} not exists---WARNING---")
        return None
    logging.info(f"[eval_asmr.py] load snapshot:{snapshot}")
    model.load_state_dict(torch.load(snapshot))


def prepare_evaluate_envs(cfg, val_loader):

    observation_space = cfg.env.observation_space

    if cfg.env_version == "ACDC_Env":
        logging.info("===============using ACDC_Env in evaluation")

        envs = ACDC_Env(val_loader, observation_space=observation_space, device=cfg.device,
                        eval=True, fixed_budget=cfg.env.eval_fixed_budget,
                        scale_reward=cfg.env.scale_reward,
                        sampled_indices=cfg.env.sampled_indices,
                        reward_mode=cfg.env.reward_mode, srange=cfg.env.srange,
                        delay_step=cfg.env.delay_step,
                        )

    return envs


def evaluate(ac, envs, num_line):
    global global_step
    avg_ssim_scores_lesion = []
    avg_psnr_scores_lesion = []
    avg_ssim_scores = []
    avg_psnr_scores = []
    avg_dice_scores = []
    num_steps = len(envs.data_loader) * num_line
    envs.factory_reset()
    logging.debug(f"[evaluate] num_steps:{num_steps}, num_line:{num_line}, num_envs:{envs.num_envs}")
    obs = envs.reset()
    episode_reward = 0
    device = 'cuda'
    obs_mt = torch.tensor(envs.get_remain_epi_lines()).to(device)
    num_done = 0
    step = 0

    seg_model = load_5segmodels()

    while True:
        step += 1
        with torch.no_grad(), eval_mode(ac):
            cur_mask = envs.get_cur_mask_2d()
            input_dict = {"kspace": obs, 'mt': obs_mt}
            action, _, _, _ = ac.get_action_and_value(input_dict, cur_mask, deterministic=True)
            # action = choose_random_action(cur_mask).to(device)
            print(action)

        with torch.no_grad():
            obs, done, info = envs.step(action, seg_model)
            obs_mt = torch.tensor(envs.get_remain_epi_lines()).to(device)

        # reward is not used in testing phase
        if done.item() == 1:
            avg_ssim_scores.append(info.get('ssim_score', 0.0).item())
            avg_psnr_scores.append(info.get('psnr_score', 0.0).item())
            avg_dice_scores.append(info.get('dice_score', 0.0).item())


            if str(info.get('ssim_score_lesion', 0.0).item()) != 'nan':
                avg_ssim_scores_lesion.append(info.get('ssim_score_lesion', 0.0).item())
                avg_psnr_scores_lesion.append(info.get('psnr_score_lesion', 0.0).item())

            num_done += 1
            logging.debug(f"Final mask after {num_done} episodes: {info.get('final_mask')}")

            if num_done == len(envs.data_loader):
                break

    avg_ssim_score = np.mean(avg_ssim_scores)
    avg_psnr_score = np.mean(avg_psnr_scores)
    avg_dice_score = np.mean(avg_dice_scores)
    avg_ssim_score_lesion = np.mean(avg_ssim_scores_lesion)
    avg_psnr_score_lesion = np.mean(avg_psnr_scores_lesion)
    print(f'[EVAL] Avg SSIM: {avg_ssim_score}, Avg PSNR: {avg_psnr_score}, Avg dice_score: {avg_dice_score}, Avg SSIM_lesion: {avg_ssim_score_lesion}, Avg PSNR_lesion: {avg_psnr_score_lesion}')

    # Return relevant statistics
    eval_stats = {
        "num_lines": num_line,
        "avg_ssim": avg_ssim_score,
        "avg_psnr": avg_psnr_score,
        "avg_ssim_lesion": avg_ssim_score_lesion,
        "avg_psnr_lesion": avg_psnr_score_lesion,
    }

    return eval_stats

def choose_random_action(cur_mask):
    batch_size, num_cols = cur_mask.shape
    actions = []

    # Use a single generator with a fixed random seed for reproducibility or without seed for randomness
    generator = torch.Generator()
    generator.seed()

    for batch in range(batch_size):
        # Get indices of True elements
        valid_indices = torch.nonzero(cur_mask[batch], as_tuple=False).squeeze()
        if len(valid_indices) > 0:
            # Randomly select one index
            selected_action = valid_indices[torch.randint(0, len(valid_indices), (1,), generator=generator)].item()
        else:
            # Handle edge case where no True indices exist
            selected_action = -1  # Use -1 to indicate no valid action
        actions.append(selected_action)

    return torch.tensor(actions, dtype=torch.long)

if __name__ == "__main__":

    main()




