# KSRO

PyTorch implementation of the paper **"[A reinforcement learning approach for optimized MRI sampling with region-specific fidelity](https://www.sciencedirect.com/science/article/pii/S092523122500788X)"**.

KSRO learns an adaptive **k-space sampling policy** for accelerated MRI. Instead of using a fixed undersampling mask, an agent trained with **Proximal Policy Optimization (PPO)** sequentially selects which k-space lines (phase-encoding lines) to acquire under a fixed budget. The reward balances **global image quality** against **region-specific (lesion) fidelity**, so the policy preferentially preserves diagnostically important regions.

## Method overview

- **Environment** (`rl/acdc_env.py`): At each step the agent observes the current undersampled k-space and the remaining acquisition budget, picks one of 256 phase-encoding lines to add to the mask, and receives a reward derived from the reconstruction.
- **Reward**: A dynamically weighted combination of global SSIM gain, lesion-region SSIM gain, and a lesion-region k-space MSE penalty. The weighting between global and lesion terms (`alpha`/`beta`) adapts over the course of an episode (see `calculate_dynamic_weight`).
- **Agent** (`rl/ppo_core.py`, `rl/ppo_core_net_mt.py`): A masked actor-critic (`KspaceMaskedActorCritic_MT`) built on FFT convolutions (`actor_critic_modules/fft_conv.py`) and a U-Net backbone (`actor_critic_modules/unet.py`). A masked categorical distribution prevents re-selecting already-acquired lines.
- **Budget**: Each episode acquires a fixed number of lines (default `budget: 32`).

## Repository structure

```
.
├── training.py                  # Entry point: PPO training + evaluation loop (Hydra-based)
├── configs/
│   └── train_acdc.yaml          # All hyperparameters, model and environment config
├── rl/
│   ├── acdc_env.py              # RL environment (state, step, reward, reset)
│   ├── ppo_core.py              # Masked actor-critic wrapper
│   ├── ppo_core_net_mt.py       # Network architecture (FFT conv + U-Net + attention)
│   ├── nn_utils.py              # MaskedCategorical and helpers
│   ├── calculate_lesion_ssim.py # Lesion-region SSIM / k-space MSE metrics
│   └── gen_sampler.py           # Builds a weighted sampler over the dataset
├── actor_critic_modules/
│   ├── fft_conv.py              # FFT-based convolution layer
│   ├── unet.py                  # U-Net model
│   └── transforms.py            # k-space / image transforms
├── data_loading/
│   ├── mr_dataset.py            # Dataset
│   ├── mr_datamodule.py         # DataModule (train/val loaders)
│   └── acdc_data.py             # ACDC-specific data handling
└── data_preprocessing/
    ├── 0-generate_csv_file.py   # Build CSV index + lesion bounding boxes from NIfTI
    ├── 1-save_singleslice.py    # Crop/pad volumes and save per-slice HDF5
    ├── 2-generate_acdc_metadata.py  # Generate metadata CSV
    ├── 3-scale_kspace_data.py   # Normalize/scale k-space and store in HDF5
    └── 4-verify_h5_scaled.py    # Sanity-check that scaling was applied
```

## Requirements

The code targets Python 3.9–3.11 with a CUDA-capable GPU. Key dependencies:

- `torch`
- `fastmri`
- `pytorch-msssim`
- `hydra-core`, `omegaconf`
- `wandb`
- `tensorboard`
- `numpy`, `scipy`, `pandas`, `h5py`
- `nibabel` (NIfTI I/O during preprocessing)
- `joblib`, `tqdm`

Install (adjust the Torch build for your CUDA version):

```bash
pip install torch fastmri pytorch-msssim hydra-core omegaconf wandb \
            tensorboard numpy scipy pandas h5py nibabel joblib tqdm
```

## Data preparation

The model is trained on the [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) cardiac MRI dataset. Raw NIfTI volumes are converted into scaled, per-slice k-space stored in HDF5. Run the preprocessing scripts in order (edit the hard-coded input/output paths inside each script first):

```bash
python data_preprocessing/0-generate_csv_file.py      # index + lesion bounding boxes
python data_preprocessing/1-save_singleslice.py       # crop/pad to 256x256, save HDF5
python data_preprocessing/2-generate_acdc_metadata.py # metadata CSV (train/val/test)
python data_preprocessing/3-scale_kspace_data.py       # scale k-space -> sc_kspace_scaled
python data_preprocessing/4-verify_h5_scaled.py        # verify scaling
```

This produces the files referenced by the environment config:

- `datadir` — directory of generated HDF5 files
- `split_csv_file` — `metadata_acdc.csv`
- `train_sampler_filename` — weighted-sampler metadata (`.p`)

## Configuration

All settings live in `configs/train_acdc.yaml` and are managed by [Hydra](https://hydra.cc/). Update the paths (currently masked as `/*******`) to point at your preprocessed data and desired output directory:

```yaml
snapshot_dir: /path/to/output
load_from_snapshot_base_dir: None   # set to a run dir to resume from best_model.pt

env:
  datadir: /path/to/generated_files
  split_csv_file: /path/to/metadata_acdc.csv
  train_sampler_filename: /path/to/meta_data.p
```

Notable hyperparameters: `budget: 32` (lines per episode), `num_envs: 50`, `num_steps: 32`, `total_timesteps: 100000000`, PPO `gamma: 0.99`, `gae_lambda: 0.9`, `clip_coef: 0.1`, optimizer AdamW with `lr: 0.0004`.

## Training

```bash
python training.py
```

Hydra creates a timestamped run directory (`hydra.run.dir`) containing model snapshots (`models/best_model.pt`, `models/last_model.pt`) and TensorBoard logs (`tb/`). Training metrics are also synced to Weights & Biases under the `ACDC` project.

Override any config value from the command line, e.g.:

```bash
python training.py budget=16 num_envs=32 optim.lr=0.0002
```

Monitor progress:

```bash
tensorboard --logdir <run_dir>/tb
```

## Evaluation

Evaluation runs automatically every `eval_interval` updates during training. The policy acts **deterministically** (greedy line selection) over the validation set, and the best model is checkpointed by average lesion SSIM. Key logged metrics include `charts/eval_mean_ssim`, `charts/eval_mean_return`, and `charts/best_ssim`.

## Acknowledgments

- [asmr](https://github.com/robinyen/asmr)
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)

## Citation

If you use this code, please cite the paper:

```bibtex
@article{ksro,
  title   = {A reinforcement learning approach for optimized MRI sampling with region-specific fidelity},
  journal = {Computer Methods and Programs in Biomedicine},
  url     = {https://www.sciencedirect.com/science/article/pii/S092523122500788X}
}
```
