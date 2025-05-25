# RL Agents for MiniHack
By [Casper Bröcheler](https://github.com/casperbroch)

This repository provides a modular and extensible reinforcement learning (RL) framework designed for experiments in MiniHack, a suite of procedurally generated environments for challenging agent learning tasks. It enables easy training, evaluation, and hyperparameter optimization of various deep RL algorithms using the [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) library and its extensions.

## What You Can Do
- Train and evaluate DRL agents using PPO, Recurrent PPO, PPO + RND, and QR-DQN.
- Visualize and compare performance using custom plotting tools.
- Extract and process MiniHack observations using a custom CNN-based feature extractor.
- Tune hyperparameters automatically with [Optuna](https://optuna.org/).
- Run training manually via command line with custom or pre-specified hyperparameters.
- Log and save training metrics, models, and performance plots automatically.

## Algorithms Included
| Algorithm      | Class               | Description                                                      |
| -------------- | ------------------- | ---------------------------------------------------------------- |
| `PPO`          | `PPOAgent`          | Standard on-policy learning with Proximal Policy Optimization    |
| `RecurrentPPO` | `RecurrentPPOAgent` | Uses LSTM-based policies to handle partial observability         |
| `PPO + RND`    | `PPORNDAgent`       | Enhances exploration using Random Network Distillation (RND)     |
| `QR-DQN`       | `QRDQNAgent`        | Distributional off-policy learning using Quantile Regression DQN |

## Custom Feature Extractor
All agents leverage a shared CNN-based architecture for processing MiniHack observations:

`MiniHackCNN`:
- Extracts glyph-level visual features (`glyphs_crop`) and combines them with numeric state features (`blstats`).
- Used across PPO, RND, and QR-DQN models.
- Flexible with custom feature dimensions and network architectures.

## Core Files

## Plotting Results


```
python -m scripts.hyperopt --algo PPO --env MiniHack-Room-Ultimate-15x15-v0 --trials 10 --steps 50000 --seed 42 --n-envs 8
```
