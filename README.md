# RL Agents for MiniHack
By [Casper Bröcheler](https://github.com/casperbroch)

This repository provides a modular and extensible reinforcement learning (RL) framework designed for experiments in [MiniHack](https://github.com/samvelyan/minihack), a suite of procedurally generated environments for challenging agent learning tasks. It enables easy training, evaluation, and hyperparameter optimization of various deep RL algorithms using the [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) library and its extensions.

<p align="center">
  <img src="data/videos/README/ezgif-8e5c0ca1cc0ac7.gif" alt="Demo"/>
</p>

## What You Can Do
- Train and evaluate DRL agents using PPO, Recurrent PPO, PPO + RND, and QR-DQN.
- Visualize and compare performance using custom plotting tools.
- Extract and process [MiniHack](https://github.com/samvelyan/minihack) observations using a custom CNN-based feature extractor.
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

## Core Script Files
Both scripts have the following CLI options:
- `--algo`: Algorithm key (ppo, RecurrentPPO, PPO_RND, QRDQN)
- `--env`: Environment ID from MiniHack
- `--steps`: Training steps
- `--seed`: Random seed
- `--n-envs`: Number of parallel environments

### `hyperopt.py` - Hyperparameter Optimization

Runs a full Optuna-based hyperparameter search followed by a final training run with the best config.

Example usage:
```console
python -m scripts.hyperopt --algo PPO --env MiniHack-Room-Ultimate-15x15-v0 --trials 10 --steps 1000000 --seed 42 --n-envs 8
```
What it does:
- Samples and trains `n_trials` agents using Optuna.
- Evaluates each using `evaluate_policy`.
- Retrains the best model using 4× more steps.
- Evaluates and saves the final model.

Outputs:

- Optuna logs
- Best trial config
- Final model in `data/models/`
- Evaluation metrics

### `train.py` - Manual Training Script

Trains an RL agent using manually defined or CLI-specified hyperparameters without doing hyperparameter optimization.

Example usage:
```console
python -m scripts.train --algo PPO --env MiniHack-Room-Ultimate-15x15-v0 --steps 1000000 --seed 42 --n-envs 8
```
What it does:
- Trains a model using given amount of steps.
- Saves the final model.

Outputs:
- Final model in `data/models/`
- Evaluation metrics


## Evaluation & Visualization
This codebase provides two powerful tools for analyzing and comparing agent performance:

### `run_plots.py` - Automated Metric Visualization
This script loads training logs (CSV files), aggregates them per algorithm, and generates:

- Smoothed Return over Episodes
- Smoothed Return over Time
- Episode Length Trends
- Final Return Distribution (Boxplot)

Output:
Plots are automatically saved to plots/<timestamp>/ and corresponding models are copied for reproducibility.

To run:
```console
python run_plots.py
```
Make sure to configure the `CONFIG['log_dirs']` list in `run_plots.py` to include the paths of the algorithms you want to compare.

### `evaluation.ipynb` – Interactive Evaluation Notebook

This Jupyter notebook offers an interactive alternative for:
- Evaluating trained policies using heatmaps and videos
- Inspecting CSV logs, models, and statistics in a step-by-step workflow

It’s especially useful during development or experimentation when you want:
- Quick insights without re-running scripts
- To compare individual model behaviors
- To experiment with new metrics or visualizations

To use:
```console
jupyter notebook evaluation.ipynb
```

## Requirements
- Linux (or WSL 2 Ubuntu when on Windows)
- Python ≥ 3.8 (3.9.22 is recommended)
- [MiniHack](https://github.com/samvelyan/minihack)
- `stable-baselines3`, `sb3-contrib`
- `optuna`, `pandas`, `matplotlib`, `gymnasium`

You can install all required packages with:

```console
pip install -r requirements.txt
```


## Author

**Casper Bröcheler**  
📧 [casper.jxb@gmail.com](mailto:casper.jxb@gmail.com) · [c.brocheler@student.maastrichtuniversity.nl](mailto:c.brocheler@student.maastrichtuniversity.nl)  
🎓 Maastricht University (Msc. Artificial Intelligence)     
🔗 [github.com/casperbroch](https://github.com/casperbroch)