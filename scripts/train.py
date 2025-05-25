# train.py    : CLI-driven training script with manual hyperparameters.
#
# Author       : Casper Bröcheler <casper.jxb@gmail.com>
# GitHub       : https://github.com/casperbroch
# Affiliation  : Maastricht University


from pathlib import Path
import argparse

from stable_baselines3.common.evaluation import evaluate_policy

from algorithms import get_agent_class
from config import MODELS_DIR

import minihack
import gymnasium as gym

# Default configuration (overridable via CLI)
DEFAULT_ALGO = "ppo"                        # e.g. "ppo", "dqn", "sac"
DEFAULT_ENV_ID = "MiniHack-Room-5x5-v0"
DEFAULT_TOTAL_STEPS = 100_000
DEFAULT_SEED = 0 
DEFAULT_N_ENVS = 12

# Manual hyperparameters per algorithm

# Found after 12 750k steps trials
HYPERPARAMS_PPO = {
    "learning_rate": 0.0008663,
    "n_steps": 128,
    "batch_size": 256,
    "n_epochs": 3,
    "gamma": 0.9873,
    "gae_lambda": 0.8058,
    "clip_range": 0.29815,
    "ent_coef": 0.00122655,
    "vf_coef": 0.9677,
    "max_grad_norm": 0.9249,
    "target_kl": 0.1699,
    "features_dim": 128,
    "net_arch": [128, 128],
}

# Found after 12 750k steps trials (copy from PPO, due to lack of time)
HYPERPARAMS_RECURRENTPPO = {
    "learning_rate": 0.0008663,
    "n_steps": 128,
    "batch_size": 256,
    "n_epochs": 3,
    "gamma": 0.9873,
    "gae_lambda": 0.8058,
    "clip_range": 0.29815,
    "ent_coef": 0.00122655,
    "vf_coef": 0.9677,
    "max_grad_norm": 0.9249,
    "target_kl": 0.1699,
    "features_dim": 128,
    "net_arch": [128, 128],
}

# Found after 12 750k steps trials
HYPERPARAMS_PPO_RND = {
    "learning_rate": 0.000347,
    "n_steps": 256,
    "batch_size": 256,
    "n_epochs": 7,
    "gamma": 0.9728,
    "gae_lambda": 0.86236,
    "clip_range": 0.27189,
    "ent_coef": 0.0010915,
    "vf_coef": 0.24747,
    "max_grad_norm": 4.1535,
    "target_kl": 0.1595,
    "features_dim": 128,
    "net_arch": [128, 128],
    "intrinsic_coef": 0.173604,
}

# Found after 12 500k steps trials
HYPERPARAMS_QRDQN = {
    "learning_rate": 0.00012677,
    "buffer_size": 200_000,
    "learning_starts": 8000,
    "batch_size": 256,
    "gamma": 0.94387,
    "train_freq": 1,
    "gradient_steps": 1,
    "target_update_interval": 4000,
    "tau": 0.88873,
    "exploration_fraction": 0.16524,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.015412,
    "max_grad_norm": 1.05683,
    "features_dim": 128,
    "net_arch": [64, 64],
}

HYPERPARAMS_MAP = {
    "ppo": HYPERPARAMS_PPO,
    "rppo": HYPERPARAMS_RECURRENTPPO,
    "ppornd": HYPERPARAMS_PPO_RND,
    "qrdqn": HYPERPARAMS_QRDQN,
}

# Parse command-line arguments for training settings
def parse_args():
    parser = argparse.ArgumentParser(description="Train an RL agent with manual hyperparameters")
    parser.add_argument("--algo", default=DEFAULT_ALGO, help="Algorithm key (e.g. ppo, dqn, sac)")
    parser.add_argument("--env", dest="env_id", default=DEFAULT_ENV_ID, help="Gym/MiniHack environment id")
    parser.add_argument("--steps", dest="total_steps", type=int, default=DEFAULT_TOTAL_STEPS,
                        help="Total training steps")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--n-envs", type=int, dest="n_envs", default=DEFAULT_N_ENVS,
                        help="Number of parallel environments")
    return parser.parse_args()

# Instantiate agent, train, evaluate, and save the model
def main():
    args = parse_args()
    # Use CLI arguments
    algo = args.algo
    env_id = args.env_id
    total_steps = args.total_steps
    seed = args.seed
    n_envs = args.n_envs

    # Instantiate agent class
    AgentCls = get_agent_class(algo)

    algo_key = algo.lower()
    hyperparams = HYPERPARAMS_MAP.get(algo_key)
    if hyperparams is None:
        raise ValueError(f"No hyperparameters defined for algorithm: {algo_key}")
    
    print(f"Starting training for {algo.upper()} on {env_id}")
    print(f"Settings: steps={total_steps}, seed={seed}, n_envs={n_envs}")
    print(f"Hyperparameters: {hyperparams}")
    agent = AgentCls(
        env_id,
        total_steps=total_steps,
        seed=seed,
        n_envs=n_envs,
        log_episodes=True,
        **hyperparams,
    )
    agent.train()

    # Evaluate the trained policy
    print("\nEvaluating the trained policy over 50 episodes...")
    mean_reward, std_reward = evaluate_policy(
        agent.model,
        agent.env,
        n_eval_episodes=50,
        deterministic=True,
    )
    print(f"Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")

    # Save the model
    save_dir = Path(MODELS_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{algo}_{env_id}_model.zip"
    agent.save(str(save_path))
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()