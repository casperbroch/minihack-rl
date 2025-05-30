# envs.py     : Environment creation utilities with logging and wrappers.
#
# Author       : Casper Bröcheler <casper.jxb@gmail.com>
# GitHub       : https://github.com/casperbroch
# Affiliation  : Maastricht University


import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from wrappers import EpisodeCSVWriter
from config   import LOGS_DIR


def _single_env(env_id: str,
                seed: int,
                log_episodes: bool,
                log_subdir: str):
    
    # Create and seed a single Gym env, optionally wrapping for episode logging
    def thunk():
        base = gym.make(
            env_id,
            observation_keys=("glyphs_crop", "blstats"),
        )
        base = TimeLimit(base, max_episode_steps=128)

        if log_episodes:
            path = LOGS_DIR / log_subdir / f"episode_seed{seed}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            env = EpisodeCSVWriter(base, path=path)
        else:
            env = base

        env.reset(seed=seed)
        return env

    return thunk

# Build a vectorized set of n_envs parallel envs with monitoring
def make_vec_env(env_id: str,
                 n_envs: int,
                 seed: int,
                 *,
                 log_episodes: bool = True,
                 log_subdir: str = "PPO"):

    thunks = [
        _single_env(env_id, seed + i, log_episodes, log_subdir)
        for i in range(n_envs)
    ]
    return VecMonitor(DummyVecEnv(thunks))