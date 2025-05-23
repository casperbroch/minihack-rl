import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from wrappers import EpisodeCSVWriter
from config   import LOGS_DIR

def _single_env(env_id: str, seed: int, log_subdir: str):
    def thunk():
        base = gym.make(
            env_id,
            observation_keys=("glyphs_crop", "blstats"),
        )
        base = TimeLimit(base, max_episode_steps=128)
        path = (LOGS_DIR / log_subdir / f"episode_seed{seed}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        env  = EpisodeCSVWriter(base, path=path)
        env.reset(seed=seed)
        return env
    return thunk

def make_vec_env(env_id: str, n_envs: int, seed: int, log_subdir: str = "PPO"):
    return VecMonitor(
        DummyVecEnv([_single_env(env_id, seed+i, log_subdir) for i in range(n_envs)])
    )

def make_eval_env(env_id: str, seed: int):
    """No logging, no time-limit shortening, extra pixel obs for video."""
    def thunk():
        env = gym.make(
            env_id,
            observation_keys=("glyphs_crop", "blstats", "pixel"),
            render_mode=None,
        )
        env = RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env
    return DummyVecEnv([thunk()])
