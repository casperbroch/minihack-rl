from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from stable_baselines3.common.callbacks import CheckpointCallback

import envs, utils, config


class BaseAgent(ABC):
    name: str
    algo_cls = None

    # constructor & core workflow
    def __init__(
        self,
        env_id: str,
        total_steps: int = 1_000_000,
        seed: int = 0,
        n_envs: int = 24,
        log_episodes: bool = True,
        **kwargs,
    ):
        utils.set_global_seeds(seed)

        self.env_id, self.total_steps = env_id, total_steps
        self.seed, self.n_envs = seed, n_envs
        self.kwargs = kwargs

        self.env = envs.make_vec_env(
            env_id,
            n_envs,
            seed,
            log_episodes=log_episodes,
            log_subdir=self.name,
        )        
        self.model = self.build_model()

    # Methods subclasses MUST override
    @abstractmethod
    def build_model(self):
        """Return an *untrained* SB3 model (or any trainer with .learn())."""

    # Canned helpers
    def train(self):
        ckpt = CheckpointCallback(
            save_freq=self.kwargs.get("save_freq", 10_000),
            save_path=(config.MODELS_DIR / self.name).as_posix(),
        )
        self.model.learn(
            self.total_steps,
            log_interval=self.kwargs.get("log_interval", 2),
            callback=ckpt,
        )

    def save(self, fname: Optional[str] = None) -> Path:      # ← fixed
        fname = fname or f"{self.name.lower()}_{self.env_id}.zip"
        path = config.MODELS_DIR / self.name / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path.as_posix())
        return path

    # convenience for notebooks
    def train_and_save(self) -> Path:
        self.train()
        return self.save()

    #  HYPER-PARAMETER OPTIMISATION HOOK, to be overwritten!
    @staticmethod
    def sample_hyperparams(trial):
        return {}
