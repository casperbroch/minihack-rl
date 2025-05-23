from sb3_contrib import QRDQN
from .base      import BaseAgent
from features   import MiniHackCNN
from utils      import linear_schedule
import config


class QRDQNAgent(BaseAgent):
    name     = "QRDQN"
    algo_cls = QRDQN

    def build_model(self):
        features_dim = self.kwargs.get("features_dim", 256)
        net_arch     = self.kwargs.get("net_arch")  # may be None

        policy_kwargs = dict(
            features_extractor_class  = MiniHackCNN,
            features_extractor_kwargs = dict(features_dim=features_dim),
        )
        if net_arch is not None:
            policy_kwargs["net_arch"] = net_arch

        return self.algo_cls(
            policy="MultiInputPolicy",
            env=self.env,
            # --- Core DQN/QR-DQN hyper-parameters ---
            learning_rate     = self.kwargs.get("learning_rate",
                                                linear_schedule(2.5e-4)),
            buffer_size       = self.kwargs.get("buffer_size",       200_000),
            learning_starts   = self.kwargs.get("learning_starts",   8_000),
            batch_size        = self.kwargs.get("batch_size",        512),
            gamma             = self.kwargs.get("gamma",             0.99),
            train_freq        = self.kwargs.get("train_freq",        (4, "step")),
            gradient_steps    = self.kwargs.get("gradient_steps",    1),
            target_update_interval = self.kwargs.get("target_update_interval", 8_000),
            tau               = self.kwargs.get("tau",               1.0),
            exploration_fraction   = self.kwargs.get("exploration_fraction",   0.12),
            exploration_initial_eps = self.kwargs.get("exploration_initial_eps", 1.0),
            exploration_final_eps   = self.kwargs.get("exploration_final_eps",   0.05),
            max_grad_norm     = self.kwargs.get("max_grad_norm",     10.0),
            policy_kwargs     = policy_kwargs,
            device            = self.kwargs.get("device", config.DEFAULT_DEVICE),
            verbose           = 1,
        )

    @staticmethod
    def sample_hyperparams(trial):
        features_dim = trial.suggest_categorical("features_dim", [128, 256, 512])
        net_arch_key = trial.suggest_categorical("net_arch", ["64x64", "128x128", "256x256"])
        net_arch = {"64x64": [64, 64], "128x128": [128, 128], "256x256": [256, 256]}[net_arch_key]

        lr          = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        buffer_size = trial.suggest_categorical("buffer_size", [100_000, 200_000, 400_000])
        batch_size  = trial.suggest_categorical("batch_size", [256, 512, 1024])
        train_freq  = trial.suggest_categorical("train_freq", [1, 4, 8])
        target_int  = trial.suggest_categorical("target_update_interval", [4_000, 8_000, 16_000])
        gamma       = trial.suggest_float("gamma", 0.90, 0.9999)
        tau         = trial.suggest_float("tau", 0.8, 1.0)
        max_grad    = trial.suggest_float("max_grad_norm", 5.0, 15.0)
        expl_frac   = trial.suggest_float("exploration_fraction", 0.05, 0.25)
        expl_final  = trial.suggest_float("exploration_final_eps", 0.01, 0.1)

        return {
            "learning_rate":  linear_schedule(lr),
            "buffer_size":    buffer_size,
            "batch_size":     batch_size,
            "train_freq":     (train_freq, "step"),
            "target_update_interval": target_int,
            "gamma":          gamma,
            "tau":            tau,
            "max_grad_norm":  max_grad,
            "exploration_fraction": expl_frac,
            "exploration_final_eps": expl_final,
            "features_dim":   features_dim,
            "net_arch":       net_arch,
        }
