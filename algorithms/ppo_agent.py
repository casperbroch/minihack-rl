from stable_baselines3 import PPO

from .base        import BaseAgent
from features   import MiniHackCNN
from utils      import linear_schedule
import config

class PPOAgent(BaseAgent):
    name     = "PPO"
    algo_cls = PPO

    #  Build the (untrained) SB3 model
    def build_model(self):
        # network size
        features_dim = self.kwargs.get("features_dim", 256)
        net_arch     = self.kwargs.get("net_arch")        # may be None

        policy_kwargs = dict(
            features_extractor_class  = MiniHackCNN,
            features_extractor_kwargs = dict(features_dim=features_dim),
        )
        if net_arch is not None:
            policy_kwargs["net_arch"] = net_arch

        return self.algo_cls(
            policy="MultiInputPolicy",
            env=self.env,
            # --------- all tune-able hyper-parameters ------------------
            learning_rate = self.kwargs.get("learning_rate",
                                            linear_schedule(3e-4)),
            batch_size    = self.kwargs.get("batch_size",    2048),
            n_steps       = self.kwargs.get("n_steps",       512),
            n_epochs      = self.kwargs.get("n_epochs",      10),
            gamma         = self.kwargs.get("gamma",         0.995),
            gae_lambda    = self.kwargs.get("gae_lambda",    0.95),
            clip_range    = self.kwargs.get("clip_range",    0.2),
            ent_coef      = self.kwargs.get("ent_coef",      1e-3),
            vf_coef       = self.kwargs.get("vf_coef",       0.5),
            max_grad_norm = self.kwargs.get("max_grad_norm", 1.0),
            target_kl     = self.kwargs.get("target_kl",     0.03),
            # -----------------------------------------------------------
            policy_kwargs = policy_kwargs,
            device        = self.kwargs.get("device", config.DEFAULT_DEVICE),
            verbose       = 1,
        )

    @staticmethod
    def sample_hyperparams(trial):
        # 1) enumerate all (n_steps, batch_size) pairs that divide exactly
        base_steps      = [128, 256, 512]
        possible_batches= [128, 256, 512]
        legal_pairs = [
            (s, b) for s in base_steps for b in possible_batches if (s * 6) % b == 0
        ]

        # 2) sample the index of the pair
        idx = trial.suggest_int("step_batch_idx", 0, len(legal_pairs) - 1)
        n_steps, batch_size = legal_pairs[idx]

        # 3) Drop the heaviest CNN feature dims and deep nets
        features_dim = trial.suggest_categorical("features_dim", [128, 256])
        net_arch_key = trial.suggest_categorical("net_arch", ["64x64", "128x128"])
        net_arch_map = {
            "64x64":  [64, 64],
            "128x128":[128, 128],
        }
        net_arch = net_arch_map[net_arch_key]

        # 4) Keep the rest of the hyperparameters as before
        lr         = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        n_epochs   = trial.suggest_int("n_epochs", 3, 15)
        gamma      = trial.suggest_float("gamma", 0.90, 0.9999)
        gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.98)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
        ent_coef   = trial.suggest_float("ent_coef", 1e-5, 1e-2, log=True)
        vf_coef    = trial.suggest_float("vf_coef", 0.1, 1.0)
        max_grad   = trial.suggest_float("max_grad_norm", 0.5, 5.0)
        target_kl  = trial.suggest_float("target_kl", 0.01, 0.2)

        return {
            "learning_rate":  linear_schedule(lr),
            "batch_size":     batch_size,
            "n_steps":        n_steps,
            "n_epochs":       n_epochs,
            "gamma":          gamma,
            "gae_lambda":     gae_lambda,
            "clip_range":     clip_range,
            "ent_coef":       ent_coef,
            "vf_coef":        vf_coef,
            "max_grad_norm":  max_grad,
            "target_kl":      target_kl,
            "features_dim":   features_dim,
            "net_arch":       net_arch,
        }