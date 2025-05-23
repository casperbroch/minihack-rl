from stable_baselines3 import PPO

from .base        import BaseAgent
from features   import MiniHackCNN
from utils      import linear_schedule
import config

class PPOAgent(BaseAgent):
    name      = "PPO"
    algo_cls  = PPO

    def build_model(self):
        policy_kwargs = dict(
            features_extractor_class  = MiniHackCNN,
            features_extractor_kwargs = dict(features_dim=256),
        )
        return self.algo_cls(
            "MultiInputPolicy",
            self.env,
            learning_rate = linear_schedule(3e-4),
            n_steps       = 512,
            batch_size    = 2048,
            n_epochs      = 10,
            gamma         = 0.995,
            gae_lambda    = 0.95,
            clip_range    = linear_schedule(0.2),
            ent_coef      = 1e-3,
            vf_coef       = 0.5,
            target_kl     = 0.03,
            max_grad_norm = 1.0,
            policy_kwargs = policy_kwargs,
            device        = self.kwargs.get("device", config.DEFAULT_DEVICE),
            verbose       = 1,
        )
