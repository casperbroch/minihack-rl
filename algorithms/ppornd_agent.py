# ppornd_agent.py  : PPO with RND intrinsic rewards for exploration.
#
# Author       : Casper Bröcheler <casper.jxb@gmail.com>
# GitHub       : https://github.com/casperbroch
# Affiliation  : Maastricht University


import torch as th, torch.nn as nn
from torch.optim import Adam
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.running_mean_std import RunningMeanStd
from .base        import BaseAgent
from features     import MiniHackCNN
from utils        import linear_schedule
import config

# Predictor–target network for RND intrinsic reward
class _RNDNet(nn.Module):
     # Freeze encoder and target, init predictor optimizer
    def __init__(self, obs_space, feat_dim=256, emb_dim=128, device="cpu"):
        super().__init__()
        self.enc = MiniHackCNN(obs_space, feat_dim)
        for p in self.enc.parameters(): p.requires_grad_(False)
        self.tgt = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(),
                                 nn.Linear(256, emb_dim))
        for p in self.tgt.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
            p.requires_grad_(False)
        self.pred = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(),
                                  nn.Linear(256, emb_dim))
        self.opt  = Adam(self.pred.parameters(), lr=1e-4)
        self.to(device)
        self.device = device

    # Compute predicted and target embeddings
    def forward(self, obs):
        with th.no_grad():
            f = self.enc(obs)
            t = self.tgt(f)
        p = self.pred(f)
        return p, t

# Wrap VecEnv to add RND-based intrinsic rewards
class _RNDWrap(VecEnvWrapper):
    def __init__(self, venv, coef=0.25, feat_dim=256, device="cpu"):
        super().__init__(venv)
        self.rnd = _RNDNet(venv.observation_space, feat_dim, device=device)
        self.rms = RunningMeanStd(shape=())
        self.coef = coef

    # Step env, compute and normalize intrinsic reward, update RND predictor
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        obs_tensor = {k: th.as_tensor(v, device=self.rnd.device) for k, v in obs.items()}
        pred, tgt = self.rnd(obs_tensor)
        int_reward = (pred - tgt).pow(2).mean(1).detach().cpu().numpy()
        self.rms.update(int_reward)
        int_reward /= (self.rms.var ** 0.5 + 1e-8)
        loss = (pred - tgt).pow(2).mean()
        self.rnd.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.rnd.opt.step()
        return obs, rewards + self.coef * int_reward, dones, infos
    
    # Reset underlying env
    def reset(self):
        return self.venv.reset()
    
class PPORNDAgent(BaseAgent):
    name     = "PPO_RND"
    algo_cls = PPO

     # Build PPO model wrapped with RND environment
    def build_model(self):
        feat_dim = self.kwargs.get("features_dim", 256)
        net_arch = self.kwargs.get("net_arch")
        int_coef = self.kwargs.get("intrinsic_coef", 0.25)

        # Wrap env to inject intrinsic rewards
        self.env = _RNDWrap(self.env, coef=int_coef,
                            feat_dim=feat_dim,
                            device=self.kwargs.get("device",
                                                   config.DEFAULT_DEVICE))

        pk = dict(features_extractor_class  = MiniHackCNN,
                  features_extractor_kwargs = dict(features_dim=feat_dim))
        if net_arch is not None: pk["net_arch"] = net_arch

        return self.algo_cls(
            policy="MultiInputPolicy",
            env=self.env,
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
            policy_kwargs = pk,
            device        = self.kwargs.get("device", config.DEFAULT_DEVICE),
            verbose       = 1,
        )
    
    # Define hyperparameter search space for optimizer Optuna
    @staticmethod
    def sample_hyperparams(trial):
        # enumerate all (n_steps, batch_size) pairs that divide exactly
        base_steps, batches = [128, 256, 512], [128, 256, 512]
        legal = [(s, b) for s in base_steps for b in batches if (s*6)%b==0]

        # sample the index of the pair
        idx = trial.suggest_int("sb_idx", 0, len(legal)-1)
        n_steps, batch_size = legal[idx]

        feat_dim = trial.suggest_categorical("features_dim", [128, 256])
        arch_key = trial.suggest_categorical("net_arch", ["64x64", "128x128"])
        arch_map = {"64x64":[64,64], "128x128":[128,128]}
        lr       = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        n_epochs = trial.suggest_int("n_epochs", 3, 15)
        gamma    = trial.suggest_float("gamma", 0.90, 0.9999)
        gae_lam  = trial.suggest_float("gae_lambda", 0.8, 0.98)
        clip_r   = trial.suggest_float("clip_range", 0.1, 0.3)
        ent_c    = trial.suggest_float("ent_coef", 1e-5, 1e-2, log=True)
        vf_c     = trial.suggest_float("vf_coef", 0.1, 1.0)
        max_g    = trial.suggest_float("max_grad_norm", 0.5, 5.0)
        tkl      = trial.suggest_float("target_kl", 0.01, 0.2)
        int_c    = trial.suggest_float("intrinsic_coef", 0.05, 0.5)

        return {
            "learning_rate":  linear_schedule(lr),
            "batch_size":     batch_size,
            "n_steps":        n_steps,
            "n_epochs":       n_epochs,
            "gamma":          gamma,
            "gae_lambda":     gae_lam,
            "clip_range":     clip_r,
            "ent_coef":       ent_c,
            "vf_coef":        vf_c,
            "max_grad_norm":  max_g,
            "target_kl":      tkl,
            "features_dim":   feat_dim,
            "net_arch":       arch_map[arch_key],
            "intrinsic_coef": int_c,
        }
