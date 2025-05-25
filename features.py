# features.py  : CNN-based feature extractor for MiniHack observations.
#
# Author       : Casper Bröcheler <casper.jxb@gmail.com>
# GitHub       : https://github.com/casperbroch
# Affiliation  : Maastricht University


import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MiniHackCNN(BaseFeaturesExtractor):
    # Extract and embed glyphs, then combine with blstats via conv + MLP
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        glyphs = observation_space.spaces["glyphs_crop"]
        n_glyphs = int(glyphs.high.max()) + 1
        H, W     = glyphs.shape

        self.embed = nn.Embedding(n_glyphs, 32)
        self.conv  = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Flatten()
        )
        self._conv_out = 64 * H * W
        bl_dim = observation_space.spaces["blstats"].shape[0]

        self.fc = nn.Sequential(
            nn.Linear(self._conv_out + bl_dim, features_dim),
            nn.ReLU(),
        )

    # Embed glyphs, apply conv, concatenate blstats, then MLP to features
    def forward(self, obs):
        x = self.embed(obs["glyphs_crop"].long()).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = th.cat([x, obs["blstats"].float()], dim=1)
        return self.fc(x)
