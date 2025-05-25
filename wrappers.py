# wrappers.py  : Custom Gym environment wrappers.
#
# Author       : Casper Bröcheler <casper.jxb@gmail.com>
# GitHub       : https://github.com/casperbroch
# Affiliation  : Maastricht University


import csv, time
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

class EpisodeCSVWriter(gym.Wrapper):
    # set up CSV writer, open file & record start time
    def __init__(self, env, path):
        super().__init__(RecordEpisodeStatistics(env))
        self.file  = open(path, "w", newline="")
        self.csv   = csv.DictWriter(self.file, fieldnames=["r", "l", "t"])
        self.csv.writeheader()
        self.t0 = time.time()

    # perform action; if episode ends, compute elapsed time & append stats to CSV
    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        if "episode" in info:     # end of episode
            info["episode"]["t"] = round(time.time() - self.t0, 6)
            self.csv.writerow(info["episode"])
            self.file.flush()
        return obs, r, term, trunc, info
    
    # close CSV file then cleanly close wrapped environment
    def close(self):
        self.file.close()
        super().close()
