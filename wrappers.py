import csv, time
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

class EpisodeCSVWriter(gym.Wrapper):
    """
    Write every finished episode to <logs_dir>/episode_seedX.csv with:
        r (return), l (length), t (wall-clock)
    """
    def __init__(self, env, path):
        super().__init__(RecordEpisodeStatistics(env))
        self.file  = open(path, "w", newline="")
        self.csv   = csv.DictWriter(self.file, fieldnames=["r", "l", "t"])
        self.csv.writeheader()
        self.t0 = time.time()

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        if "episode" in info:     # end of episode
            info["episode"]["t"] = round(time.time() - self.t0, 6)
            self.csv.writerow(info["episode"])
            self.file.flush()
        return obs, r, term, trunc, info

    def close(self):
        self.file.close()
        super().close()
