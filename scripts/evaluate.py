from pathlib import Path
import imageio.v3 as iio
from stable_baselines3 import PPO

from features import MiniHackCNN
from envs     import make_eval_env
from config   import VIDEOS_DIR

def record_video(model_path: str,
                 env_id: str,
                 seed: int = 0,
                 n_episodes: int = 5,
                 fps: int = 1):
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    env   = make_eval_env(env_id, seed)
    model = PPO.load(
        model_path,
        device="cuda",
        custom_objects={"minihack_rl.features.MiniHackCNN": MiniHackCNN},
    )

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed+ep)
        frames = []
        done   = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
            frames.append(env.envs[0].get_last_render())  # pixel obs

        fname = VIDEOS_DIR / f"run_{ep}.mp4"
        iio.imwrite(fname, frames, fps=fps)
        print("🎞️  saved", fname)
