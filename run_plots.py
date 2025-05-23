# plot_all_logs.py
# merge every *csv episode log under ROOT_DIR, sort by wall-clock t,
# make standard RL plots (smoothed return, smoothed ep-length, final boxplot)

import os, glob, re, warnings
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns

ROOT_DIR     = "./data/logs"   # env folders or flat csv files live here
SMOOTH_WIN   = 20              # moving-avg window (episodes)
TAIL_COUNT   = 20             # last N episodes per run for boxplot
OUTDIR       = "./plots"
os.makedirs(OUTDIR, exist_ok=True)

def read_csv(path):
    df  = pd.read_csv(path)
    fn  = os.path.basename(path)
    env = alg = None

    # expected pattern <env>_<alg>_seed##.csv but fallbacks allowed
    m = re.match(r"([^_]+)_([^_]+)_seed(\d+)\.csv", fn)
    if m:
        env, alg, seed = m.group(1), m.group(2), int(m.group(3))
    else:
        env = os.path.basename(os.path.dirname(path))
        m2 = re.search(r"([A-Za-z0-9\-]+?)_", fn)
        alg = m2.group(1) if m2 else env
        m3 = re.search(r"\d+", fn)
        seed = int(m3.group()) if m3 else -1
        if seed == -1:
            warnings.warn(f"no seed in {fn}")
    df["env_id"], df["algorithm"], df["seed"] = env, alg, seed
    return df

files = glob.glob(os.path.join(ROOT_DIR, "**", "*.csv"), recursive=True)
if not files:
    raise FileNotFoundError(ROOT_DIR)

data = pd.concat([read_csv(f) for f in files], ignore_index=True)
data = data.sort_values("t").reset_index(drop=True)
data.to_csv(os.path.join(OUTDIR, "merged_all_envs.csv"), index=False)

data["episode"]   = data.groupby(["env_id","algorithm","seed"]).cumcount()
data["steps"]     = data.groupby(["env_id","algorithm","seed"])["l"].cumsum()
roll              = lambda s: s.rolling(SMOOTH_WIN, min_periods=1).mean()
data["r_smooth"]  = data.groupby(["env_id","algorithm","seed"])["r"].transform(roll)
data["l_smooth"]  = data.groupby(["env_id","algorithm","seed"])["l"].transform(roll)

for env, df_env in data.groupby("env_id"):

    # smoothed return
    plt.figure(figsize=(7,4))
    for alg, g in df_env.groupby("algorithm"):
        m = g.groupby("steps")["r_smooth"].mean()
        s = g.groupby("steps")["r_smooth"].std()
        plt.plot(m.index, m, label=alg)
        plt.fill_between(m.index, m-s, m+s, alpha=.25)
    plt.title(f"{env} – return"); plt.xlabel("env steps"); plt.ylabel("return")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{env}_return.png"), dpi=200)

    # smoothed episode length
    plt.figure(figsize=(7,4))
    for alg, g in df_env.groupby("algorithm"):
        m = g.groupby("steps")["l_smooth"].mean()
        s = g.groupby("steps")["l_smooth"].std()
        plt.plot(m.index, m, label=alg)
        plt.fill_between(m.index, m-s, m+s, alpha=.25)
    plt.title(f"{env} – episode length"); plt.xlabel("env steps"); plt.ylabel("length")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{env}_length.png"), dpi=200)

    # final boxplot
    tails = [g.tail(min(TAIL_COUNT, len(g)))
             for _, g in df_env.groupby(["algorithm","seed"])]
    final = pd.concat(tails, ignore_index=True)
    plt.figure(figsize=(5,4))
    sns.boxplot(data=final, x="algorithm", y="r")
    plt.title(f"{env} – final return"); plt.ylabel("return"); plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{env}_box.png"), dpi=200)

print("csv   ->", os.path.join(OUTDIR, "merged_all_envs.csv"))
print("plots ->", OUTDIR)
