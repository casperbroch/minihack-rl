# run_plots.py  : Load training logs, aggregate results, and generate performance plots.
#
# Author       : Casper Bröcheler <casper.jxb@gmail.com>
# GitHub       : https://github.com/casperbroch
# Affiliation  : Maastricht University


import glob
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

CONFIG = {
    'log_dirs': [
        'data/logs/PPO',
        #'data/logs/RecurrentPPO',
        #'data/logs/PPO_RND',
        #'data/logs/QRDQN',
    ],
    'smoothing_window': 100,                # for moving average
    'last_n_episodes': 100,                 # for boxplot
    'output_base_dir': 'plots',             # base folder; a timestamp subfolder will be made under here
    'model_source_base': 'data/models',        # where to find saved models
}

# Read all CSVs in log_dir, add algorithm tag, and sort by elapsed time
def load_and_concatenate_csvs(log_dir, algorithm_name):
    pattern = os.path.join(log_dir, '*.csv')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {log_dir}")
    df_list = []
    for path in files:
        df = pd.read_csv(path, comment='#')
        df['algorithm'] = algorithm_name
        df_list.append(df)
    combined = pd.concat(df_list, ignore_index=True)
    combined.sort_values('t', inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined

# Ensure output_dir exists, then write the combined DataFrame to CSV
def save_combined_csv(df, output_dir, algorithm_name):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{algorithm_name}_combined_monitor.csv"
    out_path = os.path.join(output_dir, filename)
    df.to_csv(out_path, index=False)
    print(f"Saved combined CSV to {out_path}")
    return out_path

# Compute a rolling mean over the given window
def moving_average(series, window):
    return series.rolling(window=window, min_periods=1).mean()

# Plot and save smoothed returns over episodes for each algorithm
def plot_smoothed_return(all_dfs, window, output_dir):
    plt.figure()
    for df in all_dfs:
        df['return_smooth'] = moving_average(df['r'], window)
        plt.plot(df.index, df['return_smooth'], label=df['algorithm'].iloc[0])
    plt.xlabel('Episode')
    plt.ylabel(f'Return (MA {window})')
    plt.title('Smoothed Return Over Episodes')
    plt.legend()
    out = os.path.join(output_dir, "compare_smoothed_return.png")
    plt.savefig(out)
    print(f"Plot saved: {out}")
    plt.close()

# Plot and save smoothed episode lengths over episodes
def plot_episode_length(all_dfs, window, output_dir):
    plt.figure()
    for df in all_dfs:
        df['length_smooth'] = moving_average(df['l'], window)
        plt.plot(df.index, df['length_smooth'], label=df['algorithm'].iloc[0])
    plt.xlabel('Episode')
    plt.ylabel(f'Episode Length (MA {window})')
    plt.title('Episode Length Over Episodes')
    plt.legend()
    out = os.path.join(output_dir, "compare_episode_length.png")
    plt.savefig(out)
    print(f"Plot saved: {out}")
    plt.close()

# Create and save a boxplot of the last_n episode returns
def plot_final_performance_boxplot(all_dfs, last_n, output_dir):
    data = [df['r'].tail(last_n).values for df in all_dfs]
    labels = [df['algorithm'].iloc[0] for df in all_dfs]
    plt.figure()
    plt.boxplot(data, labels=labels)
    plt.ylabel('Return')
    plt.title(f'Final {last_n} Episodes Performance')
    out = os.path.join(output_dir, "compare_final_performance_boxplot.png")
    plt.savefig(out)
    print(f"Plot saved: {out}")
    plt.close()

# Plot and save smoothed returns as a function of real time
def plot_smoothed_return_time(all_dfs, window, output_dir):
    plt.figure()
    for df in all_dfs:
        df['return_time_smooth'] = moving_average(df['r'], window)
        plt.plot(df['t'], df['return_time_smooth'], label=df['algorithm'].iloc[0])
    plt.xlabel('Time (s)')
    plt.ylabel(f'Return (MA {window})')
    plt.title('Smoothed Return Over Real Time')
    plt.legend()
    out = os.path.join(output_dir, "compare_smoothed_return_time.png")
    plt.savefig(out)
    print(f"Plot saved: {out}")
    plt.close()

# Copy the saved model directory for the algorithm into the output folder
def copy_model_folder(alg_name, model_source_base, out_dir):
    src = os.path.join(model_source_base, alg_name)
    dst_parent = os.path.join(out_dir, 'models')
    dst = os.path.join(dst_parent, alg_name)
    if not os.path.isdir(src):
        print(f"Warning: model directory not found: {src}")
        return
    os.makedirs(dst_parent, exist_ok=True)
    shutil.copytree(src, dst)
    print(f"Copied model folder from {src} to {dst}")

# Set up output directories, process each log_dir, and generate all plots
def main():
    cfg               = CONFIG
    log_dirs          = cfg['log_dirs']
    window            = cfg['smoothing_window']
    last_n            = cfg['last_n_episodes']
    base_out          = cfg['output_base_dir']
    model_src_base    = cfg['model_source_base']

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = os.path.join(base_out, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    all_dfs = []
    for log_dir in log_dirs:
        alg_name = os.path.basename(log_dir.rstrip('/'))
        df = load_and_concatenate_csvs(log_dir, alg_name)
        save_combined_csv(df, out_dir, alg_name)
        all_dfs.append(df)
        copy_model_folder(alg_name, model_src_base, out_dir)

    plot_smoothed_return(all_dfs, window, out_dir)
    plot_episode_length(all_dfs, window, out_dir)
    plot_final_performance_boxplot(all_dfs, last_n, out_dir)
    plot_smoothed_return_time(all_dfs, window, out_dir)

if __name__ == '__main__':
    main()
