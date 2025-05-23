#!/usr/bin/env python3

"""
plot_monitor_csvs.py

A simple script to:
 1. Load all Monitor CSVs from a specified directory.
 2. Concatenate and sort by time (column 't').
 3. Save the combined CSV (including algorithm name).
 4. Generate four plots (filenames include algorithm name):
    - Smoothed episodic return over episodes.
    - Episode length over episodes.
    - Boxplot of final policy performance (last N episodes).
    - Smoothed return over real time (seconds).

Configure parameters in the `CONFIG` dictionary below and run:
    python plot_monitor_csvs.py
"""

import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===
CONFIG = {
    # Directory containing Monitor CSVs
    'log_dir': 'data/logs/PPO',
    # Smoothing window (number of episodes)
    'smoothing_window': 100,
    # Number of last episodes for boxplot
    'last_n_episodes': 100,
    # Output directory for CSV and plots
    'output_dir': 'plots',
}
# ======================


def load_and_concatenate_csvs(log_dir):
    pattern = os.path.join(log_dir, '*.csv')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {log_dir}")
    df_list = []
    for path in files:
        df = pd.read_csv(path, comment='#')
        df_list.append(df)
    combined = pd.concat(df_list, ignore_index=True)
    combined.sort_values('t', inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def save_combined_csv(df, output_dir, algorithm_name):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{algorithm_name}_combined_monitor.csv"
    out_path = os.path.join(output_dir, filename)
    df.to_csv(out_path, index=False)
    print(f"Saved combined CSV to {out_path}")
    return out_path


def moving_average(series, window):
    return series.rolling(window=window, min_periods=1).mean()


def plot_smoothed_return(df, window, output_dir, algorithm_name):
    df['return_smooth'] = moving_average(df['r'], window)
    plt.figure()
    plt.plot(df.index, df['return_smooth'])
    plt.xlabel('Episode')
    plt.ylabel(f'Return (MA {window})')
    plt.title(f'Smoothed Return Over Episodes ({algorithm_name})')
    filename = f"{algorithm_name}_smoothed_return.png"
    out = os.path.join(output_dir, filename)
    plt.savefig(out)
    print(f"Plot saved: {out}")
    plt.close()


def plot_episode_length(df, window, output_dir, algorithm_name):
    df['length_smooth'] = moving_average(df['l'], window)
    plt.figure()
    plt.plot(df.index, df['length_smooth'])
    plt.xlabel('Episode')
    plt.ylabel(f'Episode Length (MA {window})')
    plt.title(f'Episode Length Over Episodes ({algorithm_name})')
    filename = f"{algorithm_name}_episode_length.png"
    out = os.path.join(output_dir, filename)
    plt.savefig(out)
    print(f"Plot saved: {out}")
    plt.close()


def plot_final_performance_boxplot(df, last_n, output_dir, algorithm_name):
    final_returns = df['r'].tail(last_n)
    plt.figure()
    plt.boxplot(final_returns)
    plt.xticks([1], [algorithm_name])
    plt.ylabel('Return')
    plt.title(f'Final {last_n} Episodes Performance ({algorithm_name})')
    filename = f"{algorithm_name}_final_performance_boxplot.png"
    out = os.path.join(output_dir, filename)
    plt.savefig(out)
    print(f"Plot saved: {out}")
    plt.close()


def plot_smoothed_return_time(df, window, output_dir, algorithm_name):
    df['return_time_smooth'] = moving_average(df['r'], window)
    plt.figure()
    plt.plot(df['t'], df['return_time_smooth'])
    plt.xlabel('Time (s)')
    plt.ylabel(f'Return (MA {window})')
    plt.title(f'Smoothed Return Over Real Time ({algorithm_name})')
    filename = f"{algorithm_name}_smoothed_return_time.png"
    out = os.path.join(output_dir, filename)
    plt.savefig(out)
    print(f"Plot saved: {out}")
    plt.close()


def main():
    cfg = CONFIG
    log_dir = cfg['log_dir']
    window = cfg['smoothing_window']
    last_n = cfg['last_n_episodes']
    out_dir = cfg['output_dir']
    algorithm = os.path.basename(log_dir.rstrip('/'))

    df = load_and_concatenate_csvs(log_dir)
    save_combined_csv(df, out_dir, algorithm)

    plot_smoothed_return(df, window, out_dir, algorithm)
    plot_episode_length(df, window, out_dir, algorithm)
    plot_final_performance_boxplot(df, last_n, out_dir, algorithm)
    plot_smoothed_return_time(df, window, out_dir, algorithm)


if __name__ == '__main__':
    main()
