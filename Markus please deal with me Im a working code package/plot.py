#!/usr/bin/env python

import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    metrics_file = 'metrics.csv'
    if not os.path.exists(metrics_file):
        print(f"Metrics file {metrics_file} not found. Please ensure your training script outputs this file.")
        return

    # Load metrics from the CSV file.
    df = pd.read_csv(metrics_file)
    if 'episode' not in df.columns:
        print("The CSV file must contain an 'episode' column.")
        return

    # Plot Episode Reward over Time.
    if 'reward' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['episode'], df['reward'], label='Episode Reward', color='blue')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Reward Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig('episode_reward.png')
        plt.show()

    # Plot Average Reward (per 100 episodes) over Time.
    if 'avg_reward' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['episode'], df['avg_reward'], label='Average Reward (per 100 episodes)', color='green')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Average Reward Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig('average_reward.png')
        plt.show()

    # Plot Epsilon Decay over Time.
    if 'epsilon' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['episode'], df['epsilon'], label='Epsilon', color='red')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig('epsilon_decay.png')
        plt.show()

if __name__ == "__main__":
    main()
