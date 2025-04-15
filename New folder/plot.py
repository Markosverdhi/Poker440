import argparse
import csv
import re
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Plot results from training or simulation.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing results.")
    parser.add_argument("--metric", type=str, default="reward", choices=["reward", "episode_outcome", "custom"],
                        help="Metric to plot: reward, episode_outcome, or custom (requires --custom_metric). Default is reward.")
    parser.add_argument("--custom_metric", type=str, default="info",
                        help="Custom metric to plot if --metric is set to custom.")
    return parser.parse_args()

def extract_episode(ep_str):
    """
    Extracts the first integer found in the episode string.
    Returns an integer if found; otherwise, returns None.
    """
    match = re.search(r'\d+', ep_str)
    return int(match.group(0)) if match else None

def plot_reward_episodes(csv_file):
    episodes = []
    rewards = []

    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            ep_raw = row.get('Episode') or row.get('episode')
            rew_raw = row.get('Reward') or row.get('reward')
            if ep_raw is not None and rew_raw is not None:
                ep = extract_episode(ep_raw)
                if ep is None:
                    continue
                try:
                    rew = float(rew_raw)
                except ValueError:
                    continue
                episodes.append(ep)
                rewards.append(rew)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, marker='o', linestyle='-', label='Episode Rewards')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Save image to current directory
    output_file = "plots/episode_rewards.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()

def plot_episode_outcomes(csv_file):
    episodes = []
    outcomes = []

    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            ep_raw = row.get('Episode') or row.get('episode')
            info = row.get('Info') or row.get('info')
            if ep_raw is not None and info is not None:
                ep = extract_episode(ep_raw)
                if ep is None:
                    continue
                episodes.append(ep)
                outcomes.append(info)

    plt.figure(figsize=(10, 6))
    plt.scatter(episodes, outcomes, marker='o', label='Episode Outcomes')
    plt.title('Episode Outcomes')
    plt.xlabel('Episode')
    plt.ylabel('Outcome')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Save image to current directory
    output_file = "plots/episode_outcomes.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()

def plot_custom_metric(csv_file, custom_metric):
    episodes = []
    custom_values = []

    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            ep_raw = row.get('Episode') or row.get('episode')
            value_raw = row.get(custom_metric) or row.get(custom_metric.lower()) or row.get(custom_metric.capitalize())
            if ep_raw is not None and value_raw is not None:
                ep = extract_episode(ep_raw)
                if ep is None:
                    continue
                episodes.append(ep)
                try:
                    custom_values.append(float(value_raw))
                except ValueError:
                    custom_values.append(value_raw)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, custom_values, marker='o', linestyle='-', label=f'Custom Metric: {custom_metric}')
    plt.title(f'Custom Metric: {custom_metric}')
    plt.xlabel('Episode')
    plt.ylabel(custom_metric.capitalize())
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Save image to current directory
    output_file = f"plots/custom_metric_{custom_metric}.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()

def main():
    args = parse_args()

    if args.metric == "reward":
        plot_reward_episodes(args.csv_file)
    elif args.metric == "episode_outcome":
        plot_episode_outcomes(args.csv_file)
    elif args.metric == "custom":
        plot_custom_metric(args.csv_file, args.custom_metric)

if __name__ == "__main__":
    main()
