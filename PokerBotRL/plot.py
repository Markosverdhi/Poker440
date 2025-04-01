import argparse
import csv
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Plot results from training or simulation.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing results.")
    parser.add_argument("--metric", type=str, default="reward", choices=["reward", "episode_outcome", "custom"],
                        help="Metric to plot: reward, episode_outcome, or custom (requires --custom_metric). Default is reward.")
    parser.add_argument("--custom_metric", type=str, default="info",
                        help="Custom metric to plot if --metric is set to custom.")
    return parser.parse_args()

def plot_reward_episodes(csv_file):
    episodes = []
    rewards = []

    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Try both capitalized and lowercase keys
            ep = row.get('Episode') or row.get('episode')
            rew = row.get('Reward') or row.get('reward')
            if ep is not None and rew is not None:
                episodes.append(int(ep))
                rewards.append(float(rew))

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, marker='o', linestyle='-', color='b', label='Episode Rewards')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_episode_outcomes(csv_file):
    episodes = []
    outcomes = []

    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            ep = row.get('Episode') or row.get('episode')
            info = row.get('Info') or row.get('info')
            if ep is not None and info is not None:
                episodes.append(int(ep))
                outcomes.append(info)

    plt.figure(figsize=(10, 6))
    plt.scatter(episodes, outcomes, marker='o', color='r', label='Episode Outcomes')
    plt.title('Episode Outcomes')
    plt.xlabel('Episode')
    plt.ylabel('Outcome')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_custom_metric(csv_file, custom_metric):
    episodes = []
    custom_values = []

    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            ep = row.get('Episode') or row.get('episode')
            # Try various casings for the custom metric
            value = row.get(custom_metric) or row.get(custom_metric.lower()) or row.get(custom_metric.capitalize())
            if ep is not None and value is not None:
                episodes.append(int(ep))
                try:
                    # Convert to float if possible; if not, leave as is.
                    custom_values.append(float(value))
                except ValueError:
                    custom_values.append(value)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, custom_values, marker='o', linestyle='-', color='g', label=f'Custom Metric: {custom_metric}')
    plt.title(f'Custom Metric: {custom_metric}')
    plt.xlabel('Episode')
    plt.ylabel(custom_metric.capitalize())
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
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
