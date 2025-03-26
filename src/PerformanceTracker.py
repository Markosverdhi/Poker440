import os
import pandas as pd
import matplotlib.pyplot as plt

def plot():
    metrics_file = 'metrics.csv'
    plot_dir = 'plots'
    
    if not os.path.exists(metrics_file):
        print(f"Metrics file {metrics_file} not found. Ensure your training script outputs this file.")
        return

    # Create directory for saving plots if it doesn’t exist
    os.makedirs(plot_dir, exist_ok=True)

    # Load CSV into Pandas DataFrame
    df = pd.read_csv(metrics_file)
    if 'episode' not in df.columns:
        print("The CSV file must contain an 'episode' column.")
        return

    # Set up figure with multiple subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()  # Flatten for easier indexing
    plot_index = 0  # Track which subplot index to use

    # Plot Episode Reward
    if 'reward' in df.columns:
        axes[plot_index].plot(df['episode'], df['reward'], color='blue', label='Episode Reward')
        axes[plot_index].set_xlabel('Episode')
        axes[plot_index].set_ylabel('Reward')
        axes[plot_index].set_title('Episode Reward Over Time')
        axes[plot_index].grid(True)
        axes[plot_index].legend()
        plot_index += 1

    # Plot Average Reward (Rolling Mean for Smoother Trend)
    if 'avg_reward' in df.columns:
        df['smoothed_avg_reward'] = df['avg_reward'].rolling(window=10).mean()  # Rolling mean window=10
        axes[plot_index].plot(df['episode'], df['smoothed_avg_reward'], color='green', label='Smoothed Avg Reward')
        axes[plot_index].set_xlabel('Episode')
        axes[plot_index].set_ylabel('Avg Reward')
        axes[plot_index].set_title('Smoothed Average Reward Over Time')
        axes[plot_index].grid(True)
        axes[plot_index].legend()
        plot_index += 1

    # Plot Epsilon Decay
    if 'epsilon' in df.columns:
        axes[plot_index].plot(df['episode'], df['epsilon'], color='red', label='Epsilon')
        axes[plot_index].set_xlabel('Episode')
        axes[plot_index].set_ylabel('Epsilon')
        axes[plot_index].set_title('Epsilon Decay Over Time')
        axes[plot_index].grid(True)
        axes[plot_index].legend()
        plot_index += 1

    # Plot Loss Curve (MSE Loss Over Time)
    if 'loss' in df.columns:
        axes[plot_index].plot(df['episode'], df['loss'], color='purple', label='MSE Loss')
        axes[plot_index].set_xlabel('Episode')
        axes[plot_index].set_ylabel('Loss')
        axes[plot_index].set_title('Loss Curve Over Time')
        axes[plot_index].grid(True)
        axes[plot_index].legend()
        plot_index += 1

    # Plot Average Q-Value Over Time
    if 'avg_q_value' in df.columns:
        axes[plot_index].plot(df['episode'], df['avg_q_value'], color='orange', label='Avg Q-Value')
        axes[plot_index].set_xlabel('Episode')
        axes[plot_index].set_ylabel('Q-Value')
        axes[plot_index].set_title('Average Q-Value Over Time')
        axes[plot_index].grid(True)
        axes[plot_index].legend()
        plot_index += 1

    # Hide any unused subplots
    for i in range(plot_index, len(axes)):
        fig.delaxes(axes[i])

    # Adjust layout and save the figure
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, 'dqn_performance.png')
    plt.savefig(plot_path)
    plt.show()

    print(f"Plot saved to {plot_path}")