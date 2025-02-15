def plot_rewards(rewards):
    episodes = np.arange(len(rewards))
    
    def moving_average(x, window_size=20):
        return np.convolve(x, np.ones(window_size)/window_size, mode='valid')
    
    smoothed_rewards = moving_average(rewards, window_size=20)
    
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label="Reward per Episode", alpha=0.3)
    plt.plot(episodes[:len(smoothed_rewards)], smoothed_rewards, label="Moving Average (window=20)", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_rewards(rewards)
