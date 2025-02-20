import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from DQN import ModelTrain, DQN

PATH_TO_MODELS = os.path.join(os.path.dirname(__file__), "../models")

class PerformanceTracker:
    def __init__(self, agent, model_name="Unknown", epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        self.agent = agent
        self.model_name = model_name
        self.epsilon_values = [epsilon_start]
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rewards = []

    def track_rewards(self, reward):
        """Track rewards over episodes."""
        self.rewards.append(reward)

    def plot_rewards(self, rewards):
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

    def epsilon_decay(self, episode):
        """Track epsilon decay over time for DQN. This won't work with A3C so don't try to get it to work."""
        epsilon = max(self.epsilon_end, self.epsilon_values[-1] * self.epsilon_decay)
        self.epsilon_values.append(epsilon)
        return epsilon

    def plot_epsilon(self):
        """Plot the epsilon decay progression for DQN."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.epsilon_values, label="Epsilon Decay", color="green")
        plt.xlabel("Episodes")
        plt.ylabel("Epsilon Value")
        plt.title("Epsilon Decay Over Training")
        plt.legend()
        plt.grid()
        plt.show()

    def compare_models(self, test_version):
        """Load a previous model and compare it to the current model."""
        model_path = os.path.join(PATH_TO_MODELS, f"{test_version}.pth")
        
        if not os.path.exists(model_path):
            print(f"Error: Comparison model {test_version} not found in {PATH_TO_MODELS}.")
            return None

        print(f"Loading comparison model: {test_version}")
        comparison_model = ModelTrain(DQN)
        comparison_model.model.load_state_dict(torch.load(model_path))

        test_rewards = []
        for _ in range(10):
            state = comparison_model.env.reset()
            if isinstance(state, tuple):
                state = state[0]
            done = False
            total_reward = 0

            while not done:
                action = comparison_model.select_action(state)
                next_state, reward, done, _ = comparison_model.env.step(action)
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                total_reward += reward
                state = next_state

            test_rewards.append(total_reward)

        avg_test_reward = np.mean(test_rewards)
        print(f"Comparison Model ({test_version}) Average Reward: {avg_test_reward:.2f}")
        return test_rewards
