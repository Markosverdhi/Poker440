"""
train_random.py

This script implements the training loop for the poker RL agent using the BestPokerModel architecture 
and the TrainFullPokerEnv environment. In addition to the standard training procedure, it explicitly sets one 
of the opponent agents (e.g. agent with ID 1) to follow a random action policy.

The script uses an epsilon-greedy strategy for the learning agent, a replay buffer for experience replay,
and periodic updates of a target network. Model checkpoints and training metrics are saved to disk.
"""

import os
import random
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import BestPokerModel, convert_half_to_full_state_dict
from envs import TrainFullPokerEnv
from utils import encode_obs, epsilon_by_frame, ReplayBuffer, log_decision

# Global configuration.
USE_HALF_ENCODING = False  # Use full encoding.
NUM_PLAYERS = 6
STATE_DIM = 52 + 1 + (NUM_PLAYERS - 1) * 52  # For full encoding: 52 + 1 + 5*52 = 313.
NUM_ACTIONS = 6  # Actions: ['fold', 'call', 'check', 'bet_small', 'bet_big', 'all_in'].

# Training hyperparameters.
NUM_EPISODES = 1000000
MAX_EPISODE_STEPS = 500
BUFFER_CAPACITY = 10000
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
GAMMA = 0.99
TARGET_UPDATE_FREQ = 100

# Checkpoint directory.
CHECKPOINT_DIR = "checkpoints"
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# Device configuration.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_policy(obs: dict) -> str:
    """
    A simple random action policy that selects one legal action uniformly at random.
    
    Args:
        obs (dict): Observation containing the key 'legal_actions'.
    
    Returns:
        str: Selected action.
    """
    legal = obs.get('legal_actions', [])
    return random.choice(legal) if legal else 'fold'


def train(agent_checkpoint: torch.nn.Module = None) -> None:
    """
    Trains the RL agent using the BestPokerModel and TrainFullPokerEnv.
    One opponent (with ID 1) is explicitly assigned a random policy.
    
    Args:
        agent_checkpoint (torch.nn.Module, optional): A previously saved model to resume training.
    """
    # Initialize the training environment.
    env = TrainFullPokerEnv(num_players=NUM_PLAYERS)
    
    # Set opponent with ID 1 to follow the random action policy.
    env.opponent_policies[1] = random_policy

    # Instantiate the agent (DQN) and the target network.
    if agent_checkpoint is not None:
        agent = agent_checkpoint.to(DEVICE)
    else:
        agent = BestPokerModel(input_dim=STATE_DIM, num_actions=NUM_ACTIONS).to(DEVICE)
    target_net = BestPokerModel(input_dim=STATE_DIM, num_actions=NUM_ACTIONS).to(DEVICE)
    target_net.load_state_dict(agent.state_dict())
    
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)
    
    global_step = 0
    episode_rewards = []
    metrics_list = []
    checkpoint_idx = 0

    for episode in range(1, NUM_EPISODES + 1):
        obs = env.reset()
        state = encode_obs(obs, use_half_encoding=USE_HALF_ENCODING)
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done and episode_steps < MAX_EPISODE_STEPS:
            episode_steps += 1
            # If it's the agent's turn, select an action using epsilon-greedy strategy.
            if env.current_player == env.agent_id:
                epsilon = epsilon_by_frame(global_step)
                if random.random() < epsilon:
                    action_idx = random.randrange(NUM_ACTIONS)
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    with torch.no_grad():
                        q_values = agent(state_tensor)
                    action_idx = q_values.argmax(dim=1).item()
                action_str = env.action_list[action_idx]
                log_decision(f"[Agent Decision] Episode {episode}, Step {episode_steps}: {action_str}")
                
                next_obs, reward, done, info = env.step(action_str)
                next_state = encode_obs(next_obs, use_half_encoding=USE_HALF_ENCODING)
                replay_buffer.push(state, action_idx, reward, next_state, done)
                state = next_state
                global_step += 1
                episode_reward += reward
            else:
                # For non-agent turns, the environment will use the assigned opponent policies.
                # Pass a dummy action (e.g., 'call') to trigger the opponent action loop.
                _, reward, done, info = env.step('call')
                episode_reward += reward
            
            # Periodically update the target network.
            if global_step % TARGET_UPDATE_FREQ == 0 and global_step > 0:
                target_net.load_state_dict(agent.state_dict())
            
            # Perform a DQN update if sufficient samples are available.
            if len(replay_buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                states_tensor = torch.tensor(states, dtype=torch.float32, device=DEVICE)
                actions_tensor = torch.tensor(actions, dtype=torch.long, device=DEVICE).unsqueeze(1)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
                next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
                dones_tensor = torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)
                
                q_values = agent(states_tensor).gather(1, actions_tensor)
                with torch.no_grad():
                    next_q_values = target_net(next_states_tensor).max(dim=1, keepdim=True)[0]
                target = rewards_tensor + GAMMA * next_q_values * (1 - dones_tensor)
                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else episode_reward
        metrics_list.append({
            'episode': episode,
            'reward': episode_reward,
            'avg_reward': avg_reward,
            'epsilon': epsilon_by_frame(global_step)
        })
        
        # Save checkpoints and update metrics periodically.
        if episode % 50 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{checkpoint_idx}.pt")
            torch.save(agent.state_dict(), checkpoint_path)
            checkpoint_idx = (checkpoint_idx + 1) % 100
        
        if episode % 100 == 0:
            print(f"Episode {episode} - Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon_by_frame(global_step):.2f}")
        
        if episode % 10000 == 0:
            metrics_file = os.path.join(CHECKPOINT_DIR, "metrics.csv")
            with open(metrics_file, "w", newline="") as csvfile:
                fieldnames = ['episode', 'reward', 'avg_reward', 'epsilon']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metrics_list)
            print(f"Metrics saved to {metrics_file} at episode {episode}")
    
    # Save the final agent checkpoint and metrics.
    final_checkpoint = os.path.join(CHECKPOINT_DIR, "final_agent_checkpoint.pt")
    torch.save(agent.state_dict(), final_checkpoint)
    metrics_file = os.path.join(CHECKPOINT_DIR, "metrics.csv")
    with open(metrics_file, "w", newline="") as csvfile:
        fieldnames = ['episode', 'reward', 'avg_reward', 'epsilon']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_list)
    print("Training complete. Final checkpoint saved at:", final_checkpoint)


if __name__ == "__main__":
    train()
