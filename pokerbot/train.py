"""
train.py

This script trains the Poker RL Agent using the BestPokerModel architecture and the TrainFullPokerEnv.
Opponents (agents 1–5) are periodically updated in a round-robin fashion:
  - By default, they are updated using a checkpoint snapshot of the agent's model.
  - For agent 1, you can specify a special range:
      --random  : In the given episode range, agent 1 uses a random policy.
      --variable: Every 1000 episodes, agent 1 is randomly assigned either a model or a random policy.
Other opponents (agents 2–5) always use the model checkpoint policy.
  
Modifications in this version:
  • Hyperparameter defaults have been adjusted:
      - Learning rate decreased from 1e-3 to 1e-4.
      - Batch size increased from 32 to 64.
      - Target network update frequency increased from 100 to 200 steps.
  • Opponent update frequency is now set to every 100 episodes (instead of 50),
    to provide more stability.
  • Reward shaping via intermediate step penalty is inherited from envs.py.
  
All changes are minimal to preserve backward compatibility with existing checkpoints
and ensure the argparse commands remain the same.
"""

import os
import random
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import BestPokerModel, convert_half_to_full_state_dict
from envs import TrainFullPokerEnv
from utils import encode_obs, epsilon_by_frame, ReplayBuffer, log_decision

# Global configuration
USE_HALF_ENCODING = False
NUM_PLAYERS = 6
STATE_DIM = 52 + 1 + (NUM_PLAYERS - 1) * 52  # 52 + 1 + 5*52 = 313 for full encoding
NUM_ACTIONS = 6  # ['fold', 'call', 'check', 'bet_small', 'bet_big', 'all_in']

class Train:
    def __init__(self, episodes, random_range, variable_mode):
        self.num_episodes = episodes
        self.random_range = self._parse_range(random_range) if random_range else None
        self.variable_mode = variable_mode
        self.checkpoint_dir = "checkpoints"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        # New attribute to track round-robin updates among opponents 1-5.
        self.current_update_index = 0
        # Training hyperparameters (adjusted)
        self.max_episode_steps = 500
        self.buffer_capacity = 10000
        self.batch_size = 64            # Increased from 32
        self.learning_rate = 1e-4         # Decreased from 1e-3
        self.gamma = 0.99
        self.target_update_freq = 200     # Increased from 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _parse_range(self, range_str):
        """Parses a range string of the format 'start-end' into a tuple (start, end)."""
        parts = range_str.split('-')
        return (int(parts[0]), int(parts[1]))
    
    def _in_range(self, episode, range_tuple):
        """Checks if the current episode is within a given range."""
        if range_tuple is None:
            return False
        return range_tuple[0] <= episode <= range_tuple[1]
    
    def make_opponent_policy(self, opponent_model):
        """Creates a policy function based on the given opponent model."""
        action_index_to_str = {0: 'fold', 1: 'call', 2: 'check', 3: 'bet_small', 4: 'bet_big', 5: 'all_in'}
        def policy_fn(obs):
            state = encode_obs(obs, use_half_encoding=USE_HALF_ENCODING)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = opponent_model(state_tensor)
            action_idx = q_values.argmax(dim=1).item()
            return action_index_to_str[action_idx]
        return policy_fn
    
    def random_policy(self, obs):
        """A simple random action policy for opponents."""
        legal = obs.get('legal_actions', [])
        return random.choice(legal) if legal else 'fold'
    
    def update_opponent_policy(self, opponent_id, policy_type, env):
        """
        Updates the policy for the specified opponent.
        policy_type can be:
          - "model": Loads a random checkpoint of our model.
          - "random": Uses a random action policy.
        """
        if policy_type == "model":
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pt')]
            if checkpoint_files:
                random_checkpoint = random.choice(checkpoint_files)
                full_checkpoint_path = os.path.join(self.checkpoint_dir, random_checkpoint)
                opp_checkpoint = torch.load(full_checkpoint_path, map_location=self.device)
                # Guard against missing key 'fc1.weight'
                if "fc1.weight" in opp_checkpoint and opp_checkpoint["fc1.weight"].shape[1] == (26 + 1 + (NUM_PLAYERS - 1) * 26):
                    opp_checkpoint = convert_half_to_full_state_dict(opp_checkpoint)
                    print(f"Converted checkpoint {random_checkpoint} from half to full dimensions.")
                opponent_model = BestPokerModel(input_dim=STATE_DIM, num_actions=NUM_ACTIONS).to(self.device)
                opponent_model.load_state_dict(opp_checkpoint)
                opponent_model.eval()
                policy_fn = self.make_opponent_policy(opponent_model)
                env.opponent_policies[opponent_id] = policy_fn
                print(f"Updated opponent {opponent_id} with model policy from {random_checkpoint}.")
        elif policy_type == "random":
            env.opponent_policies[opponent_id] = self.random_policy
            print(f"Updated opponent {opponent_id} with random policy.")
        else:
            print("Unknown policy type for opponent update.")
    
    def run(self):
        env = TrainFullPokerEnv(num_players=NUM_PLAYERS)
        agent = BestPokerModel(input_dim=STATE_DIM, num_actions=NUM_ACTIONS).to(self.device)
        target_net = BestPokerModel(input_dim=STATE_DIM, num_actions=NUM_ACTIONS).to(self.device)
        target_net.load_state_dict(agent.state_dict())
        optimizer = optim.Adam(agent.parameters(), lr=self.learning_rate)
        replay_buffer = ReplayBuffer(capacity=self.buffer_capacity)
        
        global_step = 0
        episode_rewards = []
        metrics_list = []
        checkpoint_idx = 0
        
        # Opponent IDs: 1-5.
        opponent_ids = [1, 2, 3, 4, 5]
        
        for episode in range(1, self.num_episodes + 1):
            obs = env.reset()
            state = encode_obs(obs, use_half_encoding=USE_HALF_ENCODING)
            done = False
            episode_reward = 0
            episode_steps = 0
            
            while not done and episode_steps < self.max_episode_steps:
                episode_steps += 1
                if env.current_player == env.agent_id:
                    epsilon = epsilon_by_frame(global_step)
                    if random.random() < epsilon:
                        action_idx = random.randrange(NUM_ACTIONS)
                    else:
                        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
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
                    # For non-agent turns, the environment uses assigned opponent policies (or defaults to "call").
                    _, reward, done, info = env.step('call')
                    episode_reward += reward
                    
                if global_step % self.target_update_freq == 0 and global_step > 0:
                    target_net.load_state_dict(agent.state_dict())
                    
                if len(replay_buffer) >= self.batch_size:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)
                    states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
                    actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
                    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
                    next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
                    dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
                    
                    q_values = agent(states_tensor).gather(1, actions_tensor)
                    with torch.no_grad():
                        next_q_values = target_net(next_states_tensor).max(dim=1, keepdim=True)[0]
                    target = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)
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
            
            # Periodically update one opponent (round-robin) and save checkpoints.
            if episode % 200 == 0:  # Updated frequency: now every 100 episodes
                # Determine which opponent to update in round-robin order.
                update_opponent_id = opponent_ids[self.current_update_index]
                if update_opponent_id == 1:
                    if self.variable_mode and episode % 1000 == 0:
                        policy_type = random.choice(["model", "random"])
                    elif self._in_range(episode, self.random_range):
                        policy_type = "random"
                    else:
                        policy_type = "model"
                else:
                    policy_type = "model"
                self.update_opponent_policy(update_opponent_id, policy_type, env)
                
                # Save a checkpoint.
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{checkpoint_idx}.pt")
                torch.save(agent.state_dict(), checkpoint_path)
                print(f"Saved checkpoint at {checkpoint_path}")
                checkpoint_idx = (checkpoint_idx + 1) % 100
                
                # Advance to the next opponent in the round-robin cycle.
                self.current_update_index = (self.current_update_index + 1) % len(opponent_ids)
            
            if episode % 100 == 0:
                print(f"Episode {episode} - Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon_by_frame(global_step):.2f}")
            
            if episode % 10000 == 0:
                metrics_file = os.path.join(self.checkpoint_dir, "metrics.csv")
                with open(metrics_file, "a", newline="") as csvfile:
                    fieldnames = ['episode', 'reward', 'avg_reward', 'epsilon']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(metrics_list)
                print(f"Metrics saved to {metrics_file} at episode {episode}")
        
        # Save final checkpoint and metrics.
        final_checkpoint = os.path.join(self.checkpoint_dir, "final_agent_checkpoint.pt")
        torch.save(agent.state_dict(), final_checkpoint)
        metrics_file = os.path.join(self.checkpoint_dir, "metrics.csv")
        with open(metrics_file, "a", newline="") as csvfile:
            fieldnames = ['episode', 'reward', 'avg_reward', 'epsilon']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_list)
        print("Training complete. Final checkpoint saved at:", final_checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Poker RL Agent with periodic opponent updates.")
    parser.add_argument("--episodes", type=int, default=1000000, help="Total number of training episodes.")
    parser.add_argument("--random", type=str, default=None, help="Episode range for using random policy for opponent 1 (format: start-end).")
    parser.add_argument("--variable", action="store_true", help="Enable variable training mode for opponent 1 (switch between model and random every 1000 episodes).")
    args = parser.parse_args()
    
    trainer = Train(episodes=args.episodes, random_range=args.random, variable_mode=args.variable)
    trainer.run()
