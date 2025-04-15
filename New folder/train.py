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
  • Opponent update frequency is now set to every 200 episodes (instead of 50),
    to provide more stability.
  • Reward shaping via intermediate step penalty is inherited from envs.py.
  • Training loop adjusted for envs.py changes (persistent stacks, game end condition, turn handling).

All changes are minimal to preserve backward compatibility with existing checkpoints
and ensure the argparse commands remain the same.
"""

import os
import random
import csv
import argparse
from collections import deque # Import deque if needed for type hinting or checks, though not strictly necessary here

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import BestPokerModel, convert_half_to_full_state_dict
from envs import TrainFullPokerEnv # Use the environment with multi-round logic
from utils import encode_obs, epsilon_by_frame, ReplayBuffer, log_decision

# Global configuration
USE_HALF_ENCODING = False # Keep consistency if models were trained with this
NUM_PLAYERS = 6
# Adjust STATE_DIM based on whether half encoding is used or not
if USE_HALF_ENCODING:
    # Example dimension for half encoding (adjust if needed)
    STATE_DIM = 26 + 1 + (NUM_PLAYERS - 1) * 26 # 26 + 1 + 5*26 = 157
else:
    STATE_DIM = 52 + 1 + (NUM_PLAYERS - 1) * 52  # 52 + 1 + 5*52 = 313 for full encoding

NUM_ACTIONS = 6  # ['fold', 'call', 'check', 'bet_small', 'bet_big', 'all_in']

class Train:
    def __init__(self, episodes, random_range, variable_mode, resume_from=None):
        self.num_episodes = episodes
        self.random_range = self._parse_range(random_range) if random_range else None
        self.variable_mode = variable_mode
        self.resume_from = resume_from
        self.checkpoint_dir = "checkpoints"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.current_update_index = 0 # For round-robin opponent updates
        # Training hyperparameters
        self.max_episode_steps = 1000 # Increased max steps, as episodes can be longer now
        self.buffer_capacity = 10000
        self.batch_size = 64
        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.target_update_freq = 200
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def _parse_range(self, range_str):
        """Parses a range string 'start-end' into a tuple (start, end)."""
        try:
            parts = range_str.split('-')
            return (int(parts[0]), int(parts[1]))
        except:
            raise ValueError("Invalid range format. Use 'start-end'.")

    def _in_range(self, episode, range_tuple):
        """Checks if the current episode is within a given range."""
        return range_tuple and range_tuple[0] <= episode <= range_tuple[1]

    def make_opponent_policy(self, opponent_model):
        """Creates a policy function for an opponent model."""
        action_map = {idx: action for idx, action in enumerate(TrainFullPokerEnv().action_list)} # Get action list from env
        def policy_fn(obs):
            # Opponent policy needs the observation dictionary
            if not isinstance(obs, dict):
                 print(f"Warning: Opponent policy received non-dict obs: {type(obs)}. Defaulting fold.")
                 # This might happen if env state is inconsistent, try to recover gracefully
                 return 'fold'

            legal_actions = obs.get('legal_actions', [])
            if not legal_actions:
                 return 'fold' # No legal actions

            state = encode_obs(obs, use_half_encoding=USE_HALF_ENCODING)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = opponent_model(state_tensor)

            # Choose the best *legal* action
            best_legal_action = None
            # Sort actions by Q-value descending
            sorted_indices = torch.argsort(q_values, dim=1, descending=True).squeeze().tolist()

            for action_idx in sorted_indices:
                action_str = action_map.get(action_idx)
                if action_str in legal_actions:
                    best_legal_action = action_str
                    break

            # Fallback if no predicted action is legal (shouldn't happen ideally)
            if best_legal_action is None:
                print(f"Warning: Opponent model predictions ({q_values}) led to no legal action. Legal: {legal_actions}. Defaulting.")
                if 'check' in legal_actions: best_legal_action = 'check'
                elif 'call' in legal_actions: best_legal_action = 'call'
                else: best_legal_action = random.choice(legal_actions) # Random legal if no check/call

            return best_legal_action
        return policy_fn

    def random_policy(self, obs):
        """A simple random action policy for opponents."""
        if not isinstance(obs, dict):
            print(f"Warning: Random policy received non-dict obs: {type(obs)}. Defaulting fold.")
            return 'fold'
        legal = obs.get('legal_actions', [])
        # Simple heuristic: avoid betting/raising too often randomly
        passive_actions = [a for a in ['fold', 'check', 'call'] if a in legal]
        if passive_actions and random.random() < 0.7: # 70% chance for passive
             return random.choice(passive_actions)
        elif legal: # Otherwise choose any legal action
             return random.choice(legal)
        else:
             return 'fold' # Fallback

    def update_opponent_policy(self, opponent_id, policy_type, env):
        """Updates the policy for the specified opponent."""
        if policy_type == "model":
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pt') and f != 'final_agent_checkpoint.pt']
            if not checkpoint_files:
                print(f"Warning: No checkpoints found to update opponent {opponent_id}. Using random policy.")
                env.opponent_policies[opponent_id] = self.random_policy
                return

            random_checkpoint_name = random.choice(checkpoint_files)
            full_checkpoint_path = os.path.join(self.checkpoint_dir, random_checkpoint_name)
            try:
                opp_checkpoint = torch.load(full_checkpoint_path, map_location=self.device)
                # Check encoding compatibility if needed (using STATE_DIM)
                # Example check based on a known layer's weight shape
                first_layer_key = next(iter(opp_checkpoint)) # Get first key (e.g., 'fc1.weight')
                expected_input_dim = STATE_DIM
                actual_input_dim = opp_checkpoint[first_layer_key].shape[1] # Input dim is usually dim 1

                if actual_input_dim != expected_input_dim:
                     print(f"Warning: Checkpoint {random_checkpoint_name} input dim ({actual_input_dim}) differs from expected ({expected_input_dim}). Skipping update.")
                     # Fallback to random or keep existing policy? For now, skip.
                     # env.opponent_policies[opponent_id] = self.random_policy
                     return

                opponent_model = BestPokerModel(input_dim=STATE_DIM, num_actions=NUM_ACTIONS).to(self.device)
                opponent_model.load_state_dict(opp_checkpoint) # Use strict=True by default unless issues arise
                opponent_model.eval()
                policy_fn = self.make_opponent_policy(opponent_model)
                env.opponent_policies[opponent_id] = policy_fn
                print(f"Updated opponent {opponent_id} with model policy from {random_checkpoint_name}.")

            except Exception as e:
                print(f"Error loading checkpoint {random_checkpoint_name} for opponent {opponent_id}: {e}. Using random policy.")
                env.opponent_policies[opponent_id] = self.random_policy

        elif policy_type == "random":
            env.opponent_policies[opponent_id] = self.random_policy
            print(f"Updated opponent {opponent_id} with random policy.")
        else:
            print(f"Warning: Unknown policy type '{policy_type}' for opponent update.")


    def run(self):
        env = TrainFullPokerEnv(num_players=NUM_PLAYERS)
        agent = BestPokerModel(input_dim=STATE_DIM, num_actions=NUM_ACTIONS).to(self.device)
        target_net = BestPokerModel(input_dim=STATE_DIM, num_actions=NUM_ACTIONS).to(self.device)

        optimizer = optim.Adam(agent.parameters(), lr=self.learning_rate)
        replay_buffer = ReplayBuffer(capacity=self.buffer_capacity)

        start_episode = 1
        global_step = 0
        checkpoint_idx = 0 # For rotating checkpoint saves
        episode_rewards = [] # Track rewards per episode
        metrics_list = [] # For saving metrics periodically

        # --- Resume from checkpoint ---
        if self.resume_from and os.path.exists(self.resume_from):
             print(f"Resuming training from checkpoint: {self.resume_from}")
             checkpoint = torch.load(self.resume_from, map_location=self.device)
             agent.load_state_dict(checkpoint['agent_state_dict'])
             target_net.load_state_dict(checkpoint['target_net_state_dict'])
             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
             start_episode = checkpoint.get('episode', 1) + 1
             global_step = checkpoint.get('global_step', 0)
             # Load replay buffer if saved (optional, requires saving/loading in ReplayBuffer)
             # if 'replay_buffer' in checkpoint: replay_buffer.load_state_dict(checkpoint['replay_buffer'])
             # Load opponent policies (difficult to save/load directly, might need re-initialization)
             # For simplicity, opponents will be initialized fresh or based on latest checkpoints
             print(f"Resuming from Episode {start_episode}, Global Step {global_step}")
        else:
             # Start fresh: load initial state for target net
             target_net.load_state_dict(agent.state_dict())
             print("Starting training from scratch.")


        # Opponent IDs: 1 to NUM_PLAYERS-1
        opponent_ids = list(range(1, NUM_PLAYERS))

        # Initialize opponent policies before starting episodes
        print("Initializing opponent policies...")
        for opp_id in opponent_ids:
             # Default to model policy initially, except for special cases handled later
             self.update_opponent_policy(opp_id, "model", env)

        print("Starting training loop...")
        for episode in range(start_episode, self.num_episodes + 1):
            # --- Episode Setup ---
            obs, info = env.reset() # Unpack tuple
            if info.get("error"): # Handle case where game can't start
                print(f"Error starting episode {episode}: {info['error']}")
                continue # Skip to next episode

            state = encode_obs(obs, use_half_encoding=USE_HALF_ENCODING)
            done = False
            episode_reward = 0
            episode_steps = 0 # Tracks steps *within* this episode

            # --- Main Episode Loop ---
            while not done: # Loop relies on 'done' flag from env.step now
                # Check step limit for safety
                if episode_steps >= self.max_episode_steps:
                    print(f"Episode {episode} reached max steps ({self.max_episode_steps}). Ending episode.")
                    done = True # Force episode end
                    # Optionally, assign a penalty or handle this timeout case
                    break # Exit inner loop

                # --- Agent's Turn ---
                # CORRECTED CHECK: Check if queue is not empty and first element is agent
                if env.players_to_act and env.players_to_act[0] == env.agent_id:
                    episode_steps += 1 # Count agent actions as steps
                    epsilon = epsilon_by_frame(global_step)

                    # Choose action (Epsilon-greedy)
                    if random.random() < epsilon:
                        # Choose a random *legal* action
                        legal_actions = obs.get('legal_actions', [])
                        if legal_actions:
                            action_str = random.choice(legal_actions)
                            action_idx = env.action_list.index(action_str)
                        else:
                            # Agent has no legal actions (shouldn't happen if turn logic is right)
                            print(f"Warning: Agent {env.agent_id} has no legal actions. Forcing fold.")
                            action_str = 'fold'
                            action_idx = env.action_list.index('fold')
                    else:
                        # Choose action from model
                        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                        with torch.no_grad():
                            q_values = agent(state_tensor)
                        # Select best *legal* action from Q-values
                        sorted_indices = torch.argsort(q_values, dim=1, descending=True).squeeze().tolist()
                        action_str = None
                        legal_actions = obs.get('legal_actions', [])
                        if not legal_actions:
                             print(f"Warning: Agent {env.agent_id} has no legal actions (model turn). Forcing fold.")
                             action_str = 'fold'
                        else:
                             for idx in sorted_indices:
                                  potential_action = env.action_list[idx]
                                  if potential_action in legal_actions:
                                       action_str = potential_action
                                       break
                             if action_str is None: # Fallback if model's predictions don't match legal actions
                                  print(f"Warning: Agent model predictions led to no legal action. Legal: {legal_actions}. Defaulting.")
                                  action_str = random.choice(legal_actions) # Choose random legal as fallback

                        action_idx = env.action_list.index(action_str)


                    log_decision(f"[Agent Decision] Ep {episode}, Step {episode_steps}: Action='{action_str}'")

                    # --- Environment Step ---
                    # env.step processes agent action AND subsequent opponent actions
                    next_obs, reward, done, info = env.step(action_str)

                    # Encode the state resulting from agent + opponent actions
                    next_state = encode_obs(next_obs, use_half_encoding=USE_HALF_ENCODING)

                    # --- Store Experience ---
                    replay_buffer.push(state, action_idx, reward, next_state, done)

                    # --- Update State and Reward ---
                    state = next_state # Current state for the *next* agent decision
                    obs = next_obs # Update obs for the next loop iteration (for legal action checks)
                    episode_reward += reward
                    global_step += 1 # Increment global step after agent's transition

                    # --- Learning Step ---
                    if len(replay_buffer) >= self.batch_size:
                        states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(self.batch_size)
                        states_tensor = torch.tensor(states_b, dtype=torch.float32, device=self.device)
                        actions_tensor = torch.tensor(actions_b, dtype=torch.long, device=self.device).unsqueeze(1)
                        rewards_tensor = torch.tensor(rewards_b, dtype=torch.float32, device=self.device).unsqueeze(1)
                        next_states_tensor = torch.tensor(next_states_b, dtype=torch.float32, device=self.device)
                        dones_tensor = torch.tensor(dones_b, dtype=torch.float32, device=self.device).unsqueeze(1)

                        # Compute Q(s, a)
                        q_values = agent(states_tensor).gather(1, actions_tensor)

                        # Compute V(s') = max_a' Q_target(s', a')
                        with torch.no_grad():
                            next_q_values = target_net(next_states_tensor).max(dim=1, keepdim=True)[0]

                        # Compute target Q value: r + gamma * V(s') * (1 - done)
                        target = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)

                        # Compute loss and update agent network
                        loss = nn.MSELoss()(q_values, target)
                        optimizer.zero_grad()
                        loss.backward()
                        # Optional: Gradient clipping
                        # torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
                        optimizer.step()

                    # --- Target Network Update ---
                    if global_step % self.target_update_freq == 0:
                        target_net.load_state_dict(agent.state_dict())
                        # print(f"Step {global_step}: Updated target network.") # Optional logging

                elif not env.players_to_act:
                     # This case might occur if round ends exactly when it's not agent's turn
                     # For example, last opponent folds. 'done' should be true from env.step.
                     # If 'done' is not true here, it might indicate an env logic issue.
                     if not done:
                          print(f"Warning: Episode {episode}, Step {episode_steps}: Not agent's turn, but player queue empty and not done?")
                          # Force break or try to recover state? For safety, break.
                          break
                # Implicit else: It's not the agent's turn, and the queue is not empty.
                # The loop continues, env.step was already called previously,
                # state/obs reflect the situation waiting for the agent.

            # --- End of Episode ---
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:]) # Moving average over last 100 episodes
            metrics_list.append({
                'episode': episode, 'reward': episode_reward, 'avg_reward': avg_reward,
                'steps': episode_steps, 'epsilon': epsilon_by_frame(global_step) # Log steps and epsilon
            })
            print(f"Episode {episode} finished. Reward: {episode_reward:.2f}, Steps: {episode_steps}, Avg Reward (100ep): {avg_reward:.2f}")


            # --- Opponent Updates & Checkpointing ---
            # Update frequency adjusted to 50 as per docstring
            if episode % 50 == 0:
                # Determine which opponent to update (round-robin)
                update_opponent_id = opponent_ids[self.current_update_index]

                # Determine policy type based on flags and episode number
                policy_type = "model" # Default
                if update_opponent_id == 1: # Special handling only for opponent 1
                    if self.variable_mode and episode % 1000 == 0: # Variable mode overrides random range if conditions met
                        policy_type = random.choice(["model", "random"])
                        print(f"Variable mode trigger: Opponent 1 set to '{policy_type}' policy.")
                    elif self._in_range(episode, self.random_range):
                        policy_type = "random"
                        print(f"Random range trigger: Opponent 1 set to 'random' policy.")
                # else: Opponents 2-5 always use model policy

                # Update the selected opponent's policy
                self.update_opponent_policy(update_opponent_id, policy_type, env)

                # Save a checkpoint (include necessary info for resuming)
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{checkpoint_idx}.pt")
                torch.save({
                     'episode': episode,
                     'global_step': global_step,
                     'agent_state_dict': agent.state_dict(),
                     'target_net_state_dict': target_net.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     # Add other things if needed, e.g., replay_buffer.state_dict()
                }, checkpoint_path)
                print(f"Saved checkpoint at {checkpoint_path}")
                checkpoint_idx = (checkpoint_idx + 1) % 100 # Rotate checkpoints (keep last 100)

                # Advance round-robin index
                self.current_update_index = (self.current_update_index + 1) % len(opponent_ids)


            # --- Periodic Metrics Saving ---
            if episode % 100 == 0 or episode == self.num_episodes: # Save every 10k or at the end
                 metrics_file = os.path.join(self.checkpoint_dir, "training_metrics.csv")
                 is_new_file = not os.path.exists(metrics_file)
                 try:
                      with open(metrics_file, "a", newline="") as csvfile:
                           fieldnames = ['episode', 'reward', 'avg_reward', 'steps', 'epsilon'] # Added steps
                           writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                           if is_new_file:
                                writer.writeheader()
                           # Write metrics collected since last save (or all if first time)
                           writer.writerows(metrics_list)
                      print(f"Metrics saved to {metrics_file} at episode {episode}")
                      metrics_list = [] # Clear list after saving
                 except IOError as e:
                      print(f"Error saving metrics: {e}")


        # --- End of Training ---
        # Save final agent model separately
        final_checkpoint_path = os.path.join(self.checkpoint_dir, "final_agent_model.pt")
        torch.save(agent.state_dict(), final_checkpoint_path)
        print(f"Training complete. Final agent model saved at: {final_checkpoint_path}")
        # Ensure any remaining metrics are saved
        if metrics_list:
             metrics_file = os.path.join(self.checkpoint_dir, "training_metrics.csv")
             is_new_file = not os.path.exists(metrics_file)
             try:
                 with open(metrics_file, "a", newline="") as csvfile:
                     fieldnames = ['episode', 'reward', 'avg_reward', 'steps', 'epsilon']
                     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                     if is_new_file: writer.writeheader()
                     writer.writerows(metrics_list)
                 print(f"Final metrics saved to {metrics_file}")
             except IOError as e:
                 print(f"Error saving final metrics: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Poker RL Agent.")
    parser.add_argument("--episodes", type=int, default=1000000, help="Total number of training episodes.")
    parser.add_argument("--random", type=str, default=None, help="Episode range for using random policy for opponent 1 (format: 'start-end').")
    parser.add_argument("--variable", action="store_true", help="Enable variable training mode for opponent 1 (switch policy type every 1000 episodes).")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from.")

    args = parser.parse_args()

    trainer = Train(episodes=args.episodes, random_range=args.random, variable_mode=args.variable, resume_from=args.resume)
    trainer.run()