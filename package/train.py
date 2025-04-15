"""
train.py

This script trains the Poker RL Agent using the BestPokerModel architecture
and the Gymnasium-compliant TrainFullPokerEnv.

MODIFIED (Gymnasium Adaptation - FIX 5):
- Modified main loop:
    - Checks if it's the agent's turn using env.current_player_id.
    - If not agent's turn, sends a dummy action (-1) to env.step() to advance opponent turns.
    - Skips experience storing and learning updates when a dummy action was sent.
- Kept previous fixes for legal action handling and action mapping.
"""

import os
import random
import csv
import argparse
from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import Gymnasium-compliant environment and updated utils
try:
    # Use the Gym-compliant version of the environment (v6 expected)
    from envs import TrainFullPokerEnv
    from utils import encode_obs, epsilon_by_frame, ReplayBuffer
except ImportError:
    print("ERROR: Ensure envs.py (v6) and utils.py (v1) are available.")
    exit()

# Assuming models.py is available
try:
    from models import BestPokerModel, convert_half_to_full_state_dict
except ImportError:
    print("ERROR: Ensure models.py is available.")
    exit()


# --- Global configuration ---
USE_HALF_ENCODING = False
NUM_PLAYERS = 6
STATE_DIM = 52 + 1 + (NUM_PLAYERS - 1) * 52  # 313
# Action list/count will be derived from env instance

class Train:
    def __init__(self, episodes, random_range, variable_mode, resume_from=None):
        self.num_episodes = episodes
        self.random_range = self._parse_range(random_range) if random_range else None
        self.variable_mode = variable_mode
        self.resume_from = resume_from
        self.checkpoint_dir = "checkpoints"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.current_update_index = 0

        # Training hyperparameters
        self.buffer_capacity = 10000
        self.batch_size = 64
        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.target_update_freq = 200
        self.opponent_update_freq = 50
        self.checkpoint_save_freq = 50
        self.metrics_save_freq = 100

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def _parse_range(self, range_str):
        # (Code unchanged)
        try: parts = range_str.split('-'); return (int(parts[0]), int(parts[1])) if len(parts) == 2 else ValueError("Range must have start and end.")
        except Exception as e: raise ValueError(f"Invalid range format '{range_str}'. Use 'start-end'. Error: {e}")

    def _in_range(self, episode, range_tuple):
        # (Code unchanged)
        return range_tuple and range_tuple[0] <= episode <= range_tuple[1]

    # --- Opponent Policy Creation (Unchanged from train_py_gym_v1) ---
    def make_opponent_policy(self, opponent_model, action_list):
        # (Code unchanged)
        num_actions = len(action_list); action_map = {idx: action for idx, action in enumerate(action_list)}
        def policy_fn(obs_dict):
            if not isinstance(obs_dict, dict): return 'fold'
            legal_actions = obs_dict.get('legal_actions', []);
            if not legal_actions: return 'fold'
            state = encode_obs(obs_dict, use_half_encoding=USE_HALF_ENCODING); state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0); opponent_model.eval()
            with torch.no_grad(): q_values = opponent_model(state_tensor)
            q_values_np = q_values.squeeze().cpu().numpy(); sorted_indices = np.argsort(q_values_np)[::-1]
            for action_idx in sorted_indices:
                if 0 <= action_idx < num_actions: action_str = action_list[action_idx];
                if action_str in legal_actions: return action_str
            if 'check' in legal_actions: return 'check'
            if 'call' in legal_actions: return 'call'
            if 'fold' in legal_actions: return 'fold'
            return random.choice(legal_actions)
        return policy_fn

    def random_policy(self, obs_dict):
        # (Code unchanged)
        if not isinstance(obs_dict, dict): return 'fold'
        legal = obs_dict.get('legal_actions', []);
        if not legal: return 'fold'
        return random.choice(legal)

    # --- Opponent Update Logic (Unchanged from train_py_gym_v1) ---
    def update_opponent_policy(self, opponent_id, policy_type, env, agent_action_list):
        # (Code unchanged)
        num_actions = len(agent_action_list)
        if policy_type == "model":
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pt')]
            if not checkpoint_files: print(f"Warning: No checkpoints found for opponent {opponent_id}. Using random policy."); env.set_opponent_policy(opponent_id, self.random_policy); return
            try: checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True); latest_checkpoint_name = checkpoint_files[0]
            except ValueError: print("Warning: Could not sort checkpoints by number. Using alphabetically last."); checkpoint_files.sort(reverse=True); latest_checkpoint_name = checkpoint_files[0]
            full_checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint_name); # print(f"Attempting to load opponent model from: {full_checkpoint_path}") # Less verbose
            try:
                checkpoint = torch.load(full_checkpoint_path, map_location=self.device); opp_state_dict = checkpoint.get('agent_state_dict', checkpoint)
                if not isinstance(opp_state_dict, dict): raise TypeError("Loaded checkpoint state is not a dict.")
                cleaned_state_dict = {k.replace('module.', ''): v for k, v in opp_state_dict.items()}
                opponent_model = BestPokerModel(input_dim=STATE_DIM, num_actions=num_actions).to(self.device); opponent_model.load_state_dict(cleaned_state_dict, strict=False); opponent_model.eval()
                policy_fn = self.make_opponent_policy(opponent_model, agent_action_list); env.set_opponent_policy(opponent_id, policy_fn); # print(f"Updated opponent {opponent_id} with model policy from {latest_checkpoint_name}.") # Less verbose
            except Exception as e: print(f"Error loading checkpoint {latest_checkpoint_name} for opponent {opponent_id}: {e}. Using random policy."); env.set_opponent_policy(opponent_id, self.random_policy)
        elif policy_type == "random": env.set_opponent_policy(opponent_id, self.random_policy); # print(f"Updated opponent {opponent_id} with random policy.") # Less verbose
        else: print(f"Warning: Unknown policy type '{policy_type}' for opponent update.")


    def run(self):
        # --- Environment and Agent Setup ---
        env = TrainFullPokerEnv(num_players=NUM_PLAYERS, agent_id=0)
        agent_action_list = env.action_list
        num_actions = env.action_space.n
        action_to_string = {i: s for i, s in enumerate(agent_action_list)}
        string_to_action = {s: i for i, s in enumerate(agent_action_list)}

        agent = BestPokerModel(input_dim=STATE_DIM, num_actions=num_actions).to(self.device)
        target_net = BestPokerModel(input_dim=STATE_DIM, num_actions=num_actions).to(self.device)
        optimizer = optim.Adam(agent.parameters(), lr=self.learning_rate)
        replay_buffer = ReplayBuffer(capacity=self.buffer_capacity)

        start_episode = 1; global_step = 0; last_checkpoint_episode = 0
        episode_rewards = []; metrics_list = []

        # --- Resume from checkpoint (Unchanged from train_py_gym_v1) ---
        resumed_successfully = False
        if self.resume_from and os.path.exists(self.resume_from):
             print(f"Attempting to resume training from checkpoint: {self.resume_from}")
             try:
                 checkpoint = torch.load(self.resume_from, map_location=self.device)
                 if not isinstance(checkpoint, dict): raise TypeError("Checkpoint file is not a dictionary.")
                 if 'agent_state_dict' in checkpoint: agent.load_state_dict(checkpoint['agent_state_dict'], strict=False)
                 if 'target_net_state_dict' in checkpoint: target_net.load_state_dict(checkpoint['target_net_state_dict'], strict=False)
                 else: target_net.load_state_dict(agent.state_dict())
                 if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                 start_episode = checkpoint.get('episode', start_episode -1) + 1; global_step = checkpoint.get('global_step', global_step); last_checkpoint_episode = checkpoint.get('episode', 0)
                 print(f"Successfully resumed from Episode {start_episode}, Global Step {global_step}"); resumed_successfully = True
             except Exception as e:
                 print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"); print(f"!!! FAILED TO LOAD CHECKPOINT: {self.resume_from} !!! Error: {e}"); print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                 start_episode = 1; global_step = 0; last_checkpoint_episode = 0; agent = BestPokerModel(input_dim=STATE_DIM, num_actions=num_actions).to(self.device); target_net.load_state_dict(agent.state_dict()); optimizer = optim.Adam(agent.parameters(), lr=self.learning_rate); replay_buffer = ReplayBuffer(capacity=self.buffer_capacity); resumed_successfully = False
        if not resumed_successfully: target_net.load_state_dict(agent.state_dict()); print("Starting training from scratch (or after checkpoint load failure).")

        # Initialize opponent policies
        print("Initializing opponent policies...")
        opponent_ids = list(range(1, NUM_PLAYERS))
        for opp_id in opponent_ids:
             self.update_opponent_policy(opp_id, "model", env, agent_action_list)

        # --- Training Loop ---
        print(f"\n--- Starting Training Loop (Max Episodes: {self.num_episodes}) ---")
        for episode in range(start_episode, self.num_episodes + 1):
            state, info = env.reset()
            if info.get("error"): print(f"Error starting episode {episode}: {info['error']}"); continue

            done = False
            episode_reward = 0
            agent_steps_this_episode = 0 # Track agent actions *taken*

            # --- Episode Loop ---
            while not done:
                # --- Determine Action ---
                action_idx = -1 # Default to dummy action
                agent_took_action = False # Flag to track if agent acted this iteration

                # Check if it's the agent's turn according to the environment state
                if env.current_player_id == env.agent_id:
                    legal_actions_list = env.get_legal_actions_for_agent()
                    if not legal_actions_list:
                        # Agent is current player but has no actions (e.g., all-in)
                        print(f"Info: Agent {env.agent_id} is current player but has no legal actions. Sending dummy action -1.")
                        action_idx = -1 # Send dummy action to advance state
                    else:
                        # It's agent's turn and they have actions
                        agent_took_action = True # Agent will take a real action
                        epsilon = epsilon_by_frame(global_step)
                        if random.random() < epsilon:
                            action_str = random.choice(legal_actions_list)
                            action_idx = string_to_action.get(action_str, 0)
                        else:
                            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                            agent.eval();
                            with torch.no_grad(): q_values = agent(state_tensor)
                            agent.train();
                            q_values_np = q_values.squeeze().cpu().numpy()
                            sorted_indices = np.argsort(q_values_np)[::-1]
                            action_idx = -1
                            for idx in sorted_indices:
                                potential_action_str = action_to_string.get(idx)
                                if potential_action_str is not None and potential_action_str in legal_actions_list:
                                    action_idx = idx
                                    break
                            if action_idx == -1:
                                 action_str = random.choice(legal_actions_list)
                                 action_idx = string_to_action.get(action_str, 0)
                else:
                    # Not agent's turn, send dummy action to advance env state
                    action_idx = -1

                # --- Environment Step ---
                # Pass the chosen action index (real or dummy -1)
                # env.step handles internal opponent moves
                next_state, reward, terminated, truncated, info = env.step(action_idx)
                step_done = terminated or truncated

                # --- Store Experience & Learn (Only if Agent Acted) ---
                if agent_took_action:
                    # Store experience using state before agent acted, the action taken,
                    # and the resulting next_state, reward, done flag
                    replay_buffer.push(state, action_idx, reward, next_state, float(step_done))
                    agent_steps_this_episode += 1 # Increment agent steps *only* when agent acts
                    global_step += 1 # Increment global step *only* when agent acts

                    # --- Learning Step ---
                    if len(replay_buffer) >= self.batch_size:
                        # (Learning logic unchanged)
                        states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(self.batch_size)
                        states_tensor = torch.tensor(states_b, dtype=torch.float32, device=self.device)
                        actions_tensor = torch.tensor(actions_b, dtype=torch.long, device=self.device).unsqueeze(1)
                        rewards_tensor = torch.tensor(rewards_b, dtype=torch.float32, device=self.device).unsqueeze(1)
                        next_states_tensor = torch.tensor(next_states_b, dtype=torch.float32, device=self.device)
                        dones_tensor = torch.tensor(dones_b, dtype=np.float32, device=self.device).unsqueeze(1)
                        agent.train()
                        q_values = agent(states_tensor).gather(1, actions_tensor)
                        with torch.no_grad():
                            best_next_actions = agent(next_states_tensor).argmax(dim=1, keepdim=True)
                            target_net.eval()
                            next_q_values = target_net(next_states_tensor).gather(1, best_next_actions)
                        target = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)
                        loss = nn.MSELoss()(q_values, target)
                        optimizer.zero_grad(); loss.backward(); optimizer.step()

                    # --- Target Network Update ---
                    if global_step % self.target_update_freq == 0:
                        target_net.load_state_dict(agent.state_dict())

                # --- Update State for next loop iteration ---
                state = next_state
                # Accumulate reward regardless of who acted (reward is sparse anyway)
                episode_reward += reward

                # --- Update done flag for the next iteration's while check ---
                done = step_done
                # No explicit break needed here, while condition handles it
            # --- End of Episode Loop ---

            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:])
            current_epsilon = epsilon_by_frame(global_step)
            metrics_list.append({'episode': episode, 'reward': episode_reward, 'avg_reward': avg_reward, 'steps': agent_steps_this_episode, 'epsilon': current_epsilon})
            if episode % 10 == 0 or episode == self.num_episodes:
                print(f"Ep: {episode}, AgentSteps: {agent_steps_this_episode}, Reward: {episode_reward:.2f}, AvgRew: {avg_reward:.2f}, Eps: {current_epsilon:.4f}, GlobalStep: {global_step}")

            # --- Opponent Updates, Checkpointing, Metrics Saving ---
            if episode > 0:
                # (Opponent update logic unchanged)
                if episode % self.opponent_update_freq == 0:
                    update_opponent_id = opponent_ids[self.current_update_index]; policy_type = "model"
                    if update_opponent_id == 1:
                         if self.variable_mode and episode % 1000 == 0: policy_type = random.choice(["model", "random"]); print(f"Variable mode: Switching Opponent 1 policy type randomly to '{policy_type}' at episode {episode}")
                         elif self._in_range(episode, self.random_range): policy_type = "random"; print(f"Random range active: Setting Opponent 1 policy type to 'random' for episode {episode}")
                    # print(f"--- Updating Opponent {update_opponent_id} Policy to '{policy_type}' (Episode {episode}) ---"); # Less verbose
                    self.update_opponent_policy(update_opponent_id, policy_type, env, agent_action_list); self.current_update_index = (self.current_update_index + 1) % len(opponent_ids)

                # (Checkpointing logic unchanged)
                if episode % self.checkpoint_save_freq == 0:
                     checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{episode}.pt")
                     try: save_dict = {'episode': episode, 'global_step': global_step, 'agent_state_dict': agent.state_dict(), 'target_net_state_dict': target_net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}; torch.save(save_dict, checkpoint_path); print(f"--- Saved checkpoint at {checkpoint_path} (Episode {episode}) ---"); last_checkpoint_episode = episode
                     except Exception as e: print(f"Error saving checkpoint: {e}")

                # (Metrics saving logic unchanged)
                if episode % self.metrics_save_freq == 0 or episode == self.num_episodes:
                     if metrics_list:
                         metrics_file = os.path.join(self.checkpoint_dir, "training_metrics.csv"); is_new_file = not os.path.exists(metrics_file)
                         try:
                              with open(metrics_file, "a", newline="") as csvfile: fieldnames = ['episode', 'reward', 'avg_reward', 'steps', 'epsilon']; writer = csv.DictWriter(csvfile, fieldnames=fieldnames);
                              if is_new_file: writer.writeheader(); writer.writerows(metrics_list); print(f"--- Metrics saved to {metrics_file} at episode {episode} ---"); metrics_list = []
                         except IOError as e: print(f"Error saving metrics: {e}")

        # --- End of Training ---
        # (Saving logic unchanged)
        final_checkpoint_path = os.path.join(self.checkpoint_dir, "final_agent_model.pt")
        try: torch.save(agent.state_dict(), final_checkpoint_path); print(f"\n--- Training complete. Final agent model saved at: {final_checkpoint_path} ---")
        except Exception as e: print(f"Error saving final agent model: {e}")
        if metrics_list:
             metrics_file = os.path.join(self.checkpoint_dir, "training_metrics.csv"); is_new_file = not os.path.exists(metrics_file)
             try:
                 with open(metrics_file, "a", newline="") as csvfile: fieldnames = ['episode', 'reward', 'avg_reward', 'steps', 'epsilon']; writer = csv.DictWriter(csvfile, fieldnames=fieldnames);
                 if is_new_file: writer.writeheader(); writer.writerows(metrics_list); print(f"--- Final metrics saved to {metrics_file} ---")
             except IOError as e: print(f"Error saving final metrics: {e}")
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Poker RL Agent (Gymnasium Adapted).")
    parser.add_argument("--episodes", type=int, default=100000, help="Total number of training episodes.")
    parser.add_argument("--random", type=str, default=None, help="Episode range for random policy for opponent 1.")
    parser.add_argument("--variable", action="store_true", help="Enable variable training mode for opponent 1.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from.")
    args = parser.parse_args()
    trainer = Train(episodes=args.episodes, random_range=args.random, variable_mode=args.variable, resume_from=args.resume)
    trainer.run()
