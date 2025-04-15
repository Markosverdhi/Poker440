# filename: package/train.py
"""
train.py

This script trains the Poker RL Agent using the BestPokerModel architecture
and the Gymnasium-compliant TrainFullPokerEnv (modified for tournament play).

MODIFIED (Tournament Episodes - Prompts 1 & 2):
- Adapted for tournament-based episodes and per-round rewards in replay buffer.

MODIFIED (Fix Model Instantiation TypeError):
- Removed the unexpected 'input_dim' keyword argument when instantiating
  BestPokerModel, consistent with the updated models.py.
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
    # Use the Gym-compliant version of the environment (MUST be updated for tournament logic)
    from envs import TrainFullPokerEnv
    # Use updated utils with new encoding and state dim
    from utils import encode_obs, epsilon_by_frame, ReplayBuffer, NEW_STATE_DIM
except ImportError:
    print("ERROR: Ensure envs.py and utils.py (with NEW_STATE_DIM) are available.")
    exit()

# Assuming models.py is available and updated for NEW_STATE_DIM
try:
    from models import BestPokerModel
except ImportError:
    print("ERROR: Ensure models.py is available.")
    exit()


# --- Global configuration ---
USE_HALF_ENCODING = False # Keep False for new encoding
NUM_PLAYERS = 6
# Use state dim defined in utils.py
STATE_DIM = NEW_STATE_DIM
# Action list/count will be derived from env instance

class Train:
    def __init__(self, episodes, random_range, variable_mode, resume_from=None):
        self.num_episodes = episodes # Number of tournaments to run
        self.random_range = self._parse_range(random_range) if random_range else None
        self.variable_mode = variable_mode
        self.resume_from = resume_from
        self.checkpoint_dir = "checkpoints"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.current_update_index = 0

        # Training hyperparameters (adjust if needed for tournaments)
        self.buffer_capacity = 10000
        self.batch_size = 64
        self.learning_rate = 1e-4
        self.gamma = 0.99 # Discount factor
        self.target_update_freq = 500 # Update target less frequently? (Adjust based on steps/tournament)
        self.opponent_update_freq = 50 # Tournaments between opponent updates
        self.checkpoint_save_freq = 50 # Tournaments between checkpoints
        self.metrics_save_freq = 100 # Tournaments between metric saves

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Environment instance needed later
        self.env = None

    def _parse_range(self, range_str):
        # (Unchanged)
        try:
            parts = range_str.split('-')
            if len(parts) == 2: return (int(parts[0]), int(parts[1]))
            else: raise ValueError("Range must have start and end separated by '-'.")
        except Exception as e: raise ValueError(f"Invalid range format '{range_str}'. Use 'start-end'. Error: {e}")

    def _in_range(self, episode, range_tuple):
        # (Unchanged)
        return range_tuple and range_tuple[0] <= episode <= range_tuple[1]

    # --- Opponent Policy Creation (Unchanged) ---
    def make_opponent_policy(self, opponent_model, action_list):
        num_actions = len(action_list)
        def policy_fn(obs_dict):
            if not isinstance(obs_dict, dict): return 'fold'
            legal_actions = obs_dict.get('legal_actions', [])
            if not legal_actions: return 'fold'
            try: state = encode_obs(obs_dict) # Use training encoder
            except Exception as e: print(f"Error encoding opponent obs: {e}. Folding."); return 'fold'
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            opponent_model.eval()
            with torch.no_grad(): q_values = opponent_model(state_tensor)
            q_values_np = q_values.squeeze().cpu().numpy(); sorted_indices = np.argsort(q_values_np)[::-1]
            for action_idx in sorted_indices:
                if 0 <= action_idx < num_actions:
                    action_str = action_list[action_idx]
                    if action_str in legal_actions: return action_str
            if 'check' in legal_actions: return 'check'
            if 'call' in legal_actions: return 'call'
            if 'fold' in legal_actions: return 'fold'
            return random.choice(legal_actions) if legal_actions else 'fold'
        return policy_fn

    # --- Random Policy (Unchanged) ---
    def random_policy(self, obs_dict):
        if not isinstance(obs_dict, dict): return 'fold'
        legal = obs_dict.get('legal_actions', [])
        if not legal: return 'fold'
        return random.choice(legal)

    # --- Opponent Update Logic ---
    def update_opponent_policy(self, opponent_id, policy_type, env, agent_action_list):
        num_actions = len(agent_action_list)
        if policy_type == "model":
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pt')]
            if not checkpoint_files:
                print(f"Warning: No checkpoints found for opponent {opponent_id}. Using random policy.")
                env.set_opponent_policy(opponent_id, self.random_policy); return
            try:
                checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
                latest_checkpoint_name = checkpoint_files[0]
            except (ValueError, IndexError):
                print("Warning: Could not sort checkpoints by number. Using alphabetically last."); checkpoint_files.sort(reverse=True)
                if not checkpoint_files: print(f"Error: No checkpoint files found after sort for opponent {opponent_id}. Using random."); env.set_opponent_policy(opponent_id, self.random_policy); return
                latest_checkpoint_name = checkpoint_files[0]
            full_checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint_name)
            try:
                checkpoint = torch.load(full_checkpoint_path, map_location=self.device)
                if isinstance(checkpoint, dict): opp_state_dict = checkpoint.get('agent_state_dict', checkpoint)
                else: opp_state_dict = checkpoint
                if not isinstance(opp_state_dict, dict): raise TypeError("Loaded checkpoint state is not a dict.")
                cleaned_state_dict = {k.replace('module.', ''): v for k, v in opp_state_dict.items()}

                # ** FIXED: Instantiate model without input_dim **
                opponent_model = BestPokerModel(num_actions=num_actions).to(self.device)
                opponent_model.load_state_dict(cleaned_state_dict, strict=False); opponent_model.eval()

                policy_fn = self.make_opponent_policy(opponent_model, agent_action_list)
                env.set_opponent_policy(opponent_id, policy_fn)
            except Exception as e:
                print(f"Error loading checkpoint {latest_checkpoint_name} for opponent {opponent_id}: {e}. Using random policy.")
                env.set_opponent_policy(opponent_id, self.random_policy)
        elif policy_type == "random":
            env.set_opponent_policy(opponent_id, self.random_policy)
        else:
            print(f"Warning: Unknown policy type '{policy_type}'. Using random.")
            env.set_opponent_policy(opponent_id, self.random_policy)


    def run(self):
        # --- Environment and Agent Setup ---
        try:
            self.env = TrainFullPokerEnv(num_players=NUM_PLAYERS, agent_id=0)
        except Exception as e: print(f"FATAL: Failed to initialize environment: {e}"); return

        agent_action_list = self.env.action_list
        num_actions = self.env.action_space.n
        action_to_string = {i: s for i, s in enumerate(agent_action_list)}
        string_to_action = {s: i for i, s in enumerate(agent_action_list)}

        # ** FIXED: Instantiate models without input_dim **
        agent = BestPokerModel(num_actions=num_actions).to(self.device)
        target_net = BestPokerModel(num_actions=num_actions).to(self.device)
        optimizer = optim.Adam(agent.parameters(), lr=self.learning_rate)
        replay_buffer = ReplayBuffer(capacity=self.buffer_capacity)

        start_episode = 1; global_step = 0; last_checkpoint_episode = 0
        episode_rewards = []; metrics_list = []

        # --- Resume from checkpoint ---
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
                 start_episode = checkpoint.get('episode', start_episode -1) + 1
                 global_step = checkpoint.get('global_step', global_step)
                 last_checkpoint_episode = checkpoint.get('episode', 0)
                 print(f"Successfully resumed from Tournament {start_episode}, Global Step {global_step}")
                 resumed_successfully = True
             except Exception as e:
                 print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                 print(f"!!! FAILED TO LOAD CHECKPOINT: {self.resume_from} !!! Error: {e}")
                 print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                 start_episode = 1; global_step = 0; last_checkpoint_episode = 0
                 # ** FIXED: Instantiate models without input_dim **
                 agent = BestPokerModel(num_actions=num_actions).to(self.device)
                 target_net = BestPokerModel(num_actions=num_actions).to(self.device)
                 optimizer = optim.Adam(agent.parameters(), lr=self.learning_rate)
                 replay_buffer = ReplayBuffer(capacity=self.buffer_capacity)
                 resumed_successfully = False
        if not resumed_successfully:
            target_net.load_state_dict(agent.state_dict())
            print("Starting training from scratch (or after checkpoint load failure).")

        # Initialize opponent policies
        print("Initializing opponent policies...")
        opponent_ids = list(range(1, NUM_PLAYERS))
        for opp_id in opponent_ids:
             self.update_opponent_policy(opp_id, "model", self.env, agent_action_list)

        # --- Training Loop ---
        print(f"\n--- Starting Training Loop (Max Tournaments: {self.num_episodes}) ---")
        for episode in range(start_episode, self.num_episodes + 1):
            try: state, info = self.env.reset()
            except Exception as e: print(f"FATAL: Error during env.reset() for T {episode}: {e}"); break
            if info.get("error"): print(f"Error starting T {episode}: {info['error']}"); continue

            done = False; tournament_reward = 0; tournament_agent_steps = 0

            # Inner loop: Runs for one tournament
            while not done:
                # --- Determine Action ---
                action_idx = -1; agent_took_action = False
                try: current_player = self.env.current_player_id; agent_id = self.env.agent_id
                except AttributeError as e: print(f"FATAL: Env missing attribute: {e}. Stopping T."); done = True; break

                if current_player == agent_id:
                    try: legal_actions_list = self.env.get_legal_actions_for_agent()
                    except Exception as e: print(f"Error getting legal actions: {e}"); legal_actions_list = ['fold']
                    if not legal_actions_list: action_idx = -1
                    else:
                        agent_took_action = True; epsilon = epsilon_by_frame(global_step)
                        if random.random() < epsilon:
                            action_str = random.choice(legal_actions_list); action_idx = string_to_action.get(action_str)
                            if action_idx is None: print(f"Warning: Random action '{action_str}' not mapped."); action_idx = 0
                        else:
                            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                            agent.eval();
                            with torch.no_grad(): q_values = agent(state_tensor)
                            agent.train(); q_values_np = q_values.squeeze().cpu().numpy(); sorted_indices = np.argsort(q_values_np)[::-1]
                            action_idx = -1
                            for idx in sorted_indices:
                                potential_action_str = action_to_string.get(idx)
                                if potential_action_str is not None and potential_action_str in legal_actions_list: action_idx = idx; break
                            if action_idx == -1:
                                 action_str = random.choice(legal_actions_list); action_idx = string_to_action.get(action_str)
                                 if action_idx is None: print(f"Warning: Fallback action '{action_str}' not mapped."); action_idx = 0
                else: action_idx = -1

                # --- Environment Step ---
                next_state, step_reward, terminated, truncated, info = None, 0.0, False, False, {}
                try:
                    next_state, step_reward, terminated, truncated, info = self.env.step(action_idx)
                    step_done = terminated or truncated
                except ValueError as e:
                    if "Cannot step in a round that is already over" in str(e): print(f"ERROR CAUGHT in T {episode}: {e}. Forcing T end."); done = True; continue
                    else: print(f"Unexpected ValueError during env.step() in T {episode}: {e}"); raise e
                except Exception as e: print(f"Unexpected Error during env.step() in T {episode}: {type(e).__name__}: {e}"); done = True; continue

                # --- Store Experience & Learn ---
                if agent_took_action:
                    if state is not None and next_state is not None and action_idx != -1:
                        is_round_over = info.get('round_over', False)
                        reward_to_store = info.get('round_reward', 0.0) if is_round_over else 0.0
                        replay_buffer.push(state, action_idx, reward_to_store, next_state, float(step_done))
                        tournament_agent_steps += 1; global_step += 1
                        if len(replay_buffer) >= self.batch_size:
                            states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(self.batch_size)
                            states_tensor = torch.tensor(states_b, dtype=torch.float32, device=self.device); actions_tensor = torch.tensor(actions_b, dtype=torch.long, device=self.device).unsqueeze(1)
                            rewards_tensor = torch.tensor(rewards_b, dtype=torch.float32, device=self.device).unsqueeze(1); next_states_tensor = torch.tensor(next_states_b, dtype=torch.float32, device=self.device)
                            dones_tensor = torch.tensor(dones_b, dtype=np.float32, device=self.device).unsqueeze(1)
                            agent.train(); q_values = agent(states_tensor).gather(1, actions_tensor)
                            with torch.no_grad():
                                best_next_actions = agent(next_states_tensor).argmax(dim=1, keepdim=True); target_net.eval(); next_q_values = target_net(next_states_tensor).gather(1, best_next_actions)
                            target = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor); loss = nn.MSELoss()(q_values, target)
                            optimizer.zero_grad(); loss.backward(); optimizer.step()
                        if global_step % self.target_update_freq == 0: target_net.load_state_dict(agent.state_dict())

                # --- Update State ---
                if next_state is not None: state = next_state
                else: done = True # End tournament if state becomes invalid
                if isinstance(step_reward, (int, float)): tournament_reward += step_reward
                done = step_done # Update tournament done flag
            # --- End of Tournament Loop ---

            # Log results for the completed tournament
            episode_rewards.append(tournament_reward); avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0; current_epsilon = epsilon_by_frame(global_step)
            metrics_list.append({'episode': episode, 'reward': tournament_reward, 'avg_reward': avg_reward, 'steps': tournament_agent_steps, 'epsilon': current_epsilon})
            if episode % 10 == 0 or episode == self.num_episodes: print(f"T: {episode}, AgentSteps: {tournament_agent_steps}, T-Reward: {tournament_reward:.2f}, Avg T-Rew: {avg_reward:.2f}, Eps: {current_epsilon:.4f}, GlobalStep: {global_step}")

            # --- Opponent Updates, Checkpointing, Metrics Saving ---
            if episode > 0:
                if episode % self.opponent_update_freq == 0:
                    update_opponent_id = opponent_ids[self.current_update_index]; policy_type = "model"; # ... (rest of opponent update logic unchanged) ...
                    if update_opponent_id == 1:
                         if self.variable_mode and episode % 1000 == 0: policy_type = random.choice(["model", "random"]); print(f"Variable mode: Switching Opponent 1 policy type randomly to '{policy_type}' at T {episode}")
                         elif self._in_range(episode, self.random_range): policy_type = "random"; print(f"Random range active: Setting Opponent 1 policy type to 'random' for T {episode}")
                    self.update_opponent_policy(update_opponent_id, policy_type, self.env, agent_action_list); self.current_update_index = (self.current_update_index + 1) % len(opponent_ids)
                if episode % self.checkpoint_save_freq == 0:
                     checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{episode}.pt");
                     try: save_dict = { 'episode': episode, 'global_step': global_step, 'agent_state_dict': agent.state_dict(), 'target_net_state_dict': target_net.state_dict(), 'optimizer_state_dict': optimizer.state_dict() }; torch.save(save_dict, checkpoint_path); print(f"--- Saved checkpoint to {checkpoint_path} (T {episode}) ---"); last_checkpoint_episode = episode
                     except Exception as e: print(f"Error saving checkpoint: {e}")
                if episode % self.metrics_save_freq == 0 or episode == self.num_episodes:
                     if metrics_list:
                         metrics_file = os.path.join(self.checkpoint_dir, "training_metrics.csv"); is_new_file = not os.path.exists(metrics_file)
                         try:
                              with open(metrics_file, "a", newline="") as csvfile: fieldnames = ['episode', 'reward', 'avg_reward', 'steps', 'epsilon']; writer = csv.DictWriter(csvfile, fieldnames=fieldnames);
                              if is_new_file: writer.writeheader(); writer.writerows(metrics_list); print(f"--- Metrics saved to {metrics_file} (Up to T {episode}) ---"); metrics_list = []
                         except IOError as e: print(f"Error saving metrics: {e}")
        # --- End of Training (Outer loop) ---

        # --- Final Saving ---
        final_checkpoint_path = os.path.join(self.checkpoint_dir, "final_agent_model.pt")
        try: final_save_dict = { 'episode': self.num_episodes, 'global_step': global_step, 'agent_state_dict': agent.state_dict(), 'target_net_state_dict': target_net.state_dict(), 'optimizer_state_dict': optimizer.state_dict() }; torch.save(final_save_dict, final_checkpoint_path); print(f"\n--- Training complete. Final agent model saved at: {final_checkpoint_path} ---")
        except Exception as e: print(f"Error saving final agent model: {e}")
        if metrics_list: # Save remaining metrics
             metrics_file = os.path.join(self.checkpoint_dir, "training_metrics.csv"); is_new_file = not os.path.exists(metrics_file)
             try:
                 with open(metrics_file, "a", newline="") as csvfile: fieldnames = ['episode', 'reward', 'avg_reward', 'steps', 'epsilon']; writer = csv.DictWriter(csvfile, fieldnames=fieldnames);
                 if is_new_file: writer.writeheader(); writer.writerows(metrics_list); print(f"--- Final metrics batch saved to {metrics_file} ---")
             except IOError as e: print(f"Error saving final metrics: {e}")
        if self.env:
             try: self.env.close(); print("Environment closed.")
             except Exception as e: print(f"Error closing environment: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Poker RL Agent (Tournament Episodes).")
    parser.add_argument("--episodes", type=int, default=10000, help="Total number of training tournaments (episodes).")
    parser.add_argument("--random", type=str, default=None, help="Tournament range for random policy for opponent 1 (format: start-end).")
    parser.add_argument("--variable", action="store_true", help="Enable variable training mode for opponent 1.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from.")
    args = parser.parse_args()
    trainer = Train(episodes=args.episodes, random_range=args.random, variable_mode=args.variable, resume_from=args.resume)
    trainer.run()
