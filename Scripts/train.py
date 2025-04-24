# filename: train.py
import os
import random
import csv
import argparse
import time # Import time at the top
from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Assume these imports work correctly and point to your environment and model files
try:
    # Ensure relative imports work if train.py is outside Back_End
    # If train.py is inside Back_End, use: from .envs import TrainFullPokerEnv etc.
    from Back_End.envs import TrainFullPokerEnv
    from Back_End.utils import encode_obs, epsilon_by_frame, ReplayBuffer # Assuming NEW_STATE_DIM is handled by encode_obs
    from Back_End.models import BestPokerModel
except ImportError as e:
    print(f"ERROR: Could not import required modules: {e}")
    print("Ensure envs.py, utils.py, and models.py are accessible from your current directory.")
    print("If train.py is not in the parent directory of Back_End, adjust the import paths.")
    exit()


# --- Constants ---
# NUM_PLAYERS should ideally match the environment's default or be configurable
NUM_PLAYERS = 6
# *** ADDED: Threshold for skipping slow episodes ***
MAX_EPISODE_DURATION_SEC = 30.0


class Train:
    def __init__(self, episodes, resume_from=None):
        self.num_episodes = episodes
        self.resume_from = resume_from
        self.checkpoint_dir = "checkpoints"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.current_update_index = 0 # For rotating opponent models

        # --- Hyperparameters (can be overridden by args) ---
        self.buffer_capacity = 10000
        self.batch_size = 64
        self.learning_rate = 1e-4
        self.gamma = 0.99 # Discount factor
        self.target_update_freq = 50 # Steps between updating target network
        self.opponent_update_freq = 15 # Episodes between updating one opponent model
        self.checkpoint_save_freq = 200 # Episodes between saving checkpoints
        self.metrics_save_freq = 10 # Episodes between saving metrics
        self.max_steps_per_tournament = 500 # Safety break for episodes

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.env = None # Initialized in run()

    def make_opponent_policy(self, opponent_model, action_list):
        """Creates a policy function for an opponent using a trained model."""
        num_actions = len(action_list)
        string_to_action_map = {s: i for i, s in enumerate(action_list)}

        def policy_fn(obs_dict):
            # Basic validation of observation
            if not isinstance(obs_dict, dict):
                return 'fold'

            legal_actions = obs_dict.get('legal_actions', [])
            if not legal_actions:
                return 'fold' # Should ideally not happen if game logic is correct

            # Attempt to encode observation
            try:
                # Ensure encode_obs handles the opponent's observation format correctly
                state = encode_obs(obs_dict) # Relies on utils.encode_obs
            except Exception as e:
                # print(f"Error encoding opponent obs: {e}. Folding.") # Less verbose
                return 'fold'

            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Get Q-values from the opponent model
            opponent_model.eval() # Ensure model is in evaluation mode
            with torch.no_grad():
                q_values = opponent_model(state_tensor)

            q_values_np = q_values.squeeze().cpu().numpy()
            sorted_indices = np.argsort(q_values_np)[::-1] # Best actions first

            # Choose the best action that is legal
            for action_idx in sorted_indices:
                if 0 <= action_idx < num_actions:
                    action_str = action_list[action_idx]
                    if action_str in legal_actions:
                        return action_str # Return the best valid action

            # Fallback if no predicted action is legal
            if 'check' in legal_actions: return 'check'
            if 'call' in legal_actions: return 'call'
            if 'fold' in legal_actions: return 'fold'

            # Absolute fallback: choose randomly from legal actions
            return random.choice(legal_actions)

        return policy_fn

    def random_policy(self, obs_dict):
        """A simple policy that chooses a random legal action, prioritizing safe actions."""
        if not isinstance(obs_dict, dict): return 'fold'
        legal = obs_dict.get('legal_actions', [])
        if not legal: return 'fold'
        # Prioritize check/call/fold over random raise/bet if available
        if 'check' in legal: return 'check'
        if 'call' in legal: return 'call'
        if 'fold' in legal: return 'fold'
        return random.choice(legal)

    def update_opponent_policy(self, opponent_id, policy_type, env, agent_action_list):
        """Sets the policy for a specific opponent based on latest checkpoint or random."""
        num_actions = len(agent_action_list)
        if policy_type == "model":
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pt')]
            if not checkpoint_files:
                print(f"Warning: No checkpoints found for opponent {opponent_id}. Using random policy.")
                env.set_opponent_policy(opponent_id, self.random_policy)
                return

            try:
                # Sort by episode number (integer part of the filename)
                checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
                latest_checkpoint_name = checkpoint_files[0] # Use the most recent one
            except (ValueError, IndexError):
                # Fallback if parsing filename fails
                print("Warning: Could not sort checkpoints by number. Using alphabetically last.")
                checkpoint_files.sort(reverse=True)
                if not checkpoint_files:
                     print(f"Error: No checkpoint files found after sort for opponent {opponent_id}. Using random.")
                     env.set_opponent_policy(opponent_id, self.random_policy)
                     return
                latest_checkpoint_name = checkpoint_files[0]

            full_checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint_name)
            # print(f"Attempting to load opponent {opponent_id} policy from: {latest_checkpoint_name}") # Less verbose
            try:
                checkpoint = torch.load(full_checkpoint_path, map_location=self.device)

                # Handle both dictionary and raw state_dict checkpoints
                if isinstance(checkpoint, dict):
                    opp_state_dict = checkpoint.get('agent_state_dict', checkpoint) # Prefer 'agent_state_dict' if available
                else:
                    opp_state_dict = checkpoint # Assume it's just the state_dict

                if not isinstance(opp_state_dict, dict):
                    raise TypeError(f"Loaded checkpoint state for opponent {opponent_id} is not a dictionary.")

                # Remove 'module.' prefix if saved with DataParallel
                cleaned_state_dict = {k.replace('module.', ''): v for k, v in opp_state_dict.items()}

                # Create and load the opponent model
                opponent_model = BestPokerModel(num_actions=num_actions).to(self.device)
                # Use strict=False initially to handle potential architecture mismatches during development
                load_info = opponent_model.load_state_dict(cleaned_state_dict, strict=False)
                if load_info.missing_keys or load_info.unexpected_keys:
                    print(f"Warning loading opponent {opponent_id}: Missing={load_info.missing_keys}, Unexpected={load_info.unexpected_keys}")

                opponent_model.eval() # Set to evaluation mode

                # Create and set the policy function
                policy_fn = self.make_opponent_policy(opponent_model, agent_action_list)
                env.set_opponent_policy(opponent_id, policy_fn)
                # print(f"Successfully loaded model policy for opponent {opponent_id} from {latest_checkpoint_name}") # Less verbose

            except Exception as e:
                print(f"ERROR loading checkpoint {latest_checkpoint_name} for opponent {opponent_id}: {e}. Using random policy.")
                env.set_opponent_policy(opponent_id, self.random_policy)

        elif policy_type == "random":
            # print(f"Setting opponent {opponent_id} to random policy.") # Less verbose
            env.set_opponent_policy(opponent_id, self.random_policy)
        else:
            print(f"Warning: Unknown policy type '{policy_type}' for opponent {opponent_id}. Using random.")
            env.set_opponent_policy(opponent_id, self.random_policy)

    def update_one_opponent_from_checkpoint(self, opponent_id, checkpoint_path, action_list):
        """Loads a specific checkpoint for a single opponent."""
        print(f"Updating opponent {opponent_id} policy from specific checkpoint: {os.path.basename(checkpoint_path)}")
        num_actions = len(action_list)
        try:
            ck = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ck.get('agent_state_dict', ck) if isinstance(ck, dict) else ck
            if not isinstance(state_dict, dict):
                 raise TypeError("Loaded checkpoint state is not a dict.")

            cleaned = {k.replace('module.', ''): v for k, v in state_dict.items()}

            model = BestPokerModel(num_actions=num_actions).to(self.device)
            load_info = model.load_state_dict(cleaned, strict=False) # Use strict=False for flexibility
            if load_info.missing_keys or load_info.unexpected_keys:
                 print(f"  Load info for opponent {opponent_id}: {load_info}")
            model.eval()

            policy_fn = self.make_opponent_policy(model, action_list)
            self.env.set_opponent_policy(opponent_id, policy_fn)
            print(f"  Successfully updated opponent {opponent_id}.")
        except Exception as e:
            print(f"  ERROR loading specific checkpoint {os.path.basename(checkpoint_path)} for opponent {opponent_id}: {e}. Keeping previous policy.")
            # Optionally set to random here, or just let it keep its current policy
            # self.env.set_opponent_policy(opponent_id, self.random_policy)

    # *** ADDED: Helper method to trigger opponent update ***
    def _trigger_opponent_update(self, episode, opponent_ids, agent_action_list):
        """Selects an opponent and updates their policy from the latest checkpoint."""
        if not opponent_ids: return # No opponents to update

        # Choose an opponent seat to update (round-robin)
        seat_to_update = opponent_ids[self.current_update_index % len(opponent_ids)]
        print(f"\n--- Updating opponent policy at seat {seat_to_update} (Triggered at Episode {episode}) ---")

        # Find available checkpoints
        chk_files = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith("checkpoint_") and f.endswith(".pt")
        ]

        if chk_files:
            # Option: Choose the latest checkpoint
            try:
                chk_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
                chosen_chk_file = chk_files[0]
            except: # Fallback to alphabetic sort if number parsing fails
                 chk_files.sort(reverse=True)
                 chosen_chk_file = chk_files[0] if chk_files else None

            if chosen_chk_file:
                full_path = os.path.join(self.checkpoint_dir, chosen_chk_file)
                self.update_one_opponent_from_checkpoint(
                    seat_to_update, full_path, agent_action_list
                )
            else:
                 print("No suitable checkpoint found; skipped opponent rotation.")

        else:
            print("No checkpoints found; skipped opponent rotation.")

        self.current_update_index += 1 # Move to the next opponent for the next update


    def run(self):
        """Main training loop."""
        try:
            # Initialize environment - ensure agent_id=0 is correct
            # Pass seat_config if you want specific opponent types, otherwise defaults to 'model'
            self.env = TrainFullPokerEnv(num_players=NUM_PLAYERS, agent_id=0)
            print("Poker environment initialized.")
        except Exception as e:
            print(f"FATAL: Failed to initialize environment: {e}")
            return # Cannot proceed without environment

        # Get action space details from the environment
        agent_action_list = self.env.action_list # List of action strings
        num_actions = self.env.action_space.n # Number of actions
        action_to_string = {i: s for i, s in enumerate(agent_action_list)}
        string_to_action = {s: i for i, s in enumerate(agent_action_list)}
        print(f"Agent Action Space ({num_actions} actions): {agent_action_list}")

        # Initialize Agent and Target Network
        agent = BestPokerModel(num_actions=num_actions).to(self.device)
        target_net = BestPokerModel(num_actions=num_actions).to(self.device)
        print("Agent and Target networks initialized.")

        # Initialize Optimizer and Replay Buffer
        optimizer = optim.Adam(agent.parameters(), lr=self.learning_rate)
        replay_buffer = ReplayBuffer(capacity=self.buffer_capacity)
        print(f"Optimizer (Adam, lr={self.learning_rate}) and Replay Buffer (cap={self.buffer_capacity}) initialized.")

        # --- State Initialization ---
        start_episode = 1
        global_step = 0
        episode_rewards_deque = deque(maxlen=100) # Store last 100 episode rewards for avg calculation
        metrics_list = [] # Temporary storage for metrics before saving
        loss_history = deque(maxlen=100) # Track recent losses

        # --- Resume Logic ---
        resumed_successfully = False
        if self.resume_from and os.path.exists(self.resume_from):
            print(f"Attempting to resume training from checkpoint: {self.resume_from}")
            try:
                checkpoint = torch.load(self.resume_from, map_location=self.device)
                if not isinstance(checkpoint, dict):
                    raise TypeError("Checkpoint file is not a dictionary.")

                # Load Agent state
                if 'agent_state_dict' in checkpoint:
                    load_info_agent = agent.load_state_dict(checkpoint['agent_state_dict'], strict=True) # Be strict on resume
                    print(f"  Agent load info: {load_info_agent}")
                else:
                     print("Warning: 'agent_state_dict' not found in checkpoint.")

                # Load Target Net state (or copy from agent if missing)
                if 'target_net_state_dict' in checkpoint:
                    load_info_target = target_net.load_state_dict(checkpoint['target_net_state_dict'], strict=True)
                    print(f"  Target Net load info: {load_info_target}")
                else:
                    print("Warning: 'target_net_state_dict' not found. Copying from loaded agent state.")
                    target_net.load_state_dict(agent.state_dict()) # Initialize target from agent

                # Load Optimizer state
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("  Optimizer state loaded.")
                     # Ensure optimizer state is moved to the correct device if necessary
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
                else:
                     print("Warning: 'optimizer_state_dict' not found in checkpoint.")

                # Load training progress
                start_episode = checkpoint.get('episode', start_episode - 1) + 1 # Resume from next episode
                global_step = checkpoint.get('global_step', global_step)
                # Load opponent update index if saved
                self.current_update_index = checkpoint.get('current_update_index', self.current_update_index)

                print(f"Successfully resumed from Tournament {start_episode}, Global Step {global_step}")
                resumed_successfully = True

            except Exception as e:
                print(f"!!! FAILED TO LOAD CHECKPOINT: {e} !!!")
                print("Starting from scratch.")
                start_episode = 1
                global_step = 0
                # Ensure target net is initialized from agent if resume fails
                target_net.load_state_dict(agent.state_dict())

        if not resumed_successfully:
            # Initialize target network from agent if not resuming or if resume failed
            target_net.load_state_dict(agent.state_dict())
            print("Starting training from scratch (or after checkpoint load failure). Target network initialized.")

        # --- Initialize Opponent Policies ---
        opponent_ids = list(range(1, NUM_PLAYERS)) # IDs 1 to NUM_PLAYERS-1
        print(f"Initializing policies for opponents: {opponent_ids}")
        for opp_id in opponent_ids:
            # Initialize opponents with the latest available checkpoint model or random
            self.update_opponent_policy(opp_id, "model", self.env, agent_action_list)

        print(f"\n--- Starting Training Loop (Episode {start_episode} to {self.num_episodes}) ---")

        # --- Main Training Loop ---
        for episode in range(start_episode, self.num_episodes + 1):
            episode_start_time = time.time()
            episode_reward_cumulative = 0.0 # Track cumulative reward *within* this episode
            episode_round_rewards = []
            num_rounds_in_episode = 0

            try:
                # Reset environment for a new tournament (episode)
                state, info = self.env.reset()
                if not isinstance(state, np.ndarray):
                     print(f"Warning: env.reset() did not return a numpy array state for episode {episode}. Type: {type(state)}. Attempting to continue.")

            except Exception as e:
                print(f"ERROR: Failed env.reset() for Tournament {episode}: {e}. Skipping tournament.")
                time.sleep(1) # Add a small delay if resets fail often
                continue # Skip to the next episode

            # Check for errors reported by the environment during reset
            if info and info.get("error"):
                print(f"Error reported by env.reset() for Tournament {episode}: {info['error']}. Skipping tournament.")
                continue

            terminated = False
            truncated = False # Gym standard flags
            tournament_agent_steps = 0
            tournament_total_steps = 0 # Includes opponent steps processed by env

            # --- Inner Loop (Steps within a tournament/episode) ---
            # Continues until the tournament (episode) is terminated or truncated
            while not terminated and not truncated:
                # Safety break for excessively long tournaments
                if tournament_total_steps >= self.max_steps_per_tournament:
                    print(f"WARNING: Tournament {episode} exceeded max steps ({self.max_steps_per_tournament}). Truncating episode.")
                    truncated = True # Use truncated flag
                    break # Exit the inner while loop

                # Determine whose turn it is (assuming env updates this)
                current_player = self.env.current_player_id

                try:
                    # --- Agent's Turn ---
                    if current_player == self.env.agent_id:
                        # Ensure state is valid
                        if not isinstance(state, np.ndarray):
                             print(f"ERROR: Agent's turn in T {episode}, Step {tournament_total_steps}, but state is not a numpy array ({type(state)}). Terminating episode.")
                             terminated = True # Terminate on critical error
                             break

                        # Get legal actions for the agent
                        legal = self.env.get_legal_actions_for_agent()
                        if not legal:
                            # This might happen if agent is already all-in but somehow gets turn
                            print(f"Warning: Agent {self.env.agent_id} has no legal actions in T {episode}, Step {tournament_total_steps}. Env issue? Forcing check/fold.")
                            if 'check' in self.env._get_legal_actions(current_player):
                                action_idx = string_to_action.get('check', 0)
                            else:
                                action_idx = string_to_action.get('fold', 0)

                        else:
                            # Epsilon-Greedy Action Selection
                            epsilon = epsilon_by_frame(global_step)
                            if random.random() < epsilon:
                                action_str = random.choice(legal)
                                action_idx = string_to_action.get(action_str, 0)
                            else:
                                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                                agent.eval()
                                with torch.no_grad():
                                    q_vals = agent(state_tensor).squeeze().cpu().numpy()
                                agent.train()
                                best_legal_action_idx = -1
                                best_q_val = -float('inf')
                                for act_str in legal:
                                    idx = string_to_action.get(act_str)
                                    if idx is not None and idx < len(q_vals):
                                        if q_vals[idx] > best_q_val:
                                            best_q_val = q_vals[idx]
                                            best_legal_action_idx = idx
                                    else:
                                        print(f"Warning: Action string '{act_str}' or index {idx} invalid for Q-values.")

                                if best_legal_action_idx != -1:
                                    action_idx = best_legal_action_idx
                                else:
                                    print(f"Warning: Could not find best legal action via Q-values for agent in T {episode}, Step {tournament_total_steps}. Choosing random from {legal}.")
                                    action_str = random.choice(legal)
                                    action_idx = string_to_action.get(action_str, 0)

                        # --- Environment Step (Agent) ---
                        next_state, step_reward, terminated, truncated, info = self.env.step(action_idx)

                        # --- Store Experience in Replay Buffer ---
                        if isinstance(state, np.ndarray) and isinstance(next_state, np.ndarray):
                             replay_buffer.push(state, action_idx, step_reward, next_state, float(terminated or truncated))
                        else:
                             print(f"Warning: Invalid state ({type(state)}) or next_state ({type(next_state)}) type during agent step {tournament_total_steps}. Skipping buffer push.")

                        # Accumulate reward for the episode's total
                        episode_reward_cumulative += step_reward

                        # --- DQN Learning Update ---
                        if len(replay_buffer) >= self.batch_size:
                            (states_b, actions_b, rewards_b,
                             next_states_b, dones_b) = replay_buffer.sample(self.batch_size)
                            states_t = torch.tensor(states_b, dtype=torch.float32, device=self.device)
                            actions_t = torch.tensor(actions_b, dtype=torch.long, device=self.device).unsqueeze(1)
                            rewards_t = torch.tensor(rewards_b, dtype=torch.float32, device=self.device).unsqueeze(1)
                            next_states_t = torch.tensor(next_states_b, dtype=torch.float32, device=self.device)
                            dones_t = torch.tensor(dones_b, dtype=torch.float32, device=self.device).unsqueeze(1)

                            with torch.no_grad():
                                best_next_actions = agent(next_states_t).argmax(dim=1, keepdim=True)
                                next_q_values_target = target_net(next_states_t).gather(1, best_next_actions)
                                target_q_values = rewards_t + self.gamma * next_q_values_target * (1 - dones_t)

                            current_q_values = agent(states_t).gather(1, actions_t)
                            loss = nn.MSELoss()(current_q_values, target_q_values)
                            loss_history.append(loss.item())

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            if global_step % self.target_update_freq == 0:
                                target_net.load_state_dict(agent.state_dict())

                        # --- Update State and Stats for Agent's Step ---
                        state = next_state
                        tournament_agent_steps += 1
                        global_step += 1

                    # --- Opponent's Turn ---
                    else:
                        next_state_after_opponents, step_reward, terminated, truncated, info = self.env.step(-1)
                        state = next_state_after_opponents
                        episode_reward_cumulative += step_reward

                    # --- Check for round end and store round reward ---
                    if info.get('round_over', False):
                        num_rounds_in_episode += 1
                        round_reward = info.get('round_reward', 0.0)
                        episode_round_rewards.append(round_reward)

                    tournament_total_steps += 1

                except Exception as e:
                    print(f"ERROR during step processing in Tournament {episode}, Total Step {tournament_total_steps}, Current Player {current_player}: {e}")
                    import traceback
                    traceback.print_exc()
                    terminated = True
                    break # Exit the inner while loop

            # --- End of Tournament (Episode) ---
            episode_end_time = time.time()
            episode_duration = episode_end_time - episode_start_time

            # *** ADDED: Check for long duration and skip/update opponents ***
            if episode_duration > MAX_EPISODE_DURATION_SEC:
                print(f"WARNING: Episode {episode} duration ({episode_duration:.2f}s) exceeded threshold ({MAX_EPISODE_DURATION_SEC:.1f}s).")
                print("         Forcing opponent update and skipping metrics/checkpoint save for this episode.")
                # Trigger opponent update immediately
                self._trigger_opponent_update(episode, opponent_ids, agent_action_list)
                # Skip the rest of the loop iteration (logging, saving)
                continue # Go to the next episode

            # --- Regular End-of-Episode Processing (if duration check passed) ---

            avg_round_reward = float(np.mean(episode_round_rewards)) if episode_round_rewards else 0.0
            episode_rewards_deque.append(episode_reward_cumulative)
            avg_reward = float(np.mean(episode_rewards_deque))
            current_epsilon = epsilon_by_frame(global_step)
            avg_loss = float(np.mean(loss_history)) if loss_history else 0.0

            metrics_list.append({
                'episode': episode,
                'reward': episode_reward_cumulative,
                'avg_reward': avg_reward,
                'agent_steps': tournament_agent_steps,
                'total_steps': tournament_total_steps,
                'epsilon': current_epsilon,
                'avg_loss': avg_loss,
                'duration_sec': episode_duration,
                'num_rounds': num_rounds_in_episode,
                'avg_round_reward': avg_round_reward
            })

            # --- Logging ---
            if episode % 10 == 0 or episode == self.num_episodes:
                print(f"T: {episode}, AgSteps: {tournament_agent_steps}, TotSteps: {tournament_total_steps}, "
                      f"Rounds: {num_rounds_in_episode}, AvgRndRew: {avg_round_reward:.2f}, "
                      f"EpReward: {episode_reward_cumulative:.2f}, AvgEpRew(100): {avg_reward:.2f}, "
                      f"AvgLoss(100): {avg_loss:.4f}, Eps: {current_epsilon:.4f}, "
                      f"GStep: {global_step}, Dur: {episode_duration:.1f}s")

            # --- Save Checkpoint ---
            if episode % self.checkpoint_save_freq == 0 or episode == self.num_episodes:
                chk = {
                    'agent_state_dict': agent.state_dict(),
                    'target_net_state_dict': target_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode': episode,
                    'global_step': global_step,
                    'current_update_index': self.current_update_index
                }
                path = os.path.join(self.checkpoint_dir, f'checkpoint_{episode}.pt')
                try:
                    torch.save(chk, path)
                    print(f"--- Checkpoint saved to {path} (Episode {episode}) ---")
                except Exception as e:
                    print(f"ERROR saving checkpoint to {path}: {e}")

            # --- Save Metrics ---
            if episode % self.metrics_save_freq == 0 or episode == self.num_episodes:
                metrics_file = os.path.join(self.checkpoint_dir, "training_metrics.csv")
                is_new_file = not os.path.exists(metrics_file)
                try:
                    with open(metrics_file, "a", newline="") as csvfile:
                        fieldnames = [
                            'episode', 'reward', 'avg_reward', 'agent_steps',
                            'total_steps', 'epsilon', 'avg_loss', 'duration_sec',
                            'num_rounds', 'avg_round_reward'
                        ]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        if is_new_file or os.path.getsize(metrics_file) == 0:
                            writer.writeheader()
                        writer.writerows(metrics_list)
                    metrics_list.clear()
                except Exception as e:
                    print(f"ERROR saving metrics to {metrics_file}: {e}")

            # --- Regular Opponent Policy Update ---
            # *** MODIFIED: Use the helper method ***
            if episode > start_episode and episode % self.opponent_update_freq == 0:
                self._trigger_opponent_update(episode, opponent_ids, agent_action_list)


        # --- End of Training ---
        print("\n--- Training Loop Finished ---")

        # --- Save Final Model ---
        final_path = os.path.join(self.checkpoint_dir, "final_agent_model.pt")
        final_dict = {
            'agent_state_dict': agent.state_dict(),
            'target_net_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode': self.num_episodes,
            'global_step': global_step,
            'action_list': agent_action_list,
            'current_update_index': self.current_update_index
        }
        try:
            torch.save(final_dict, final_path)
            print(f"--- Final model saved to {final_path} ---")
        except Exception as e:
            print(f"ERROR saving final model to {final_path}: {e}")

        # --- Save any remaining metrics ---
        if metrics_list:
            metrics_file = os.path.join(self.checkpoint_dir, "training_metrics.csv")
            try:
                with open(metrics_file, "a", newline="") as csvfile:
                     fieldnames = [
                         'episode', 'reward', 'avg_reward', 'agent_steps',
                         'total_steps', 'epsilon', 'avg_loss', 'duration_sec',
                         'num_rounds', 'avg_round_reward'
                     ]
                     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                     if not os.path.exists(metrics_file) or os.path.getsize(metrics_file) == 0:
                          writer.writeheader()
                     writer.writerows(metrics_list)
                print(f"--- Final metrics batch saved to {metrics_file} ---")
            except Exception as e:
                print(f"ERROR saving final metrics batch to {metrics_file}: {e}")

        # --- Close Environment ---
        try:
            if self.env:
                self.env.close()
                print("Environment closed.")
        except Exception as e:
            print(f"Error closing environment: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Poker RL Agent (Tournament Episodes).")
    parser.add_argument("--episodes", type=int, default=10000, help="Total number of training tournaments (episodes).")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DQN updates.")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Replay buffer capacity.")
    parser.add_argument("--target_update", type=int, default=50, help="Frequency (in agent steps) to update target network.")
    parser.add_argument("--opponent_update", type=int, default=15, help="Frequency (in episodes) to update one opponent model.")
    parser.add_argument("--save_freq", type=int, default=200, help="Frequency (in episodes) to save checkpoints.")
    parser.add_argument("--metrics_freq", type=int, default=10, help="Frequency (in episodes) to save metrics.")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per tournament episode before truncation.")


    args = parser.parse_args()

    # Pass relevant args to Train constructor/attributes
    trainer = Train(episodes=args.episodes,
                    resume_from=args.resume)

    # Override defaults with args
    trainer.learning_rate = args.lr
    trainer.batch_size = args.batch_size
    trainer.buffer_capacity = args.buffer_size
    trainer.target_update_freq = args.target_update
    trainer.opponent_update_freq = args.opponent_update
    trainer.checkpoint_save_freq = args.save_freq
    trainer.metrics_save_freq = args.metrics_freq
    trainer.max_steps_per_tournament = args.max_steps

    print("\n--- Training Configuration ---")
    print(f"Total Episodes: {args.episodes}")
    print(f"Resume Checkpoint: {args.resume}")
    print(f"Learning Rate: {trainer.learning_rate}")
    print(f"Batch Size: {trainer.batch_size}")
    print(f"Buffer Capacity: {trainer.buffer_capacity}")
    print(f"Target Update Freq (steps): {trainer.target_update_freq}")
    print(f"Opponent Update Freq (episodes): {trainer.opponent_update_freq}")
    print(f"Checkpoint Save Freq (episodes): {trainer.checkpoint_save_freq}")
    print(f"Metrics Save Freq (episodes): {trainer.metrics_save_freq}")
    print(f"Max Steps per Episode: {trainer.max_steps_per_tournament}")
    print(f"Device: {trainer.device}")
    print("----------------------------\n")

    trainer.run()
