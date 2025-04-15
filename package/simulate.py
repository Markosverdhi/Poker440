"""
simulate.py

Simulates games using a trained Poker RL Agent against configured opponents
using the Gymnasium-compliant BaseFullPokerEnv.

MODIFIED (Gymnasium Adaptation - FIX 2):
- Replaced internal env._get_obs() call with env.get_legal_actions_for_agent()
  for determining legal actions during agent's action selection.
- Corrected action masking for model-based actions (index to string check).
- Corrected checkpoint loading try/except block structure.
"""

import os
import argparse
import torch
import numpy as np
import random
import csv
import json

# Import Gymnasium-compliant environment and updated utils
try:
    # Use the Gym-compliant version of the environment (v5 expected)
    from envs import BaseFullPokerEnv
    from utils import encode_obs_eval, log_decision # utils_py_gym_v1 expected
except ImportError:
    print("ERROR: Ensure envs.py (v5) and utils.py (v1) are available.")
    exit()

# Assuming models.py is available
try:
    from models import BestPokerModel, convert_half_to_full_state_dict
except ImportError:
    print("ERROR: Ensure models.py is available.")
    exit()


# --- Global configuration (ensure consistency) ---
NUM_PLAYERS = 6
STATE_DIM = 52 + 1 + (NUM_PLAYERS - 1) * 52  # 313
try:
    # Create a temporary env just to get action list/space info
    temp_env = BaseFullPokerEnv()
    ACTION_LIST = temp_env.action_list[:] # Make a copy
    NUM_ACTIONS = temp_env.action_space.n
    # Reconstruct mappings needed locally
    action_to_string = {i: s for i, s in enumerate(ACTION_LIST)}
    string_to_action = {s: i for i, s in enumerate(ACTION_LIST)}
    temp_env.close() # Close the temporary env
except Exception as e:
    print(f"Warning: Could not get action list/space from env: {e}. Using default.")
    ACTION_LIST = ['fold', 'call', 'check', 'bet_small', 'bet_big', 'all_in']
    NUM_ACTIONS = len(ACTION_LIST)
    action_to_string = {i: s for i, s in enumerate(ACTION_LIST)}
    string_to_action = {s: i for i, s in enumerate(ACTION_LIST)}


def parse_args():
    # (Argument parsing remains the same)
    parser = argparse.ArgumentParser(description="Simulate/Evaluate the trained Poker RL Agent (Gymnasium Adapted).")
    parser.add_argument("--checkpoint", type=str, default="", required=True, help="Path to the trained model checkpoint (.pt) REQUIRED for simulation.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of simulation episodes to run.")
    parser.add_argument("--opponent", type=str, default="model", choices=["model", "random", "variable"], help="Default type of opponent to use if --seat_config is not provided.")
    parser.add_argument("--output_csv", type=str, default="simulation_results.csv", help="Path to the CSV file to store simulation summary results.")
    parser.add_argument("--seat_config", type=str, default="", help="Comma-separated list for each seat (0 to NUM_PLAYERS-1). Seat 0 must be 'agent'; others can be 'model', 'random', or 'variable'.")
    parser.add_argument("--detailed_log", type=str, default="detailed_simulation_log.csv", help="Path to the CSV file to store detailed game state and action logs.")
    return parser.parse_args()

# --- Opponent Policy Creation (Unchanged from simulate_py_gym_v3) ---
def get_opponent_policy(opponent_type, agent_model):
    # (Code remains the same)
    if opponent_type == "model":
        if not agent_model: return get_opponent_policy("random", None)
        def policy_fn(obs_dict):
            if not isinstance(obs_dict, dict): return 'fold'
            legal_actions = obs_dict.get('legal_actions', []);
            if not legal_actions: return 'fold'
            state = encode_obs_eval(obs_dict, use_half_encoding=False); state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad(): q_values = agent_model(state_tensor)
            q_values_np = q_values.squeeze().cpu().numpy(); sorted_indices = np.argsort(q_values_np)[::-1]
            for action_idx in sorted_indices:
                if 0 <= action_idx < NUM_ACTIONS: action_str = action_to_string.get(action_idx); # Use local map
                if action_str and action_str in legal_actions: return action_str
            if 'check' in legal_actions: return 'check'
            if 'call' in legal_actions: return 'call'
            if 'fold' in legal_actions: return 'fold'
            return random.choice(legal_actions)
        return policy_fn
    elif opponent_type == "random":
        def policy_fn(obs_dict):
            if not isinstance(obs_dict, dict): return 'fold'
            legal = obs_dict.get("legal_actions", []); return random.choice(legal) if legal else "fold"
        return policy_fn
    elif opponent_type == "variable":
        if not agent_model: return get_opponent_policy("random", None)
        model_policy = get_opponent_policy("model", agent_model); random_policy = get_opponent_policy("random", None)
        def policy_fn(obs_dict): return model_policy(obs_dict) if random.random() < 0.5 else random_policy(obs_dict)
        return policy_fn
    else: return get_opponent_policy("random", None)

# --- Simulation Episode (Adapted for Gym API & Enhanced Logging) ---
def simulate_episode(env: BaseFullPokerEnv, agent: torch.nn.Module, episode: int, detailed_writer) -> (float, dict):
    """
    Runs simulation episode, logs state, position, RFI status, raiser position, action.
    Uses env.get_legal_actions_for_agent().
    """
    state, info = env.reset() # Returns encoded state and info dict
    if info.get("error"): print(f"Error starting sim ep {episode}: {info['error']}"); return 0.0, info
    agent_pos = info.get('agent_position', 'N/A'); is_rfi = info.get('is_rfi_opportunity', False); raiser_pos = info.get('raiser_position', None)
    done = False; episode_reward = 0.0; step_count = 0; last_info = info

    while not done:
        step_count += 1
        action_idx = -1; action_str = "N/A"; current_obs_dict = {}

        # FIX 1: Use public helper to get legal actions
        legal_actions_list = env.get_legal_actions_for_agent()

        # Get current obs dict mainly for logging purposes now
        try: current_obs_dict = env._get_obs(env.agent_id) # TODO: Improve this access
        except AttributeError: print("Error: Cannot access env._get_obs for logging.")

        if not legal_actions_list:
             print(f"Warning: Agent {env.agent_id} has no legal actions in ep {episode}, step {step_count}. Env state: {env.stage}")
             action_idx = 0; action_str = action_to_string.get(action_idx, "Action0") # Use map
        else:
            # Agent selects greedy action based on encoded state
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0); agent.eval()
            with torch.no_grad(): q_values = agent(state_tensor)
            q_values_np = q_values.squeeze().cpu().numpy(); sorted_indices = np.argsort(q_values_np)[::-1]
            action_idx = -1
            # FIX 1: Use legal_actions_list for masking
            for idx in sorted_indices:
                potential_action_str = action_to_string.get(idx) # Use map
                if potential_action_str is not None and potential_action_str in legal_actions_list:
                    action_idx = idx; action_str = potential_action_str; break
            if action_idx == -1: # Fallback
                 action_str = random.choice(legal_actions_list)
                 action_idx = string_to_action.get(action_str, 0) # Use map
                 # print(f"Warning: Model predicted no legal action. Fallback random: Idx {action_idx} ('{action_str}')") # Less verbose

        # --- Log State *Before* Taking the Step ---
        try: obs_json = json.dumps(current_obs_dict)
        except Exception as e: print(f"Error serializing obs: {e}"); obs_json = "{'error': 'logging failed'}"
        detailed_writer.writerow([ episode, step_count, env.agent_id + 1, action_str, "0.00", obs_json, agent_pos, is_rfi, raiser_pos if raiser_pos else "N/A" ])

        # --- Environment Step ---
        next_state, reward, terminated, truncated, info = env.step(action_idx)
        done = terminated or truncated

        # --- Update State, Reward, and Context for Next Log ---
        state = next_state; episode_reward += reward; last_info = info
        agent_pos = info.get('agent_position', 'N/A'); is_rfi = info.get('is_rfi_opportunity', False); raiser_pos = info.get('raiser_position', None)

    # --- End of Episode ---
    final_info = last_info; final_info['final_episode_reward'] = episode_reward
    return episode_reward, final_info

# --- Main Simulation Logic ---
def main():
    args = parse_args()
    agent = BestPokerModel(input_dim=STATE_DIM, num_actions=NUM_ACTIONS)

    # --- Checkpoint Loading --- FIX 2 ---
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        try:
            checkpoint_data = torch.load(args.checkpoint, map_location="cpu")
            state_dict = None
            if isinstance(checkpoint_data, dict):
                state_dict = checkpoint_data.get('agent_state_dict',
                                                checkpoint_data.get('state_dict', checkpoint_data))
            else:
                state_dict = checkpoint_data
                if not hasattr(state_dict, 'keys'):
                     raise TypeError("Checkpoint is not a dictionary or a valid state_dict.")
            if state_dict is None:
                 raise TypeError("Could not find state_dict in checkpoint file.")

            cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            agent.load_state_dict(cleaned_state_dict, strict=False)
            print(f"Successfully loaded checkpoint.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Using untrained agent.")
            agent = BestPokerModel(input_dim=STATE_DIM, num_actions=NUM_ACTIONS)
    else:
        if not args.checkpoint: print("Error: Checkpoint path is required for simulation."); exit()
        else: print(f"Error: Checkpoint file not found at {args.checkpoint}. Using untrained agent."); agent = BestPokerModel(input_dim=STATE_DIM, num_actions=NUM_ACTIONS)
    # --- Checkpoint Loading End ---

    agent.eval()
    env = BaseFullPokerEnv(num_players=NUM_PLAYERS, render_mode=None)

    # --- Opponent Setup (same as simulate_py_gym_v3) ---
    if args.seat_config: seat_config_list = [s.strip().lower() for s in args.seat_config.split(',')];
    if len(seat_config_list) != NUM_PLAYERS: print(f"Error: --seat_config must have {NUM_PLAYERS} values."); exit(1)
    if seat_config_list[0] != "agent": print("Warning: Seat 0 must be 'agent'. Forcing."); seat_config_list[0] = "agent"
    else: seat_config_list = ["agent"] + [args.opponent for _ in range(NUM_PLAYERS - 1)]
    print(f"Seat configuration: {seat_config_list}")
    for seat_id in range(1, NUM_PLAYERS): opp_type = seat_config_list[seat_id]; policy_func = get_opponent_policy(opp_type, agent); env.set_opponent_policy(seat_id, policy_func);

    # --- Setup Logging ---
    try:
        with open(args.output_csv, mode='w', newline='') as sf: sw = csv.writer(sf); sw.writerow(["Episode", "Reward", "Info"])
    except IOError as e: print(f"Error opening summary CSV {args.output_csv}: {e}"); return
    try:
        with open(args.detailed_log, mode='w', newline='') as df:
            dw = csv.writer(df)
            dw.writerow(["Episode", "Step", "PlayerID", "Action", "StepReward", "ObservationDict", "AgentPosition", "IsRFIOpportunity", "RaiserPosition"]) # Header includes new fields
            total_reward = 0.0
            print(f"\n--- Starting Simulation ({args.episodes} episodes) ---")
            for ep in range(1, args.episodes + 1):
                ep_reward, final_info = simulate_episode(env, agent, ep, dw)
                total_reward += ep_reward
                try:
                     with open(args.output_csv, mode='a', newline='') as sf: sw = csv.writer(sf); info_str = json.dumps(final_info); sw.writerow([ep, f"{ep_reward:.2f}", info_str])
                except IOError as e: print(f"Error writing summary ep {ep}: {e}")
                except TypeError as e: print(f"Error serializing info ep {ep}: {e}");
                with open(args.output_csv, mode='a', newline='') as sf: sw = csv.writer(sf); sw.writerow([ep, f"{ep_reward:.2f}", "{'error': 'info serialization failed'}"])
                if ep % max(1, args.episodes // 10) == 0: print(f"  Completed Episode {ep}/{args.episodes}...")
            avg_reward = total_reward / args.episodes
            print(f"\n--- Simulation Complete ---"); print(f"Average Reward: {avg_reward:.2f}"); print(f"Summary: {args.output_csv}"); print(f"Detailed: {args.detailed_log}")
    except IOError as e: print(f"Error opening detailed CSV {args.detailed_log}: {e}")
    finally: env.close()

if __name__ == "__main__":
    main()
