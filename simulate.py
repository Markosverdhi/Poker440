# filename: package/simulate.py
"""
simulate.py

Simulates games using a trained Poker RL Agent against configured opponents
using the Gymnasium-compliant BaseFullPokerEnv (adapted for tournament play).

MODIFIED (Fix UnboundLocalError):
- Restructured opponent setup logic.

MODIFIED (Fix Model Instantiation TypeError):
- Removed the unexpected 'input_dim' keyword argument when instantiating
  BestPokerModel in the main function, consistent with updated models.py.
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
    # Use the environment class compatible with tournament logic
    from envs import BaseFullPokerEnv # Or TrainFullPokerEnv if needed
    # Use updated utils with new encoding and state dim
    from utils import encode_obs_eval, log_decision, NEW_STATE_DIM
except ImportError:
    print("ERROR: Ensure envs.py and utils.py (with NEW_STATE_DIM) are available.")
    exit()

# Assuming models.py is available and updated for NEW_STATE_DIM
try:
    from models import BestPokerModel
except ImportError:
    print("ERROR: Ensure models.py is available.")
    exit()


# --- Global configuration (ensure consistency) ---
NUM_PLAYERS = 6
# Use the dimension defined in utils.py
STATE_DIM = NEW_STATE_DIM
ACTION_LIST = ['fold', 'call', 'check', 'bet_small', 'bet_big', 'all_in'] # Default/Example
NUM_ACTIONS = len(ACTION_LIST)
action_to_string = {i: s for i, s in enumerate(ACTION_LIST)}
string_to_action = {s: i for i, s in enumerate(ACTION_LIST)}

# Attempt to get action list from env dynamically
try:
    # Use the same env class intended for simulation
    temp_env = BaseFullPokerEnv(num_players=NUM_PLAYERS)
    ACTION_LIST = temp_env.action_list[:]
    NUM_ACTIONS = temp_env.action_space.n
    action_to_string = {i: s for i, s in enumerate(ACTION_LIST)}
    string_to_action = {s: i for i, s in enumerate(ACTION_LIST)}
    temp_env.close()
    print(f"Dynamically obtained Action List: {ACTION_LIST}")
except Exception as e:
    print(f"Warning: Could not get action list/space from env: {e}. Using default: {ACTION_LIST}")


def parse_args():
    # (Argument parsing remains the same)
    parser = argparse.ArgumentParser(description="Simulate/Evaluate the trained Poker RL Agent (Tournament Mode).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pt) REQUIRED for simulation.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of simulation episodes (tournaments) to run.")
    parser.add_argument("--opponent", type=str, default="model", choices=["model", "random", "variable"], help="Default type of opponent to use if --seat_config is not provided.")
    parser.add_argument("--output_csv", type=str, default="simulation_results.csv", help="Path to the CSV file to store simulation summary results.")
    parser.add_argument("--seat_config", type=str, default="", help="Comma-separated list for each seat (0 to NUM_PLAYERS-1). Seat 0 must be 'agent'. Example: 'agent,model,random,model,random,model'")
    parser.add_argument("--detailed_log", type=str, default="detailed_simulation_log.csv", help="Path to the CSV file to store detailed game state and action logs.")
    return parser.parse_args()

# --- Opponent Policy Creation (Unchanged from previous fix) ---
def get_opponent_policy(opponent_type, agent_model):
    """ Creates a policy function for non-human opponents. """
    if opponent_type == "model":
        if not agent_model:
            print("Warning: 'model' opponent type selected but no agent model loaded. Using 'random'.")
            return get_opponent_policy("random", None)
        def policy_fn(obs_dict):
            if not isinstance(obs_dict, dict): return 'fold'
            legal_actions = obs_dict.get('legal_actions', [])
            if not legal_actions: return 'fold'
            try: state = encode_obs_eval(obs_dict)
            except Exception as e: print(f"Error encoding opponent obs: {e}. Folding."); return 'fold'
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            agent_model.eval()
            with torch.no_grad(): q_values = agent_model(state_tensor)
            q_values_np = q_values.squeeze().cpu().numpy(); sorted_indices = np.argsort(q_values_np)[::-1]
            for action_idx in sorted_indices:
                if 0 <= action_idx < NUM_ACTIONS:
                    action_str = action_to_string.get(action_idx)
                    if action_str and action_str in legal_actions: return action_str
            if 'check' in legal_actions: return 'check'
            if 'call' in legal_actions: return 'call'
            if 'fold' in legal_actions: return 'fold'
            return random.choice(legal_actions) if legal_actions else 'fold'
        return policy_fn
    elif opponent_type == "random":
        def policy_fn(obs_dict):
            if not isinstance(obs_dict, dict): return 'fold'
            legal = obs_dict.get("legal_actions", [])
            return random.choice(legal) if legal else "fold"
        return policy_fn
    elif opponent_type == "variable":
        model_policy = get_opponent_policy("model", agent_model)
        random_policy = get_opponent_policy("random", None)
        def policy_fn(obs_dict): return model_policy(obs_dict) if random.random() < 0.5 else random_policy(obs_dict)
        return policy_fn
    else:
        print(f"Warning: Unknown opponent type '{opponent_type}'. Using 'random'.")
        return get_opponent_policy("random", None)


# --- Simulation Episode (Unchanged from previous fix) ---
def simulate_episode(env: BaseFullPokerEnv, agent: torch.nn.Module, episode: int, detailed_writer) -> (float, dict):
    """ Runs one simulation tournament (episode). """
    print(f"--- Starting Simulation Tournament {episode} ---")
    try:
        state, info = env.reset()
    except Exception as e:
        print(f"Error during env.reset() for tournament {episode}: {e}")
        return 0.0, {"error": f"Reset failed: {e}"}

    if info.get("error"):
        print(f"Error starting sim tournament {episode}: {info['error']}")
        return 0.0, info

    done = False; tournament_reward = 0.0; step_count = 0; last_info = info

    while not done:
        step_count += 1; action_idx = -1; action_str = "N/A"; current_obs_dict = {}

        if env.current_player_id == env.agent_id:
            try:
                if hasattr(env, '_get_obs_dict'):
                     current_obs_dict = env._get_obs_dict(env.agent_id)
                     legal_actions_list = current_obs_dict.get('legal_actions', [])
                else: current_obs_dict = {}; legal_actions_list = env.get_legal_actions_for_agent()
            except Exception as e: print(f"Error getting obs/legal actions: {e}"); legal_actions_list = ['fold']

            if not legal_actions_list: action_idx = -1; action_str = "None"
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0); agent.eval()
                with torch.no_grad(): q_values = agent(state_tensor)
                q_values_np = q_values.squeeze().cpu().numpy(); sorted_indices = np.argsort(q_values_np)[::-1]
                action_idx = -1
                for idx in sorted_indices:
                    potential_action_str = action_to_string.get(idx)
                    if potential_action_str is not None and potential_action_str in legal_actions_list:
                        action_idx = idx; action_str = potential_action_str; break
                if action_idx == -1: action_str = random.choice(legal_actions_list); action_idx = string_to_action.get(action_str, 0)

            # Log state before step
            agent_pos = current_obs_dict.get('position', 'N/A'); is_rfi = False; raiser_pos = env.last_raiser
            try: obs_json = json.dumps(current_obs_dict, sort_keys=True, default=str)
            except Exception as e: print(f"Error serializing obs_dict: {e}"); obs_json = "{'error': 'logging failed'}"
            if detailed_writer: detailed_writer.writerow([ episode, step_count, env.agent_id + 1, action_str, "0.00", obs_json, agent_pos, is_rfi, raiser_pos if raiser_pos is not None else "N/A" ])

            # Environment Step for Agent
            try: next_state, reward, terminated, truncated, info = env.step(action_idx)
            except Exception as e: print(f"Error during agent env.step(): {e}"); done=True; break # End episode on error
            done = terminated or truncated

        else:
            # Environment Step for Opponent (dummy action)
            try: next_state, reward, terminated, truncated, info = env.step(-1)
            except Exception as e: print(f"Error during opponent env.step(): {e}"); done=True; break # End episode on error
            done = terminated or truncated

        state = next_state; last_info = info
        if isinstance(reward, (int, float)): tournament_reward += reward

    final_info = last_info; final_info['final_tournament_reward'] = tournament_reward
    print(f"--- Finished Simulation Tournament {episode}. Final Reward: {tournament_reward:.2f} ---")
    return tournament_reward, final_info

# --- Main Simulation Logic ---
def main():
    args = parse_args()

    # --- Load Agent Model ---
    try:
        # ** FIXED: Instantiate model without input_dim **
        agent = BestPokerModel(num_actions=NUM_ACTIONS)
    except NameError:
        print("Error: NUM_ACTIONS not defined. Check dynamic action list loading.")
        exit(1)
    except Exception as e:
        # This is where the user's error message comes from
        print(f"Error creating model: {e}")
        exit(1)

    # Load checkpoint (unchanged logic, but now into correctly sized model)
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        try:
            checkpoint_data = torch.load(args.checkpoint, map_location=torch.device("cpu"))
            state_dict = None
            if isinstance(checkpoint_data, dict): state_dict = checkpoint_data.get('agent_state_dict', checkpoint_data.get('state_dict', checkpoint_data))
            else: state_dict = checkpoint_data
            if state_dict is None or not hasattr(state_dict, 'keys'): raise TypeError("Could not find valid state_dict.")
            cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # Load with strict=False, but size mismatch will now cause error here if checkpoint is old
            agent.load_state_dict(cleaned_state_dict, strict=False)
            print(f"Successfully loaded checkpoint.")
        except Exception as e:
            # This includes the size mismatch error if checkpoint is incompatible
            print(f"Error loading checkpoint: {e}. Using untrained agent.")
            agent = BestPokerModel(num_actions=NUM_ACTIONS) # Re-initialize
    else:
        print(f"Error: Checkpoint file not found at {args.checkpoint}. Using untrained agent.")
        agent = BestPokerModel(num_actions=NUM_ACTIONS)

    agent.eval()

    # --- Initialize Environment ---
    try:
        # Ensure agent_id=0 matches assumption in seat config logic below
        env = BaseFullPokerEnv(num_players=NUM_PLAYERS, agent_id=0, render_mode=None)
    except Exception as e: print(f"Error initializing environment: {e}"); exit(1)

    # --- Opponent Setup (Unchanged from previous fix) ---
    if args.seat_config:
        seat_config_list = [s.strip().lower() for s in args.seat_config.split(',')]
        if len(seat_config_list) != NUM_PLAYERS: print(f"Error: --seat_config must have {NUM_PLAYERS} values. Received: {len(seat_config_list)}"); exit(1)
        if seat_config_list[0] != "agent": print("Error: Seat 0 in --seat_config must be 'agent'."); exit(1)
    else: seat_config_list = ["agent"] + [args.opponent for _ in range(NUM_PLAYERS - 1)]

    print(f"Seat configuration: {seat_config_list}")
    for seat_id in range(NUM_PLAYERS):
        if seat_id == env.agent_id: continue
        opp_type = seat_config_list[seat_id]
        policy_func = get_opponent_policy(opp_type, agent) # Pass loaded agent model
        env.set_opponent_policy(seat_id, policy_func)
        print(f"Set Seat {seat_id+1} policy to: {opp_type}")


    # --- Setup Logging (Unchanged) ---
    summary_file_path = args.output_csv; detailed_file_path = args.detailed_log
    detailed_writer = None; detailed_file_handle = None
    try:
        with open(summary_file_path, mode='w', newline='') as sf: sw = csv.writer(sf); sw.writerow(["Tournament", "TotalReward", "FinalInfo"])
    except IOError as e: print(f"Error opening summary CSV {summary_file_path}: {e}. Summary logging disabled."); summary_file_path = None
    try:
        detailed_file_handle = open(detailed_file_path, mode='w', newline=''); detailed_writer = csv.writer(detailed_file_handle)
        detailed_writer.writerow(["Tournament", "Step", "PlayerID", "Action", "StepReward", "ObservationDict", "AgentPosition", "IsRFIOpportunity", "RaiserPosition"])
    except IOError as e: print(f"Error opening detailed CSV {detailed_file_path}: {e}. Detailed logging disabled."); detailed_writer = None;
    if detailed_file_handle and detailed_writer is None: detailed_file_handle.close() # Close file if writer failed


    # --- Run Simulation (Unchanged) ---
    total_reward_all_tournaments = 0.0
    print(f"\n--- Starting Simulation ({args.episodes} tournaments) ---")
    for ep in range(1, args.episodes + 1):
        ep_reward, final_info = simulate_episode(env, agent, ep, detailed_writer)
        total_reward_all_tournaments += ep_reward
        if summary_file_path:
            try:
                 info_str = json.dumps(final_info, default=str)
                 with open(summary_file_path, mode='a', newline='') as sf: sw = csv.writer(sf); sw.writerow([ep, f"{ep_reward:.2f}", info_str])
            except IOError as e: print(f"Error writing summary T {ep}: {e}")
            except TypeError as e: print(f"Error serializing final_info T {ep}: {e}");
            with open(summary_file_path, mode='a', newline='') as sf: sw = csv.writer(sf); sw.writerow([ep, f"{ep_reward:.2f}", "{'error': 'info serialization failed'}"])
        if ep % max(1, args.episodes // 10) == 0: print(f"  Completed Tournament {ep}/{args.episodes}...")

    avg_reward = total_reward_all_tournaments / args.episodes if args.episodes > 0 else 0.0
    print(f"\n--- Simulation Complete ---"); print(f"Average Tournament Reward: {avg_reward:.2f}")
    if summary_file_path: print(f"Summary results saved to: {summary_file_path}")
    if detailed_file_handle: print(f"Detailed logs saved to: {detailed_file_path}")

    # --- Cleanup ---
    if detailed_file_handle: detailed_file_handle.close()
    env.close()

if __name__ == "__main__":
    main()
