# filename: package/simulate.py
"""
simulate.py

Simulates games using a trained Poker RL Agent against configured opponents
using the Gymnasium-compliant BaseFullPokerEnv (adapted for tournament play).

MODIFIED (Add 'eval' opponent type):
- Added 'eval' choice to --opponent argument.
- Updated main function to handle 'eval' type:
    - Looks for checkpoints in './checkpoints/'.
    - Randomly selects a checkpoint file (.pt).
    - Loads the model from the selected checkpoint.
    - Uses the loaded model for the opponent policy.
    - Falls back to 'random' if no checkpoints are found.
"""

import os
import argparse
import torch
import numpy as np
import random
import csv
import json
import glob # Import glob for finding files

# Import Gymnasium-compliant environment and updated utils
try:
    # Use the environment class compatible with tournament logic
    from Back_End.envs import BaseFullPokerEnv # Or TrainFullPokerEnv if needed
    # Use updated utils with new encoding and state dim, plus refactored functions
    from Back_End.utils import encode_obs_eval, log_decision, NEW_STATE_DIM, load_agent_model, get_opponent_policy
except ImportError:
    print("ERROR: Ensure envs.py and utils.py (with NEW_STATE_DIM and refactored functions) are available.")
    exit()

# Assuming models.py is available and updated for NEW_STATE_DIM
try:
    from Back_End.models import BestPokerModel
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
CHECKPOINT_DIR = "checkpoints" # Define checkpoint directory

# Attempt to get action list from env dynamically
try:
    # Use the same env class intended for simulation
    # Need a dummy seat config for instantiation if env requires it now
    dummy_config = {i: 'model' for i in range(NUM_PLAYERS)}
    temp_env = BaseFullPokerEnv(num_players=NUM_PLAYERS, seat_config=dummy_config)
    ACTION_LIST = temp_env.action_list[:]
    NUM_ACTIONS = temp_env.action_space.n
    action_to_string = {i: s for i, s in enumerate(ACTION_LIST)}
    string_to_action = {s: i for i, s in enumerate(ACTION_LIST)}
    temp_env.close()
    print(f"Dynamically obtained Action List: {ACTION_LIST}")
except Exception as e:
    print(f"Warning: Could not get action list/space from env: {e}. Using default: {ACTION_LIST}")
    # Reset defaults if dynamic loading fails
    ACTION_LIST = ['fold', 'call', 'check', 'bet_small', 'bet_big', 'all_in']
    NUM_ACTIONS = len(ACTION_LIST)
    action_to_string = {i: s for i, s in enumerate(ACTION_LIST)}
    string_to_action = {s: i for i, s in enumerate(ACTION_LIST)}


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate/Evaluate the trained Poker RL Agent (Tournament Mode).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pt) REQUIRED for simulation.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of simulation episodes (tournaments) to run.")
    # *** ADDED 'eval' to choices ***
    parser.add_argument("--opponent", type=str, default="model", choices=["model", "random", "variable", "eval"], help="Default type of opponent to use if --seat_config is not provided.")
    parser.add_argument("--output_csv", type=str, default="Output_CSVs/simulation_results.csv", help="Path to the CSV file to store simulation summary results.")
    parser.add_argument("--seat_config", type=str, default="", help="Comma-separated list for each seat (0 to NUM_PLAYERS-1). Seat 0 must be 'agent'. Example: 'agent,model,random,eval,random,model'")
    parser.add_argument("--detailed_log", type=str, default="Output_CSVs/detailed_simulation_log.csv", help="Path to the CSV file to store detailed game state and action logs.")
    return parser.parse_args()

# --- REMOVED get_opponent_policy function (now imported from utils) ---


# --- Simulation Episode (Logging Fixed) ---
def simulate_episode(env: BaseFullPokerEnv, agent: torch.nn.Module, episode: int, detailed_writer, device) -> (float, dict):
    """ Runs one simulation tournament (episode). """
    print(f"--- Starting Simulation Tournament {episode} ---")
    try:
        # Env reset now uses seat_config stored during init
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
        agent_took_action_this_step = False # Flag to track if agent acted

        # Check if current player is valid and not empty
        current_player_id = env.current_player_id if hasattr(env, 'current_player_id') else None
        is_empty_seat = env.seat_config.get(current_player_id) == 'empty'

        if current_player_id is None or is_empty_seat:
            # If turn is on an empty seat or invalid, step with dummy action to advance
            action_idx = -1
            action_str = "SKIP (Empty/Invalid)"
            # print(f"Debug: Skipping turn for player {current_player_id} (Empty/Invalid)") # Debug
        elif current_player_id == env.agent_id:
            agent_took_action_this_step = True # Mark that the agent is determining an action
            try:
                # Use _get_obs_dict which should handle empty/invalid players
                current_obs_dict = env._get_obs_dict(env.agent_id)
                legal_actions_list = current_obs_dict.get('legal_actions', [])
            except Exception as e:
                print(f"Error getting obs/legal actions for agent: {e}"); legal_actions_list = ['fold']

            if not legal_actions_list:
                action_idx = -1; action_str = "None (No Legal)"
            else:
                # Use encode_obs_eval for consistency
                state_encoded = encode_obs_eval(current_obs_dict)
                state_tensor = torch.tensor(state_encoded, dtype=torch.float32, device=device).unsqueeze(0)
                agent.eval()
                with torch.no_grad(): q_values = agent(state_tensor)
                q_values_np = q_values.squeeze().cpu().numpy(); sorted_indices = np.argsort(q_values_np)[::-1]
                action_idx = -1
                for idx in sorted_indices:
                    potential_action_str = action_to_string.get(idx)
                    if potential_action_str is not None and potential_action_str in legal_actions_list:
                        action_idx = idx; action_str = potential_action_str; break
                if action_idx == -1: # Fallback if no legal action found in Q-values
                    action_str = random.choice(legal_actions_list); action_idx = string_to_action.get(action_str, 0)
                    print(f"Warning: Agent model failed to find legal action, chose random: {action_str}")

            # --- Logging moved AFTER env.step() ---

        else: # Opponent's turn (or skip step for empty seat)
            action_idx = -1 # Signal environment to use internal policy or skip

        # --- Environment Step ---
        step_reward = 0.0 # Initialize reward for this step
        try:
            next_state_encoded, step_reward, terminated, truncated, info = env.step(action_idx)
            done = terminated or truncated
            last_info = info # Store latest info regardless of who acted
            # Update state representation for agent if needed (only relevant if agent acts again)
            # state = next_state_encoded # This assumes obs is always encoded state
        except Exception as e:
            print(f"Error during env.step() in tournament {episode}, step {step_count}: {e}")
            import traceback
            traceback.print_exc()
            done=True; break # End episode on error

        # --- Log Agent's Action AFTER Stepping ---
        if agent_took_action_this_step and detailed_writer:
            # Use the obs_dict captured *before* the step, but reward from *after* the step
            agent_pos = current_obs_dict.get('position', 'N/A'); # Position might not be in obs_dict
            # RFI opportunity requires more context from env state, not just obs_dict
            is_rfi = False # Placeholder - TODO: Get this info from env state if needed
            raiser_pos = env.last_raiser if hasattr(env, 'last_raiser') else None # Get from env state
            try: obs_json = json.dumps(current_obs_dict, sort_keys=True, default=str)
            except Exception as e: print(f"Error serializing obs_dict: {e}"); obs_json = "{'error': 'logging failed'}"
            # Log the actual step_reward
            detailed_writer.writerow([
                episode, step_count, env.agent_id + 1, action_str,
                f"{step_reward:.2f}", # Use actual reward
                obs_json, agent_pos, is_rfi,
                raiser_pos if raiser_pos is not None else "N/A"
            ])


        # Accumulate reward (env.step now returns round reward at round end)
        # The tournament_reward accumulates the step_reward which is now correctly logged
        if isinstance(step_reward, (int, float)): tournament_reward += step_reward

    # --- End of Episode ---
    final_info = last_info if last_info else {}
    # Ensure final tournament reward is captured correctly (might be in last_info already)
    final_reward_from_info = final_info.get('final_tournament_reward', None)
    if final_reward_from_info is not None:
        tournament_reward = final_reward_from_info
    else:
        # Calculate from stack change if not provided directly
        if env.agent_id in env.stacks and env.agent_id in env.initial_stacks_this_round:
            final_stack = env.stacks.get(env.agent_id, 0)
            # Need initial stack *at tournament start* - env doesn't track this easily
            # Use accumulated step rewards as best estimate if final not given
            final_info['calculated_tournament_reward'] = tournament_reward
        else:
            final_info['final_tournament_reward'] = tournament_reward


    print(f"--- Finished Simulation Tournament {episode}. Final Reward: {tournament_reward:.2f} ---")
    return tournament_reward, final_info


# --- Main Simulation Logic ---
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Agent Model using centralized function ---
    print(f"Loading checkpoint from {args.checkpoint}")
    agent = load_agent_model(args.checkpoint, NUM_ACTIONS, device)

    if agent is None:
        print(f"FATAL: Failed to load agent model from {args.checkpoint}. Exiting.")
        exit(1)
    agent.eval()

    # --- Parse Seat Configuration ---
    seat_config_dict = {}
    agent_id_sim = 0 # Assume agent is always seat 0 for simulation setup ease
    if args.seat_config:
        seat_config_list = [s.strip().lower() for s in args.seat_config.split(',')]
        if len(seat_config_list) != NUM_PLAYERS:
            print(f"Error: --seat_config must have {NUM_PLAYERS} values. Received: {len(seat_config_list)}"); exit(1)
        if seat_config_list[0] != "agent":
            print("Error: Seat 0 (index 0) in --seat_config must be 'agent'."); exit(1)
        for i, seat_type in enumerate(seat_config_list):
            # *** ADDED 'eval' to allowed types ***
            if seat_type not in ["agent", "model", "random", "variable", "eval", "empty"]:
                print(f"Error: Invalid seat type '{seat_type}' in --seat_config. Allowed: agent, model, random, variable, eval, empty"); exit(1)
            seat_config_dict[i] = seat_type
    else:
        # Default config: agent at seat 0, others based on --opponent arg
        seat_config_dict[0] = "agent"
        for i in range(1, NUM_PLAYERS):
            seat_config_dict[i] = args.opponent

    print(f"Seat configuration: {seat_config_dict}")


    # --- Initialize Environment ---
    try:
        # Pass seat config to environment
        env = BaseFullPokerEnv(
            num_players=NUM_PLAYERS,
            agent_id=agent_id_sim, # Agent is at seat 0
            render_mode=None,
            seat_config=seat_config_dict
        )
    except Exception as e: print(f"Error initializing environment: {e}"); exit(1)

    # --- Opponent Setup ---
    # *** MODIFIED to handle 'eval' type ***
    for seat_id in range(NUM_PLAYERS):
        if seat_id == env.agent_id: continue # Skip agent seat
        opp_type = seat_config_dict.get(seat_id)
        if opp_type == 'empty': continue # Skip empty seats

        policy_func = None
        opponent_model_instance = None # Specific model instance for this seat
        policy_info_str = opp_type # Default info string

        if opp_type == 'eval':
            # Find checkpoints
            checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_*.pt"))
            if not checkpoint_files:
                print(f"Warning: No checkpoints found in '{CHECKPOINT_DIR}' for eval opponent at seat {seat_id+1}. Using 'random'.")
                opp_type = 'random' # Fallback to random
                policy_info_str = f"{opp_type} (fallback)"
            else:
                # Select a random checkpoint
                selected_checkpoint = random.choice(checkpoint_files)
                print(f"Seat {seat_id+1} ('eval'): Loading random checkpoint '{os.path.basename(selected_checkpoint)}'")
                try:
                    # Load the model state dict directly
                    checkpoint_data = torch.load(selected_checkpoint, map_location=device)
                    # Handle dict or raw state_dict
                    state_dict = checkpoint_data.get('agent_state_dict', checkpoint_data) if isinstance(checkpoint_data, dict) else checkpoint_data
                    if not isinstance(state_dict, dict): raise TypeError("Loaded state is not a dict.")

                    # Clean 'module.' prefix if needed
                    cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                    # Create a new model instance for this opponent
                    opponent_model_instance = BestPokerModel(num_actions=NUM_ACTIONS).to(device)
                    load_info = opponent_model_instance.load_state_dict(cleaned_state_dict, strict=False) # Use strict=False for flexibility
                    if load_info.missing_keys or load_info.unexpected_keys:
                         print(f"  Load info for seat {seat_id+1}: Missing={load_info.missing_keys}, Unexpected={load_info.unexpected_keys}")
                    opponent_model_instance.eval()

                    # Now use the 'model' logic in get_opponent_policy, passing the loaded model
                    opp_type = 'model' # Treat as model type for policy creation
                    policy_info_str = f"eval ({os.path.basename(selected_checkpoint)})"

                except Exception as e:
                    print(f"Error loading checkpoint {os.path.basename(selected_checkpoint)} for seat {seat_id+1}: {e}. Using 'random'.")
                    opp_type = 'random' # Fallback to random on error
                    policy_info_str = f"{opp_type} (fallback - load error)"
                    opponent_model_instance = None # Ensure no model is used

        # Get policy using the centralized function
        # Pass the specifically loaded model if opp_type is now 'model' (from 'eval')
        # Pass the main agent model if original opp_type was 'model'
        model_to_use = opponent_model_instance if opponent_model_instance is not None else agent

        policy_func = get_opponent_policy(
            opponent_type=opp_type, # Use potentially modified type ('model' or 'random')
            agent_model=model_to_use, # Use the appropriate model
            action_list=ACTION_LIST,
            num_actions=NUM_ACTIONS,
            device=device
        )

        if policy_func:
            env.set_opponent_policy(seat_id, policy_func)
            print(f"Set Seat {seat_id+1} policy to: {policy_info_str}")
        else:
            print(f"Warning: Could not get policy for seat {seat_id+1} (type: {opp_type}). Check utils.get_opponent_policy.")

    # --- Setup Logging (Header Fixed) ---
    summary_file_path = args.output_csv; detailed_file_path = args.detailed_log
    detailed_writer = None; detailed_file_handle = None
    summary_header = ["Tournament", "TotalReward", "FinalInfo"]
    # *** FIX: Changed "StepReward" to "Reward" ***
    detailed_header = ["Tournament", "Step", "PlayerID", "Action", "Reward", "ObservationDict", "AgentPosition", "IsRFIOpportunity", "RaiserPosition"]
    # *** END FIX ***
    try:
        # Write header only if file doesn't exist or is empty
        file_exists = os.path.exists(summary_file_path) and os.path.getsize(summary_file_path) > 0
        with open(summary_file_path, mode='a', newline='') as sf:
            sw = csv.writer(sf)
            if not file_exists: sw.writerow(summary_header)
    except IOError as e: print(f"Error opening summary CSV {summary_file_path}: {e}. Summary logging disabled."); summary_file_path = None

    try:
        detailed_file_exists = os.path.exists(detailed_file_path) and os.path.getsize(detailed_file_path) > 0
        # Ensure file is opened with write access ('w' or 'a')
        # Use 'w' to overwrite old log, 'a' to append
        detailed_file_handle = open(detailed_file_path, mode='w', newline=''); # Overwrite for clean log
        print(f"Overwriting detailed log file: {detailed_file_path}")
        detailed_writer = csv.writer(detailed_file_handle)
        # Always write header when overwriting
        detailed_writer.writerow(detailed_header)
    except IOError as e: print(f"Error opening detailed CSV {detailed_file_path}: {e}. Detailed logging disabled."); detailed_writer = None;
    if detailed_file_handle and detailed_writer is None: detailed_file_handle.close() # Close file if writer failed


    # --- Run Simulation (Unchanged) ---
    total_reward_all_tournaments = 0.0
    print(f"\n--- Starting Simulation ({args.episodes} tournaments) ---")
    for ep in range(1, args.episodes + 1):
        ep_reward, final_info = simulate_episode(env, agent, ep, detailed_writer, device)
        total_reward_all_tournaments += ep_reward
        if summary_file_path:
            try:
                # Attempt to serialize final_info safely
                info_str = json.dumps(final_info, default=str)
            except TypeError as e:
                print(f"Warning: Could not serialize final_info for T {ep}: {e}. Storing basic info.")
                info_str = json.dumps({'error': 'info serialization failed', 'final_reward_calc': ep_reward}, default=str)

            try:
                with open(summary_file_path, mode='a', newline='') as sf:
                    sw = csv.writer(sf); sw.writerow([ep, f"{ep_reward:.2f}", info_str])
            except IOError as e: print(f"Error writing summary T {ep}: {e}")

        if ep % max(1, args.episodes // 10) == 0: print(f"  Completed Tournament {ep}/{args.episodes}...")

    avg_reward = total_reward_all_tournaments / args.episodes if args.episodes > 0 else 0.0
    print(f"\n--- Simulation Complete ---"); print(f"Average Tournament Reward: {avg_reward:.2f}")
    if summary_file_path: print(f"Summary results appended to: {summary_file_path}")
    if detailed_file_handle: print(f"Detailed logs written to: {detailed_file_path}") # Changed appended to written

    # --- Cleanup ---
    if detailed_file_handle: detailed_file_handle.close()
    env.close()

if __name__ == "__main__":
    main()