import os
import argparse
import torch
import numpy as np
import random
import csv
import json

from models import BestPokerModel, convert_half_to_full_state_dict
from envs import BaseFullPokerEnv
from utils import encode_obs_eval, log_decision

# Global configuration for simulation.
USE_HALF_ENCODING = False  # Simulation uses full encoding.
NUM_PLAYERS = 6
STATE_DIM = 52 + 1 + (NUM_PLAYERS - 1) * 52  # 52 + 1 + 5*52 = 313 for full encoding.
NUM_ACTIONS = 6  # ['fold', 'call', 'check', 'bet_small', 'bet_big', 'all_in']

def parse_args():
    parser = argparse.ArgumentParser(description="Simulate/Evaluate the trained Poker RL Agent.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to the trained model checkpoint.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of simulation episodes to run.")
    parser.add_argument("--opponent", type=str, default="model", choices=["model", "random", "variable"],
                        help="Default type of opponent to use if --seat_config is not provided.")
    parser.add_argument("--output_csv", type=str, default="simulation_results.csv",
                        help="Path to the CSV file to store simulation summary results.")
    # New arguments:
    parser.add_argument("--seat_config", type=str, default="",
                        help="Comma-separated list for each seat (0 to NUM_PLAYERS-1). "
                             "Seat 0 should be 'agent' (the main agent); the others can be 'model', 'random', or 'variable'.")
    parser.add_argument("--detailed_log", type=str, default="detailed_simulation_log.csv",
                        help="Path to the CSV file to store detailed game state and action logs.")
    return parser.parse_args()

def get_opponent_policy(opponent_type, agent_model):
    action_index_to_str = {0: 'fold', 1: 'call', 2: 'check', 3: 'bet_small', 4: 'bet_big', 5: 'all_in'}
    if opponent_type == "model":
        def policy_fn(obs):
            state = encode_obs_eval(obs, use_half_encoding=USE_HALF_ENCODING)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = agent_model(state_tensor)
            action_idx = q_values.argmax(dim=1).item()
            return action_index_to_str[action_idx]
        return policy_fn
    elif opponent_type == "random":
        def policy_fn(obs):
            legal = obs.get("legal_actions", [])
            return random.choice(legal) if legal else "fold"
        return policy_fn
    elif opponent_type == "variable":
        def policy_fn(obs):
            # With 50% probability use the model policy, else use random.
            if random.random() < 0.5:
                state = encode_obs_eval(obs, use_half_encoding=USE_HALF_ENCODING)
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_values = agent_model(state_tensor)
                action_idx = q_values.argmax(dim=1).item()
                return action_index_to_str[action_idx]
            else:
                legal = obs.get("legal_actions", [])
                return random.choice(legal) if legal else "fold"
        return policy_fn
    else:
        # Default fallback.
        def policy_fn(obs):
            legal = obs.get("legal_actions", [])
            return random.choice(legal) if legal else "fold"
        return policy_fn

def simulate_episode(env: BaseFullPokerEnv, agent: torch.nn.Module, episode: int, detailed_writer) -> (float, dict):
    """
    Runs a single simulation episode in the provided environment using the agent's greedy policy.
    Records each step's action and state via detailed_writer.

    Args:
        env (BaseFullPokerEnv): The poker environment.
        agent (torch.nn.Module): The trained RL agent.
        episode (int): The current episode number.
        detailed_writer (csv.writer): CSV writer to log detailed step-by-step events.

    Returns:
        tuple: (episode_reward, info) where episode_reward is the cumulative reward for the episode,
               and info contains details such as winners and hand scores.
    """
    # CORRECTED LINE: Unpack the tuple returned by reset
    obs, info = env.reset() # <-- FIX HERE
    # Handle potential error during reset (e.g., not enough players to start)
    if info and info.get("error"):
         print(f"Error starting simulation episode {episode}: {info['error']}")
         # Return 0 reward and error info, or handle as appropriate
         return 0.0, info

    state = encode_obs_eval(obs, use_half_encoding=USE_HALF_ENCODING)
    done = False
    episode_reward = 0.0
    step = 0
    last_info = {} # Store the last info dict from step

    while not done:
        # Determine current player - check queue first
        curr_player = None
        if env.players_to_act:
             curr_player = env.players_to_act[0]
        else:
             # If queue is empty, round might have ended, or needs stage progression
             # env.step should handle this, but break if state seems stuck
             print(f"Warning: Episode {episode}, Step {step}: Player queue empty, but not done. Breaking loop.")
             break

        # Get current obs for the player acting (important for opponent policies)
        current_obs_for_player = env._get_obs(curr_player)
        legal_actions = current_obs_for_player.get('legal_actions', [])

        if not legal_actions:
            # Player might be all-in or state is inconsistent.
            # The environment step should handle passing turn for all-in.
            # If consistently no legal actions, log warning.
            print(f"Warning: Episode {episode}, Step {step}: Player {curr_player+1} has no legal actions. Env state: {env.stage}")
            # Assume env.step handles this state correctly (e.g., skips turn)
            # We still need an action string to pass to env.step if logic requires it.
            # Let's use a placeholder, assuming env.step internal logic is robust.
            action_str = "no_legal_action" # Placeholder
            # Or, if env.step *requires* a valid action even here, force 'fold' if possible?
            # action_str = 'fold' if 'fold' in env.action_list else env.action_list[0]

        elif curr_player == env.agent_id:
            # Agent's turn logic
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = agent(state_tensor)
            # Choose best *legal* action
            sorted_indices = torch.argsort(q_values, dim=1, descending=True).squeeze().tolist()
            action_str = None
            for idx in sorted_indices:
                 potential_action = env.action_list[idx]
                 if potential_action in legal_actions:
                      action_str = potential_action
                      break
            if action_str is None: # Fallback
                 print(f"Warning: Agent model predicted no legal action. Legal: {legal_actions}. Choosing random.")
                 action_str = random.choice(legal_actions)
            log_decision(f"[Simulation] Agent {curr_player+1} action: {action_str}")

        else:
            # Opponent's turn logic
            action_str = "fold" # Default fallback
            if curr_player in env.opponent_policies and env.opponent_policies[curr_player] is not None:
                # Pass current observation for the opponent
                action_str = env.opponent_policies[curr_player](current_obs_for_player) # Pass obs directly
                # Validate opponent action
                if action_str not in legal_actions:
                     print(f"Warning: Opponent {curr_player+1} policy chose illegal action '{action_str}'. Legal: {legal_actions}. Forcing valid.")
                     if 'call' in legal_actions: action_str = 'call'
                     elif 'check' in legal_actions: action_str = 'check'
                     else: action_str = random.choice(legal_actions) # Fallback to random legal
            else:
                # Fallback if no policy defined (shouldn't happen with setup in main)
                print(f"Warning: No policy found for opponent {curr_player+1}. Using random.")
                action_str = random.choice(legal_actions)
            log_decision(f"[Simulation] Opponent {curr_player+1} action: {action_str}")


        # --- Step in the environment ---
        # env.step handles the action and advances turns internally until agent or round end
        next_obs, reward, done, step_info = env.step(action_str)

        # Update state based on the observation *after* the step
        state = encode_obs_eval(next_obs, use_half_encoding=USE_HALF_ENCODING)
        obs = next_obs # Keep track of the latest observation dict
        episode_reward += reward
        last_info = step_info # Store the info dict from the step

        # Log detailed information
        # Use json.dumps for the observation dictionary for cleaner CSV logging
        try:
             obs_json = json.dumps(obs)
        except TypeError:
             # Handle non-serializable items in obs if they occur (e.g., complex objects)
             obs_json = json.dumps(str(obs)) # Log string representation as fallback
        detailed_writer.writerow([
            episode,
            step,
            curr_player + 1, # Log 1-based player ID
            action_str,
            f"{reward:.2f}", # Format reward
            obs_json
        ])
    
        step += 1

        # Safety break if done flag isn't correctly set by env
        if step > env.max_steps_per_round * NUM_PLAYERS * 5: # Generous step limit per episode
             print(f"Error: Episode {episode} exceeded maximum allowed steps. Terminating.")
             done = True
             last_info['error'] = 'Max steps exceeded'


    # Use the info dict returned by the *last* step, which should contain round end details
    final_info = last_info
    # Add final reward to info if needed
    final_info['final_episode_reward'] = episode_reward

    return episode_reward, final_info

def log_metrics_to_csv(metrics_file, episode, reward, info):
    """
    Logs episode metrics to a CSV file.

    Args:
        metrics_file (str): Path to the CSV file for storing metrics.
        episode (int): Episode number.
        reward (float): Episode reward or metric to log.
        info (dict): Additional information or details from the episode.
    """
    with open(metrics_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode, reward, info])

def main():
    args = parse_args()

    # Instantiate the agent model.
    agent = BestPokerModel(input_dim=STATE_DIM, num_actions=NUM_ACTIONS)
    
    # Load the checkpoint if provided.
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if "fc1.weight" not in checkpoint:
            new_checkpoint = {}
            for key, value in checkpoint.items():
                new_checkpoint[key.replace("module.", "")] = value
            checkpoint = new_checkpoint

        if "fc1.weight" in checkpoint and checkpoint["fc1.weight"].shape[1] == (26 + 1 + (NUM_PLAYERS - 1) * 26):
            checkpoint = convert_half_to_full_state_dict(checkpoint)
            print("Converted half-poker checkpoint to full-poker dimensions.")
        # Load with strict=False to allow missing keys (e.g., new res_block3 parameters)
        agent.load_state_dict(checkpoint, strict=False)
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("No valid checkpoint provided. Using untrained agent for simulation.")

    agent.eval()

    # Instantiate the simulation environment.
    env = BaseFullPokerEnv(num_players=NUM_PLAYERS)

    # Set up seat configuration.
    if args.seat_config:
        seat_config = [s.strip().lower() for s in args.seat_config.split(',')]
        if len(seat_config) != NUM_PLAYERS:
            print(f"Error: --seat_config must have exactly {NUM_PLAYERS} comma-separated values (found {len(seat_config)}).")
            exit(1)
        if seat_config[0] != "agent":
            print("Warning: Seat 0 is the main agent; forcing seat 0 to 'agent'.")
            seat_config[0] = "agent"
    else:
        # Default: main agent in seat 0 and all others using the default opponent type.
        seat_config = ["agent"] + [args.opponent for _ in range(NUM_PLAYERS - 1)]

    # Assign opponent policies according to seat_config for seats 1 through NUM_PLAYERS-1.
    for seat in range(1, NUM_PLAYERS):
        opp_type = seat_config[seat]
        if opp_type in ["agent", "model"]:
            env.opponent_policies[seat] = get_opponent_policy("model", agent)
        elif opp_type == "random":
            env.opponent_policies[seat] = get_opponent_policy("random", agent)
        elif opp_type == "variable":
            env.opponent_policies[seat] = get_opponent_policy("variable", agent)
        else:
            # Fallback option.
            env.opponent_policies[seat] = get_opponent_policy("random", agent)
    print(f"Seat configuration: {seat_config}")

    # Open CSV files for logging.
    # Summary metrics CSV.
    with open(args.output_csv, mode='w', newline='') as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow(["Episode", "Reward", "Info"])

    # Detailed log CSV.
    with open(args.detailed_log, mode='w', newline='') as detailed_file:
        detailed_writer = csv.writer(detailed_file)
        detailed_writer.writerow(["Episode", "Step", "Current_Player", "Action", "Reward", "Observation"])

        total_reward = 0.0
        # Run simulation episodes.
        for ep in range(1, args.episodes + 1):
            ep_reward, info = simulate_episode(env, agent, ep, detailed_writer)
            total_reward += ep_reward
            print(f"Episode {ep}: Reward = {ep_reward:.2f}, Info = {info}")

            # Log summary metrics.
            log_metrics_to_csv(args.output_csv, ep, ep_reward, info)

    avg_reward = total_reward / args.episodes
    print(f"Average Reward over {args.episodes} episodes: {avg_reward:.2f}")

if __name__ == "__main__":
    main()
