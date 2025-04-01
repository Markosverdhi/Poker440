import os
import argparse
import torch
import numpy as np
import random
import csv

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
                        help="Type of opponent to use: model, random, or variable.")
    parser.add_argument("--output_csv", type=str, default="simulation_results.csv",
                        help="Path to the CSV file to store simulation results.")
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
        def policy_fn(obs):
            legal = obs.get("legal_actions", [])
            return random.choice(legal) if legal else "fold"
        return policy_fn

def simulate_episode(env: BaseFullPokerEnv, agent: torch.nn.Module) -> (float, dict):
    """
    Runs a single simulation episode in the provided environment using the agent's greedy policy.

    Args:
        env (BaseFullPokerEnv): The poker environment.
        agent (torch.nn.Module): The trained RL agent.

    Returns:
        tuple: (episode_reward, info) where episode_reward is the cumulative reward for the episode,
               and info contains details such as winners and hand scores.
    """
    obs = env.reset()
    state = encode_obs_eval(obs, use_half_encoding=USE_HALF_ENCODING)
    done = False
    episode_reward = 0.0

    while not done:
        if env.current_player == env.agent_id:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = agent(state_tensor)
            action_idx = q_values.argmax(dim=1).item()
            action_str = env.action_list[action_idx]
        else:
            action_str = 'call'

        log_decision(f"[Simulation] Player {env.current_player} action: {action_str}")

        obs, reward, done, info = env.step(action_str)
        state = encode_obs_eval(obs, use_half_encoding=USE_HALF_ENCODING)
        episode_reward += reward

    return episode_reward, info

def log_metrics_to_csv(metrics_file, episode, reward, info):
    """
    Logs episode metrics to a CSV file.

    Args:
    - metrics_file (str): Path to the CSV file for storing metrics.
    - episode (int): Episode number.
    - reward (float): Episode reward or metric to log.
    - info (dict): Additional information or details from the episode.
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
        if checkpoint["fc1.weight"].shape[1] == (26 + 1 + (NUM_PLAYERS - 1) * 26):
            checkpoint = convert_half_to_full_state_dict(checkpoint)
            print("Converted half-poker checkpoint to full-poker dimensions.")
        agent.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("No valid checkpoint provided. Using untrained agent for simulation.")

    agent.eval()

    # Instantiate the simulation environment.
    env = BaseFullPokerEnv(num_players=NUM_PLAYERS)

    # Set opponent policies based on the chosen type.
    opponent_policy = get_opponent_policy(args.opponent, agent)
    for opp_id in range(1, NUM_PLAYERS):
        env.opponent_policies[opp_id] = opponent_policy
    print(f"Using '{args.opponent}' policy for opponents.")

    # Open CSV file for logging simulation results.
    with open(args.output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Reward", "Info"])

        total_reward = 0.0
        for ep in range(1, args.episodes + 1):
            ep_reward, info = simulate_episode(env, agent)
            total_reward += ep_reward
            print(f"Episode {ep}: Reward = {ep_reward:.2f}, Info = {info}")

            # Log metrics to CSV
            log_metrics_to_csv(args.output_csv, ep, ep_reward, info)

    avg_reward = total_reward / args.episodes
    print(f"Average Reward over {args.episodes} episodes: {avg_reward:.2f}")

if __name__ == "__main__":
    main()
