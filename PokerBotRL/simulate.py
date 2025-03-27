"""
simulate.py

This script evaluates a trained poker RL agent in a simulated environment.
It loads the BestPokerModel from models.py, sets up the BaseFullPokerEnv from envs.py,
and runs a specified number of episodes using a greedy policy for the agent.
Results including per-episode rewards and game outcome information are printed to the console.
"""

import os
import argparse
import torch
import numpy as np

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
    return parser.parse_args()


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
        # If it's the agent's turn, select the greedy action.
        if env.current_player == env.agent_id:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = agent(state_tensor)
            action_idx = q_values.argmax(dim=1).item()
            action_str = env.action_list[action_idx]
        else:
            # For other players, use a default action (e.g., "call").
            action_str = 'call'

        # Log decisions if enabled.
        log_decision(f"[Simulation] Player {env.current_player} action: {action_str}")

        obs, reward, done, info = env.step(action_str)
        state = encode_obs_eval(obs, use_half_encoding=USE_HALF_ENCODING)
        episode_reward += reward

    return episode_reward, info


def main():
    args = parse_args()

    # Instantiate the agent model.
    agent = BestPokerModel(input_dim=STATE_DIM, num_actions=NUM_ACTIONS)
    
    # Load the checkpoint if provided.
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        # Convert checkpoint dimensions if needed.
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

    total_reward = 0.0
    for ep in range(1, args.episodes + 1):
        ep_reward, info = simulate_episode(env, agent)
        total_reward += ep_reward
        print(f"Episode {ep}: Reward = {ep_reward:.2f}, Info = {info}")

    avg_reward = total_reward / args.episodes
    print(f"Average Reward over {args.episodes} episodes: {avg_reward:.2f}")


if __name__ == "__main__":
    main()
