"""
main.py

This is the central entry point for the Poker RL Agent project. It supports
two primary modes of operation via subcommands:

  - train: Runs the RL training loop using the BestPokerModel architecture and TrainFullPokerEnv.
  - simulate: Evaluates a trained agent in a simulation environment using a greedy policy.

Usage:
    python main.py train [--checkpoint PATH]
    python main.py simulate [--checkpoint PATH] [--episodes N]

If a checkpoint is provided, it is used to resume training or to load the agent for simulation.
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Poker RL Agent: Train or simulate the RL agent using a unified interface."
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run: 'train' or 'simulate'.")

    # Subparser for training.
    train_parser = subparsers.add_parser("train", help="Train the RL agent.")
    train_parser.add_argument(
        "--checkpoint", type=str, default="",
        help="Path to a checkpoint file to resume training (optional)."
    )
    # You could add additional training-specific arguments here (e.g., number of episodes override).

    # Subparser for simulation/evaluation.
    simulate_parser = subparsers.add_parser("simulate", help="Simulate/evaluate the trained RL agent.")
    simulate_parser.add_argument(
        "--checkpoint", type=str, default="",
        help="Path to the trained model checkpoint."
    )
    simulate_parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of simulation episodes to run."
    )

    args = parser.parse_args()

    if args.command == "train":
        # Import and run the training script.
        from train_rl import train
        if args.checkpoint:
            try:
                import torch
                checkpoint = torch.load(args.checkpoint)
                print(f"Resuming training using checkpoint: {args.checkpoint}")
                train(agent_checkpoint=checkpoint)
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting training from scratch.")
                train()
        else:
            print("Starting training from scratch.")
            train()
    elif args.command == "simulate":
        # Forward the simulate subcommand arguments to the simulate module.
        # We adjust sys.argv so that the simulate script can reuse its own argument parser.
        sys.argv = [sys.argv[0],
                    "--checkpoint", args.checkpoint,
                    "--episodes", str(args.episodes)]
        from simulate import main as simulate_main
        simulate_main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
