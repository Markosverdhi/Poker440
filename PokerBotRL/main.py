"""
main.py

This is the central entry point for the Poker RL Agent project. It supports
two primary modes of operation via subcommands:

  - train: Runs the RL training loop using the BestPokerModel architecture and TrainFullPokerEnv.
  - simulate: Evaluates a trained agent in a simulation environment using a greedy policy.

Usage:
    python main.py train [--episodes EPISODES] [--random RANGE] [--variable]
    python main.py simulate [--checkpoint PATH] [--episodes N] [--opponent {model,random,variable}]
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Poker RL Agent: Train or simulate the RL agent using a unified interface."
    )
    subparsers = parser.add_subparsers(dest="command", required=True,
                                       help="Sub-command to run: 'train' or 'simulate'.")

    # Subparser for training.
    train_parser = subparsers.add_parser("train", help="Train the RL agent.")
    train_parser.add_argument(
        "--episodes", type=int, default=1000000,
        help="Total number of training episodes."
    )
    train_parser.add_argument(
        "--random", type=str, default=None,
        help="Episode range for using random policy for opponent 1 (format: start-end)."
    )
    train_parser.add_argument(
        "--variable", action="store_true",
        help="Enable variable training mode for opponent 1 (switch between model and random every 1000 episodes)."
    )

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
    simulate_parser.add_argument(
        "--opponent", type=str, default="model", choices=["model", "random", "variable"],
        help="Type of opponent to use: model, random, or variable."
    )

    args = parser.parse_args()

    if args.command == "train":
        from train import Train
        print("Starting training.")
        trainer = Train(episodes=args.episodes, random_range=args.random, variable_mode=args.variable)
        trainer.run()
    elif args.command == "simulate":
        # Rebuild sys.argv so that simulate.py receives its expected arguments.
        new_argv = [sys.argv[0]]
        if args.checkpoint:
            new_argv += ["--checkpoint", args.checkpoint]
        new_argv += ["--episodes", str(args.episodes)]
        new_argv += ["--opponent", args.opponent]
        sys.argv = new_argv
        from simulate import main as simulate_main
        simulate_main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
