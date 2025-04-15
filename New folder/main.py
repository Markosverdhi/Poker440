import argparse
import sys
import torch  # needed for loading checkpoint

def main():
    parser = argparse.ArgumentParser(
        description="Poker RL Agent: Train, simulate, or analyze the RL agent using a unified interface."
    )
    subparsers = parser.add_subparsers(dest="command", required=True,
                                       help="Sub-command to run: 'train', 'simulate', or 'analyze'.")

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
    # New argument for resuming training from a checkpoint.
    train_parser.add_argument(
        "--checkpoint", type=str, default="",
        help="Path to an existing model checkpoint to resume training from."
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
    # New simulation arguments:
    simulate_parser.add_argument(
        "--seat_config", type=str, default="",
        help="Comma-separated list for each seat (0 to NUM_PLAYERS-1). "
             "Seat 0 must be 'agent' (the main agent); the others can be 'model', 'random', or 'variable'."
    )
    simulate_parser.add_argument(
        "--detailed_log", type=str, default="detailed_simulation_log.csv",
        help="Path to the CSV file to store detailed game state and action logs."
    )

    # New subparser for decision analysis.
    analyze_parser = subparsers.add_parser("analyze", help="Analyze detailed simulation logs and compare decisions.")
    analyze_parser.add_argument(
        "--detailed_log", type=str, default="detailed_simulation_log.csv",
        help="Path to the detailed simulation log CSV file."
    )
    analyze_parser.add_argument(
        "--output_analysis", type=str, default="decision_analysis_summary.csv",
        help="Path to save the analysis summary."
    )

    args = parser.parse_args()

    if args.command == "train":
        from train import Train, BestPokerModel
        if args.checkpoint:
            print("Resuming training from checkpoint:", args.checkpoint)
            # Load the checkpoint once.
            checkpoint = torch.load(args.checkpoint, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            # Monkey-patch BestPokerModel.__init__ so that every new instance loads the checkpoint.
            original_init = BestPokerModel.__init__
            def new_init(self, *init_args, **init_kwargs):
                original_init(self, *init_args, **init_kwargs)
                self.load_state_dict(checkpoint, strict=False)
            BestPokerModel.__init__ = new_init
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
        if args.seat_config:
            new_argv += ["--seat_config", args.seat_config]
        if args.detailed_log:
            new_argv += ["--detailed_log", args.detailed_log]
        sys.argv = new_argv
        from simulate import main as simulate_main
        simulate_main()

    elif args.command == "analyze":
        # Rebuild sys.argv so that decision_analysis.py receives its expected arguments.
        new_argv = [sys.argv[0]]
        new_argv += ["--detailed_log", args.detailed_log]
        new_argv += ["--output_analysis", args.output_analysis]
        sys.argv = new_argv
        from decision_analysis import main as analyze_main
        analyze_main()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
