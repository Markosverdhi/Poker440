import argparse
import os
import sys
import torch

# Import model implementations
from DQN import ModelTrain, DQN
from A3C import A3CAgent  # Assuming A3C is implemented similarly
from PerformanceTracker import PerformanceTracker

# Define models directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

def load_model(model, version):
    """Loads a saved model from the models directory."""
    model_path = os.path.join(MODELS_DIR, f"{version}.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model version {version} not found in {MODELS_DIR}.")
        sys.exit(1)

    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model {version} successfully!")

def play(model_type, load_version=None, save_version=None, test_mode=False, compare_version=None):
    """Handles running the model for training or testing."""

    # Ensure --load_version and --save are not used together
    if load_version and save_version:
        print("Error: Cannot specify both --load_version and --save. If you load a model, you must NOT save another version.")
        sys.exit(1)

    # Select Model
    if model_type == "DQN":
        agent = ModelTrain(DQN)
    elif model_type == "A3C":
        agent = A3CAgent()
    else:
        print("Error: Invalid model type. Choose 'DQN' or 'A3C'.")
        sys.exit(1)

    # Initialize PerformanceTracker
    tracker = PerformanceTracker(agent, model_name=f"{model_type}_{save_version or 'new'}")

    # Load pre-trained model if specified
    if load_version:
        load_model(agent.model, load_version)

    # Run Training or Testing
    if test_mode:
        print("Running in test mode...")
        tracker_rewards = tracker.compare_models(compare_version) if compare_version else None
        tracker.plot_rewards(comparison_rewards=tracker_rewards, comparison_model_name=compare_version)
    else:
        print("Training the agent...")
        for episode in range(agent.num_episodes):
            total_reward = agent.train()
            tracker.track_rewards(total_reward)
            tracker.epsilon_decay(episode)

        tracker.plot_rewards()
        tracker.plot_epsilon()

    # Save model ONLY IF load_version was NOT used
    if save_version:
        torch.save(agent.model.state_dict(), os.path.join(MODELS_DIR, f"{save_version}.pth"))
        print(f"Model saved as {save_version}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and evaluate RL models.")
    
    parser.add_argument("--model", type=str, required=True, choices=["DQN", "A3C"],
                        help="Specify which model to run: DQN or A3C.")
    parser.add_argument("--load_version", type=str, required=False,
                        help="Load a saved model version (e.g., '1.0.0').")
    parser.add_argument("--save", type=str, required=False,
                        help="Save the trained model with a version number (e.g., '1.0.0').")
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode instead of training.")
    parser.add_argument("--compare", type=str, required=False,
                        help="Compare to another saved model (e.g., '1.0.0').")

    args = parser.parse_args()
    
    play(model_type=args.model, 
         load_version=args.load_version, 
         save_version=args.save, 
         test_mode=args.test, 
         compare_version=args.compare)
