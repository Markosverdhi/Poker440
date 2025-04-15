Poker RL Agent
This repository implements a reinforcement learning (RL) agent for playing poker using a dueling DQN architecture with custom layers for exploration and improved gradient flow.

Repository Structure
envs.py

Implements poker game environments.

BaseFullPokerEnv: Core game logic (dealing, betting, stage progression, hand evaluation).

TrainFullPokerEnv: Extends the base environment for training (tracks all-in events and applies modified reward logic).

models.py

Defines the neural network architecture for the RL agent.

Implements custom layers:

NoisyLinear: Linear layer with learnable noise for exploration.

ResidualBlock: Improves gradient flow.

BestPokerModel: Dueling DQN architecture for the agent.

Includes a utility to convert checkpoints from a half-poker model to full-poker dimensions.

utils.py

Contains helper functions for:

Logging decisions.

Encoding observations for training and evaluation.

Calculating epsilon decay.

Storing experiences with a ReplayBuffer.

train.py

Runs the training loop using TrainFullPokerEnv and BestPokerModel.

Updates opponent policies in a round-robin fashion (opponents 1–5).

For opponent 1, options exist for using a random policy (via --random) or variable mode (--variable) that switches between model and random policies every 1000 episodes.

Opponents 2–5 are updated with a model-based policy.

Command-line options:

--episodes: Total training episodes.

--random: Episode range for using a random policy for opponent 1.

--variable: Enable variable training mode for opponent 1.

simulate.py

Simulates evaluation episodes using BaseFullPokerEnv.

Allows the use of a trained model checkpoint.

Command-line options:

--checkpoint: Path to a trained model checkpoint.

--episodes: Number of simulation episodes.

--opponent: Type of opponent policy ("model", "random", or "variable").

--output_csv: CSV file path to log simulation results.

plot.py

Provides plotting functionality for training/simulation results.

Supports plotting:

Episode rewards.

Episode outcomes.

Custom metrics.

Command-line options:

Positional argument: CSV file path.

--metric: Metric to plot (reward, episode_outcome, or custom).

--custom_metric: Name of the custom metric (if applicable).

main.py

Acts as the central entry point.

Supports subcommands:

train: Starts training.

simulate: Runs simulation/evaluation.

Passes relevant arguments to either train.py or simulate.py.

Usage Examples
Training the Agent
Train using default settings (1,000,000 episodes, model policy for all opponents):

bash
Copy
python main.py train
Train with custom parameters:

bash
Copy
python main.py train --episodes 500000 --random "1000-2000" --variable
Running Simulations
Simulate evaluation with a trained checkpoint:

bash
Copy
python main.py simulate --checkpoint "checkpoints/final_agent_checkpoint.pt" --episodes 20 --opponent model
Or directly:

bash
Copy
python simulate.py --checkpoint "checkpoints/final_agent_checkpoint.pt" --episodes 20 --opponent random --output_csv simulation_results.csv
Plotting Results
Plot episode rewards from a CSV file:

bash
Copy
python plot.py simulation_results.csv --metric reward
Plot episode outcomes:

bash
Copy
python plot.py simulation_results.csv --metric episode_outcome
Plot a custom metric:

bash
Copy
python plot.py simulation_results.csv --metric custom --custom_metric custom_value
Dependencies
Python 3.x

PyTorch

NumPy

Matplotlib

Standard libraries: argparse, os, random, csv

This README provides a brief overview of the project's components and how to use the various scripts. For further details, please refer to the inline documentation within each file.