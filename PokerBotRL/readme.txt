Poker RL Agent
This repository contains a reinforcement learning (RL) project designed to train and evaluate an agent for playing full poker. The agent leverages a robust DQN architecture with dueling networks, noisy linear layers, and residual blocks. The code is organized into several modular components that minimize redundancy and promote clean, maintainable design.

Repository Structure
models.py
Defines the core neural network architectures used by the agent, including BestPokerModel and its supporting custom layers (e.g., NoisyLinear and ResidualBlock). It also provides utility functions for converting checkpoints (e.g., from half-poker to full-poker input dimensions).

envs.py
Contains the poker environment classes.

BaseFullPokerEnv: Implements the basic game logic, such as dealing, betting rounds, stage progression, hand evaluation, and opponent actions.

TrainFullPokerEnv: Extends the base environment for training purposes by adding features like tracking all-in events and modified reward computation.

utils.py
Provides shared helper functions, including:

Observation encoders (encode_obs and encode_obs_eval) to transform game states into numeric vectors.

Epsilon decay computation for the epsilon-greedy policy.

A ReplayBuffer class for experience replay.

Logging utilities for debugging and decision tracking.

train_rl.py
Implements the main RL training loop. It utilizes BestPokerModel and TrainFullPokerEnv, applies an epsilon-greedy strategy, performs experience replay, periodically updates a target network, and logs performance metrics. Checkpoints and training metrics are periodically saved.

train_random.py
Similar to train_rl.py, this script trains the RL agent; however, it explicitly assigns a random action policy to one opponent (e.g., opponent with ID 1) to introduce additional stochasticity during training.

simulate.py (or evaluate.py)
Sets up a simulation environment using BaseFullPokerEnv and runs evaluation episodes using a deterministic, greedy policy. It loads a trained model checkpoint, runs the episodes, and prints detailed results (e.g., rewards, winners, hand scores).

main.py
Acts as a unified entry point for the project, enabling you to run either training or simulation via subcommands. This file ties together the modules and reduces redundancy in command-line usage.

Prerequisites
Python 3.7 or higher

PyTorch

NumPy

Other common Python libraries (e.g., random, csv)

You can install the necessary dependencies with:

bash
Copy
pip install torch numpy
(Add any additional dependency installation instructions as needed.)

How to Run
Training the Agent
To train the RL agent from scratch or resume from a checkpoint, run:

bash
Copy
python main.py train --checkpoint path/to/optional_checkpoint.pt
If no checkpoint is provided, training will start from scratch.

Training metrics (e.g., episode reward, average reward, epsilon value) are logged to the console and saved periodically as CSV files in the checkpoints directory.

Training with a Random Opponent Policy
To train the agent while one opponent follows a random action policy, run:

bash
Copy
python train_random.py --checkpoint path/to/optional_checkpoint.pt
This script is similar to the standard training loop but explicitly assigns a random policy to one of the opponent agents.

Evaluating/Simulating the Agent
To evaluate a trained model using a greedy (deterministic) policy, run:

bash
Copy
python main.py simulate --checkpoint path/to/trained_checkpoint.pt --episodes 10
This command runs 10 simulation episodes using the provided checkpoint and prints the episode rewards and additional game outcome details (like winners and scores) for analysis.

Performance Metrics Logging
Both training scripts log key performance metrics during training:

Episode Reward: The cumulative reward for each episode.

Average Reward: The moving average of rewards over recent episodes (e.g., the last 100 episodes).

Epsilon Value: The current exploration rate, as computed by the epsilon decay function.

These metrics are printed at regular intervals (e.g., every 100 episodes) and periodically saved to CSV files in the checkpoint directory, allowing you to monitor training progress and analyze performance over time.