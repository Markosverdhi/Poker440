# PokerBotRL - Poker Reinforcement Learning Agent

## Overview

This project contains a reinforcement learning agent designed to play No-Limit Texas Hold'em poker[cite: 1]. It is built using Python, PyTorch, and Gymnasium[cite: 1]. The system allows for training the agent, simulating tournament-style games, analyzing agent decisions, and playing against the agent via a graphical user interface[cite: 2].

## Features

* **RL Agent Training**: Train a poker agent using a Dueling DQN architecture (`Scripts/train.py`)[cite: 2]. Supports resuming from checkpoints and configuring opponent strategies during training[cite: 3].
* **Game Simulation**: Simulate tournament-style poker games featuring the trained agent against configurable opponents (model-based, random, variable, empty) using `Scripts/simulate.py`[cite: 3]. Generates summary and detailed logs[cite: 4].
* **Graphical User Interface**: Play poker against the trained agent and other opponent types in a visual interface launched via `main.py ui` (code in `Front_End/main_ui.py`)[cite: 4, 9].
* **Decision Analysis**: Analyze the agent's performance and decision-making based on detailed simulation logs (`Scripts/decision_analysis.py`)[cite: 4, 9].

## Folder Structure

The codebase is organized as follows:

Package/
├── main.py                 # Main entry point, command dispatcher
├── README.md               # This file
├── Back_End/               # Core back-end logic
│   ├── constants.py        # Shared game constants (deck, state dims)
│   ├── envs.py             # Gymnasium-compliant poker environment (tournament logic)
│   ├── models.py           # PyTorch neural network models (Dueling DQN)
│   └── utils.py            # Utility functions (state encoding, replay buffer, model loading)
├── Front_End/              # User interface components
│   ├── card_utils.py       # Card rendering utilities
│   ├── main_ui.py          # Main Tkinter application logic
│   ├── seat_config.py      # Helper for managing seat types/defaults
│   └── human_action_handler.py # (Appears related to analysis, might need review/relocation)
├── Scripts/                # Runnable scripts
│   ├── train.py            # Script for training the agent
│   ├── simulate.py         # Script for running simulations
│   └── decision_analysis.py # Script for analyzing simulation logs
├── checkpoints/            # Default location for saved model checkpoints
└── Output_CSVs/            # Default location for simulation/analysis output CSVs


## Setup and Installation

**Prerequisites:**

* Python 3.x
* Git (optional, for cloning)

**Installation Steps:**

1.  **Clone the Repository (Optional):**
    ```bash
    git clone <your-repository-url>
    cd Package # Or your project root directory name
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # Windows:
    .\venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    Key dependencies based on the code include:
    * `torch`
    * `numpy`
    * `gymnasium`
    * `matplotlib` (used by analysis scripts, though `plot.py` is not present)

    Install them using pip:
    ```bash
    pip install torch numpy gymnasium matplotlib
    ```
    *(Consider creating a `requirements.txt` file for easier dependency management)*

## Usage (Command-Line Interface)

The primary way to interact with the project is through `main.py`.

* **General Help:**
    ```bash
    python main.py --help
    ```

* **Train (`train`)**
    * Purpose: Train the RL agent.
    * Help: `python main.py train --help`
    * Example (start new training for 100k episodes):
        ```bash
        python main.py train --episodes 100000
        ```
    * Example (resume from checkpoint):
        ```bash
        # Replace with your actual checkpoint path
        python main.py train --episodes 50000 --resume checkpoints/checkpoint_1000.pt
        ```

* **Simulate (`simulate`)**
    * Purpose: Run simulations using a trained agent. Requires a checkpoint file.
    * Help: `python main.py simulate --help`
    * Example (run 10 episodes with a specific checkpoint):
        ```bash
        # Replace with your actual checkpoint path
        python main.py simulate --checkpoint checkpoints/final_agent_model.pt --episodes 10
        ```
    * Example (with custom seat config and log name):
        ```bash
        python main.py simulate --checkpoint checkpoints/final_agent_model.pt --episodes 5 --seat_config "agent,random,model,empty,model,random" --detailed_log Output_CSVs/my_sim_log.csv
        ```

* **Analyze (`analyze`)**
    * Purpose: Analyze a detailed simulation log file. Requires the log file to exist.
    * Help: `python main.py analyze --help`
    * Example (analyze a specific log):
        ```bash
        # Requires the log file from a previous simulation
        python main.py analyze --detailed_log Output_CSVs/detailed_simulation_log.csv
        ```

* **UI (`ui`)**
    * Purpose: Launch the graphical user interface for playing poker.
    * Help: `python main.py ui --help` (Currently no specific arguments)
    * Example:
        ```bash
        python main.py ui
        ```
    *(Note: The UI might require a default checkpoint file like `checkpoints/final_agent_model.pt` to exist to function correctly with 'model' type opponents).*

## Dependencies

* [PyTorch](https://pytorch.org/)
* [NumPy](https://numpy.org/)
* [Gymnasium](https://gymnasium.farama.org/)
* [Matplotlib](https://matplotlib.org/) (for `decision_analysis.py`)

*(Add license information here if applicable)*
*(Add contribution guidelines here if applicable)*