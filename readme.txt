PokerBotRLA reinforcement learning agent for playing No-Limit Texas Hold'em poker, built using PyTorch and Gymnasium. This project includes capabilities for training the agent, simulating games against various opponent types, analyzing agent decisions, and playing against the agent via a graphical user interface.FeaturesRL Agent Training: Train a poker agent using a Dueling DQN architecture (scripts/train.py). Supports resuming from checkpoints and configuring opponent strategies during training.Game Simulation: Simulate tournament-style poker games featuring the trained agent against configurable opponents (model-based, random, etc.) using scripts/simulate.py. Generates summary and detailed logs.Graphical User Interface: Play poker against the trained agent and other opponent types in a visual interface (scripts/main_ui.py).Decision Analysis: Analyze the agent's performance and decision-making based on simulation logs (analysis/decision_analysis.py).Plotting: Visualize training or simulation results (e.g., rewards) using analysis/plot.py.Project StructureThe codebase is organized into the following main directories:PokerBotRL/
├── main.py                 # Main entry point, command dispatcher
├── README.md               # This file
├── code_walkthrough.ipynb  # Detailed explanation and runnable examples
├── poker_rl_core/          # Core logic package (environment, models, utils)
│   ├── __init__.py
│   ├── envs.py
│   ├── models.py
│   ├── utils.py
│   ├── card_utils.py
│   └── seat_config.py
├── scripts/                # Runnable scripts (train, simulate, ui)
│   ├── train.py
│   ├── simulate.py
│   └── main_ui.py
├── analysis/               # Analysis tools (plotting, decision analysis)
│   ├── __init__.py
│   ├── decision_analysis.py
│   ├── human_action_handler.py
│   └── plot.py
├── checkpoints/            # Default location for saved model checkpoints
└── *.csv                   # Output files from simulation/analysis (e.g., detailed_simulation_log.csv)
For a detailed explanation of each file, please refer to code_walkthrough.ipynb.Setup and InstallationPrerequisites:Python 3.xGit (optional, for cloning)Clone the Repository (Optional):git clone <your-repository-url>
cd PokerBotRL
Create a Virtual Environment (Recommended):python -m venv venv
# Activate the environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
Install Dependencies:It's good practice to create a requirements.txt file. Based on the code, key dependencies are:torch
numpy
gymnasium
matplotlib
Install them using pip:pip install torch numpy gymnasium matplotlib
# Or if you create requirements.txt:
# pip install -r requirements.txt
Usage (Command-Line Interface)The primary way to interact with the project is through main.py.General Help:To see all available commands and general options:python main.py --help
Commands:Train (train)Purpose: Train the RL agent.Help: python main.py train --helpExample (start new training for 100k episodes):python main.py train --episodes 100000
Example (resume from checkpoint):# Replace with your actual checkpoint path
python main.py train --episodes 50000 --resume checkpoints/checkpoint_1000.pt
Simulate (simulate)Purpose: Run simulations using a trained agent. Requires a checkpoint file.Help: python main.py simulate --helpExample (run 10 episodes):# Replace with your actual checkpoint path
python main.py simulate --checkpoint checkpoints/final_agent_model.pt --episodes 10
Example (with custom seat config and log name):python main.py simulate --checkpoint checkpoints/final_agent_model.pt --episodes 5 --seat_config "agent,random,model,empty,model,random" --detailed_log my_sim_log.csv
Analyze (analyze)Purpose: Analyze a detailed simulation log file. Requires the log file to exist.Help: python main.py analyze --helpExample (analyze default log):# Requires detailed_simulation_log.csv from a previous simulation
python main.py analyze --detailed_log detailed_simulation_log.csv
UI (ui)Purpose: Launch the graphical user interface for playing poker.Help: python main.py ui --help (Currently no specific arguments)Example:python main.py ui
(Note: The UI might require a default checkpoint file like checkpoints/final_agent_model.pt to exist to function correctly).Code WalkthroughFor a more detailed guide through the code structure and runnable examples of the commands, please see the code_walkthrough.ipynb notebook.DependenciesPyTorchGymnasium (formerly OpenAI Gym)NumPyMatplotlib (for plot.py)(Add license information here if applicable)(Add contribution guidelines here if applicable)