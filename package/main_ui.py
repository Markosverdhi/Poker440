# main_ui_py_gym_v1.py
"""
Main UI for Poker Game using Gymnasium-compliant Environment.

Allows a human player to play against AI models (trained checkpoint, random).
Treats the human player as the 'agent' from the Gym Env perspective.

MODIFICATIONS (Gymnasium UI Integration - Prompts UI-2 to UI-7):
- Configuration (_start_game):
    - Identifies human seat.
    - Instantiates Gym Env (e.g., TrainFullPokerEnv) with human as agent_id.
    - Loads trained model checkpoint.
    - Creates and sets opponent policies (model, random) using env.set_opponent_policy.
- Game Start/Reset (_start_new_round):
    - Calls env.reset(), stores encoded state.
    - Updates UI using env helper methods and info dict.
    - Schedules first turn processing (_process_game_turn).
- Turn Logic (_process_game_turn):
    - Checks env.current_player_id.
    - If human's turn, gets legal actions via env.get_legal_actions_for_agent() and enables buttons.
    - If AI's turn, disables buttons and waits (AI moves handled within env.step).
- Human Action (_handle_human_action):
    - Gets action index from button click.
    - Calls env.step(action_idx).
    - Updates UI based on returned state/info (reflects human + opponent moves).
    - Checks for episode end (terminated/truncated).
    - Schedules next turn check.
- UI Update (_update_ui):
    - Fetches state info using env helper methods (get_stacks, get_pot, etc.)
      and the info dict returned by step/reset.
    - Updates labels for pot, community cards, stacks, player hand (human only).
    - Updates turn indicator.
    - Displays opponent actions from info['opponent_actions'].
- End of Round (_handle_round_end):
    - Accepts final info dict.
    - Extracts results (winners, stacks, showdown hands) from info.
    - Displays showdown hands and results message.
    - Prompts for next round.
- Added get_opponent_policy function (adapted from simulate.py).
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import os
import torch
import random
import json
import numpy as np
from collections import deque # May not be needed directly anymore

# Import Gymnasium-compliant environment and updated utils
try:
    # Use the Gym-compliant version of the environment (v5 expected)
    from envs import BaseFullPokerEnv, TrainFullPokerEnv, evaluate_hand
    from utils import encode_obs_eval # For model opponent policy
    from card_utils import render_card, render_hand, SUITS_UNICODE, RANKS
    from seat_config import SeatConfigManager
    # human_action_handler might be less relevant now as logic is in main UI
    # from human_action_handler import HumanActionHandler
except ImportError as e:
    print(f"ERROR: Ensure envs.py (v5), utils.py, card_utils.py, seat_config.py are available: {e}")
    exit()

# Assuming models.py is available
try:
    from models import BestPokerModel
except ImportError:
    print("ERROR: Ensure models.py is available.")
    exit()


# --- Constants ---
NUM_PLAYERS = 6
# Get action list/dim from env instance later
# ACTION_LIST = ['fold', 'call', 'check', 'bet_small', 'bet_big', 'all_in']
# STATE_DIM = 52 + 1 + (NUM_PLAYERS - 1) * 52 # 313
DEFAULT_CHECKPOINT_PREFIX = "checkpoints/checkpoint_"
# DEFAULT_STACK = 10000 # Initial stack set in env

# --- Helper: Opponent Policy Creation (Adapted from simulate.py) ---
def get_opponent_policy(opponent_type, agent_model, action_list, num_actions):
    """Creates a policy function for non-human opponents."""
    action_index_to_str = {i: s for i, s in enumerate(action_list)}

    if opponent_type == "model":
        if not agent_model: return get_opponent_policy("random", None, action_list, num_actions) # Fallback
        def policy_fn(obs_dict): # Expects dictionary observation
            if not isinstance(obs_dict, dict): return 'fold'
            legal_actions = obs_dict.get('legal_actions', [])
            if not legal_actions: return 'fold'
            # Use encode_obs_eval for opponent model inference
            state = encode_obs_eval(obs_dict, use_half_encoding=False)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad(): q_values = agent_model(state_tensor)
            q_values_np = q_values.squeeze().cpu().numpy(); sorted_indices = np.argsort(q_values_np)[::-1]
            for action_idx in sorted_indices:
                if 0 <= action_idx < num_actions:
                    action_str = action_index_to_str.get(action_idx)
                    if action_str and action_str in legal_actions: return action_str
            # Fallback
            if 'check' in legal_actions: return 'check'
            if 'call' in legal_actions: return 'call'
            if 'fold' in legal_actions: return 'fold'
            return random.choice(legal_actions) if legal_actions else 'fold'
        return policy_fn
    # --- Add other opponent types as needed ---
    # elif opponent_type == "variable": ...
    else: # Default to random
        def policy_fn(obs_dict):
            if not isinstance(obs_dict, dict): return 'fold'
            legal = obs_dict.get("legal_actions", [])
            return random.choice(legal) if legal else "fold"
        return policy_fn


class PokerApp:
    """ Main class for the Tkinter Poker Application (Gymnasium Adapted) """
    def __init__(self, root):
        self.root = root
        self.root.withdraw() # Hide the main root window initially

        self.env = None # Instantiated in _start_game
        self.agent_model = None # Loaded in _start_game
        self.seat_configs = {} # Stores type ('human', 'model', etc.)
        self.checkpoint_path = tk.StringVar(value="final_agent_model.pt") # Default checkpoint

        self.game_window = None
        self.config_window = None
        self.seat_config_manager = SeatConfigManager(num_players=NUM_PLAYERS)
        # self.human_action_handler = HumanActionHandler(ACTION_LIST) # Less needed now

        # UI Elements
        self.seat_frames = {}
        self.seat_status_labels = {}
        self.seat_action_labels = {}
        self.seat_stack_labels = {}
        self.seat_showdown_card_labels = {}
        self.player_card_labels = []
        self.community_card_labels = []
        self.pot_label = None
        self.current_bet_label = None
        self.turn_label = None
        self.action_buttons = {}
        self.player_hand_frame = None # Reference to update title

        # Game state tracking
        self.current_encoded_state = None # Stores the NumPy array state
        self.human_player_seat = -1 # Seat index of the human player
        self.action_list = [] # Will get from env
        self.num_actions = 0 # Will get from env
        self._action_string_to_idx = {} # For mapping human clicks to action index
        self.last_info = {} # Store info from last step/reset

        self._create_config_window()

    # --- Configuration and Setup ---

    def _validate_checkpoint_path(self, suffix_or_path):
        """Validates checkpoint path."""
        if not suffix_or_path: return None
        # Simple check if it looks like a path or just suffix
        if "/" not in suffix_or_path and "\\" not in suffix_or_path:
             # Assume suffix, prepend default dir if it exists
             default_dir = "checkpoints"
             if os.path.isdir(default_dir):
                  return os.path.join(default_dir, suffix_or_path)
        # Assume full path or relative path from CWD
        return suffix_or_path

    def _load_model(self, path, state_dim, num_actions):
        """Loads the PyTorch model."""
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", f"Checkpoint not found: {path}")
            return None
        try:
            model = BestPokerModel(input_dim=state_dim, num_actions=num_actions)
            checkpoint_data = torch.load(path, map_location=torch.device('cpu'))
            state_dict = None
            if isinstance(checkpoint_data, dict): state_dict = checkpoint_data.get('agent_state_dict', checkpoint_data.get('state_dict', checkpoint_data))
            else: raise TypeError("Unrecognized checkpoint format")
            if not state_dict: raise TypeError("Could not find state_dict in checkpoint")

            cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(cleaned_state_dict, strict=False)
            model.eval()
            print(f"Successfully loaded model from {path}")
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load checkpoint {path}: {e}")
            return None

    def _create_config_window(self):
        """Creates the initial configuration window."""
        # (UI layout code remains similar to original main_ui.py)
        if self.config_window: self.config_window.lift(); return
        self.config_window = tk.Toplevel(self.root); self.config_window.title("Poker Game Configuration"); self.config_window.protocol("WM_DELETE_WINDOW", self.root.quit)
        main_frame = ttk.Frame(self.config_window, padding="10"); main_frame.grid(row=0, column=0, sticky="nsew")
        ttk.Label(main_frame, text="Configure Seats:", font="-weight bold").grid(row=0, column=0, columnspan=2, pady=(0, 10))
        self.seat_vars = []
        options = self.seat_config_manager.get_options(); # Ensure 'human' is an option
        if 'human' not in options: options.insert(0,'human') # Add if missing
        for i in range(NUM_PLAYERS):
            ttk.Label(main_frame, text=f"Seat {i+1}:").grid(row=i+1, column=0, sticky=tk.W, padx=5)
            default_val = 'human' if i == 0 else 'model' # Default human in seat 1
            var = tk.StringVar(value=default_val); dropdown = ttk.OptionMenu(main_frame, var, default_val, *options); dropdown.grid(row=i+1, column=1, sticky="ew", padx=5); self.seat_vars.append(var)
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(row=NUM_PLAYERS + 1, column=0, columnspan=2, sticky="ew", pady=10)
        ttk.Label(main_frame, text="Model Checkpoint:").grid(row=NUM_PLAYERS + 2, column=0, sticky=tk.W, padx=5)
        checkpoint_entry = ttk.Entry(main_frame, textvariable=self.checkpoint_path, width=30); checkpoint_entry.grid(row=NUM_PLAYERS + 2, column=1, sticky="ew", padx=5)
        start_button = ttk.Button(main_frame, text="Start Game", command=self._start_game); start_button.grid(row=NUM_PLAYERS + 3, column=0, columnspan=2, pady=(10, 0))
        self.config_window.resizable(False, False)

    def _start_game(self):
        """ (Prompt UI-2) Validates config, loads model, creates env & game window, starts round. """
        # 1. Validate Seat Configuration (Ensure exactly one 'human')
        selected_types = [var.get() for var in self.seat_vars]
        human_count = selected_types.count('human')
        if human_count == 0: messagebox.showerror("Config Error", "No seat assigned as 'human'."); return
        if human_count > 1: messagebox.showerror("Config Error", "More than one seat assigned as 'human'."); return

        # 2. Store Config & Find Human Player
        self.seat_configs = {}
        self.human_player_seat = -1
        for i, seat_type in enumerate(selected_types):
             self.seat_configs[i] = seat_type
             if seat_type == "human":
                 self.human_player_seat = i # This is the agent_id for the env
        print(f"Seat Configurations: {self.seat_configs}, Human Seat (Agent ID): {self.human_player_seat}")

        # 3. Instantiate Environment with human as agent_id
        try:
             # Use Train env if you need its features, else Base
             self.env = TrainFullPokerEnv(
                 num_players=NUM_PLAYERS,
                 agent_id=self.human_player_seat, # Pass human seat as agent_id
                 render_mode="human" # Or None if not rendering via env.render()
             )
             self.action_list = self.env.action_list
             self.num_actions = self.env.action_space.n
             self._action_string_to_idx = {s: i for i, s in enumerate(self.action_list)}
             state_dim = self.env.observation_space.shape[0] # Get from env
        except Exception as e:
             messagebox.showerror("Error", f"Failed to create environment: {e}")
             return

        # 4. Validate and Load Checkpoint (if needed)
        self.agent_model = None
        needs_model = any(stype == "model" for stype in self.seat_configs.values())
        if needs_model:
             checkpoint_input = self.checkpoint_path.get()
             full_checkpoint_path = self._validate_checkpoint_path(checkpoint_input)
             if not full_checkpoint_path:
                 messagebox.showerror("Config Error", "A model checkpoint is required for 'model' opponents.")
                 self.env.close(); self.env = None; return
             self.agent_model = self._load_model(full_checkpoint_path, state_dim, self.num_actions)
             if not self.agent_model:
                 messagebox.showerror("Error", "Model loading failed. Cannot start with 'model' opponents.")
                 self.env.close(); self.env = None; return
        else:
             print("No 'model' opponents selected. Running without loading trained model.")

        # 5. Set Opponent Policies in Environment
        for i in range(NUM_PLAYERS):
             if i == self.human_player_seat: continue # Skip human player
             seat_type = self.seat_configs[i]
             # Pass necessary args to get_opponent_policy
             policy_func = get_opponent_policy(seat_type, self.agent_model, self.action_list, self.num_actions)
             self.env.set_opponent_policy(i, policy_func)
             print(f"Set Seat {i+1} policy to: {seat_type}")

        # 6. Close Config Window & Create Game Window
        if self.config_window: self.config_window.destroy(); self.config_window = None
        self._create_game_window()

        # 7. Start the first round
        self._start_new_round()


    def _create_game_window(self):
        """Creates the main game window UI elements."""
        # (UI layout code largely similar to original, ensure elements are stored in self.* attributes)
        if self.game_window: self.game_window.lift(); return
        self.game_window = tk.Toplevel(self.root); self.game_window.title("Poker Game (Gym Env)"); self.game_window.geometry("900x750"); self.game_window.protocol("WM_DELETE_WINDOW", self.root.quit)

        top_frame = ttk.Frame(self.game_window, padding=5); top_frame.pack(fill=tk.X)
        middle_frame = ttk.Frame(self.game_window, padding=10); middle_frame.pack(fill=tk.BOTH, expand=True)
        bottom_frame = ttk.Frame(self.game_window, padding=10); bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.current_bet_label = ttk.Label(top_frame, text="Current Bet: $0", font="-weight bold"); self.current_bet_label.pack(side=tk.LEFT, padx=20)
        self.turn_label = ttk.Label(top_frame, text="Turn: -", font="-weight bold", foreground="blue"); self.turn_label.pack(side=tk.RIGHT, padx=20)

        left_seats_frame = ttk.Frame(middle_frame, padding=5); left_seats_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), anchor='n')
        table_frame = ttk.Frame(middle_frame, padding=10); table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_seats_frame = ttk.Frame(middle_frame, padding=5); right_seats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0), anchor='n')

        self.seat_frames = {}; self.seat_status_labels = {}; self.seat_action_labels = {}; self.seat_stack_labels = {}; self.seat_showdown_card_labels = {}
        for i in range(NUM_PLAYERS):
            parent_frame = left_seats_frame if i < 3 else right_seats_frame
            seat_frame = ttk.LabelFrame(parent_frame, text=f"Seat {i+1}", padding=10); seat_frame.pack(pady=5, fill=tk.X, anchor='n'); self.seat_frames[i] = seat_frame
            status_txt = f"Type: {self.seat_configs.get(i, 'N/A').title()}" + (" (You)" if i == self.human_player_seat else "")
            status_label = ttk.Label(seat_frame, text=status_txt); status_label.pack(anchor=tk.W); self.seat_status_labels[i] = status_label
            stack_label = ttk.Label(seat_frame, text="Stack: $0"); stack_label.pack(anchor=tk.W); self.seat_stack_labels[i] = stack_label
            action_label = ttk.Label(seat_frame, text="Last Action: -", foreground="gray", wraplength=120); action_label.pack(anchor=tk.W); self.seat_action_labels[i] = action_label
            showdown_label = ttk.Label(seat_frame, text="", foreground="darkblue", font="-weight bold"); showdown_label.pack(anchor=tk.W, pady=(5,0)); self.seat_showdown_card_labels[i] = showdown_label

        community_frame = ttk.LabelFrame(table_frame, text="Community Cards", padding=10); community_frame.pack(pady=10, anchor='center')
        self.community_card_labels = []
        for _ in range(5): lbl = ttk.Label(community_frame, text="", font=("Courier", 14), relief="ridge", width=7, anchor="center", padding=5); lbl.pack(side=tk.LEFT, padx=3); self.community_card_labels.append(lbl)

        self.player_hand_frame = ttk.LabelFrame(table_frame, text="Your Hand (Seat ?)", padding=10); self.player_hand_frame.pack(pady=10, anchor='center')
        self.player_card_labels = []
        for _ in range(2): lbl = ttk.Label(self.player_hand_frame, text="", font=("Courier", 16, "bold"), relief="solid", width=7, anchor="center", padding=5); lbl.pack(side=tk.LEFT, padx=5); self.player_card_labels.append(lbl)

        self.pot_label = ttk.Label(table_frame, text="Pot: $0", font=("-weight bold", 14)); self.pot_label.pack(pady=(15, 5), anchor='center')

        action_frame = ttk.Frame(bottom_frame); action_frame.pack(pady=(0,10))
        self.action_buttons = {}
        # Use action_list from env
        button_actions = self.action_list[:] # Copy
        for action in button_actions: btn = ttk.Button(action_frame, text=action.replace('_', ' ').title(), width=10, state=tk.DISABLED, command=lambda a=action: self._handle_human_action(a)); btn.pack(side=tk.LEFT, padx=5); self.action_buttons[action] = btn

        control_frame = ttk.Frame(bottom_frame); control_frame.pack(pady=(5,0))
        reconfig_button = ttk.Button(control_frame, text="Reconfigure Seats", command=self._reconfigure); reconfig_button.pack(side=tk.LEFT, padx=10)
        exit_button = ttk.Button(control_frame, text="Exit Application", command=self.root.quit); exit_button.pack(side=tk.LEFT, padx=10)

    # --- Game Flow Logic ---

    def _start_new_round(self):
        """ (Prompt UI-3) Resets env, updates UI, schedules first turn check. """
        if not self.env: print("Error: Environment not initialized."); return
        print("\n--- Starting New Round (UI) ---")

        # Reset internal UI states
        self.last_info = {}
        for i in range(NUM_PLAYERS):
            if i in self.seat_action_labels: self.seat_action_labels[i].config(text="Last Action: -", foreground="gray")
            if i in self.seat_showdown_card_labels: self.seat_showdown_card_labels[i].config(text="")

        # Reset the environment
        self.current_encoded_state, info = self.env.reset()
        self.last_info = info # Store initial info

        # Initial UI Update
        self._update_ui() # Update based on info from reset

        # Schedule the first turn check
        self.root.after(100, self._process_game_turn)

    def _process_game_turn(self):
        """ (Prompt UI-4) Checks whose turn it is and enables human controls if needed. """
        if not self.env: return
        if self.env.round_over: # Don't process turns if round ended
             # End of round handled by _handle_human_action or _handle_round_end
             return

        current_player_id = self.env.current_player_id # Use property

        # Update turn label
        if current_player_id is not None:
             turn_text = f"Turn: Seat {current_player_id + 1}" + (" (You)" if current_player_id == self.human_player_seat else "")
             fg_color = "blue" if current_player_id == self.human_player_seat else "black"
             self.turn_label.config(text=turn_text, foreground=fg_color)
             # Highlight current player frame
             for i, frame in self.seat_frames.items():
                 relief_style = "solid" if i == current_player_id else "flat"
                 # frame.config(relief=relief_style) # Requires frame references stored
        else:
             self.turn_label.config(text="Turn: -", foreground="black")

        if current_player_id == self.human_player_seat:
            # Human's turn: Get legal actions and enable buttons
            legal_actions = self.env.get_legal_actions_for_agent() # Use helper
            print(f"Human turn (Seat {self.human_player_seat+1}). Legal actions: {legal_actions}") # Debug
            for action, button in self.action_buttons.items():
                if action in legal_actions:
                    button.config(state=tk.NORMAL)
                else:
                    button.config(state=tk.DISABLED)
            # UI waits for button press -> _handle_human_action
        else:
            # AI's turn (or game state issue): Disable human buttons
            for button in self.action_buttons.values():
                button.config(state=tk.DISABLED)
            # Optional: Indicate AI is "thinking" or waiting
            # Note: No env.step() call here for AI

    def _handle_human_action(self, action_str):
        """ (Prompt UI-5) Handles human button click, calls env.step, updates state/UI. """
        if not self.env or self.env.current_player_id != self.human_player_seat:
            print("Warning: Human action received, but not human's turn.")
            return

        print(f"Human (Seat {self.human_player_seat + 1}) chose action: {action_str}")

        # Map action string to index
        action_idx = self._action_string_to_idx.get(action_str, -1)
        if action_idx == -1:
            print(f"Error: Could not map action string '{action_str}' to index.")
            return

        # Disable buttons immediately
        for btn in self.action_buttons.values(): btn.config(state=tk.DISABLED)

        # Call environment step
        try:
            next_encoded_state, reward, terminated, truncated, info = self.env.step(action_idx)
            self.current_encoded_state = next_encoded_state # Store new state
            self.last_info = info # Store latest info dict
            done = terminated or truncated

            print(f"  env.step returned: reward={reward:.2f}, terminated={terminated}, truncated={truncated}")
            # Optionally print opponent actions from info
            # if 'opponent_actions' in info and info['opponent_actions']:
            #    print(f"  Opponent actions this step: {info['opponent_actions']}")

            # Update UI based on the state *after* the step (incl. opponent moves)
            self._update_ui()

            # Check for episode end
            if done:
                self._handle_round_end(info) # Pass final info
            else:
                # Schedule next turn check after a delay
                self.root.after(500, self._process_game_turn) # Delay allows UI to show opponent actions briefly

        except Exception as e:
            messagebox.showerror("Environment Error", f"An error occurred during env.step(): {e}")
            # Consider how to recover or reset
            self._reconfigure()


    def _update_ui(self):
        """ (Prompt UI-6) Updates all UI elements based on env state/info. """
        if not self.game_window or not self.env: return

        # Get state from helper methods or last info dict
        stacks = self.last_info.get('stacks', self.env.get_stacks(include_all=True)) # Use info first
        pot = self.last_info.get('pot', self.env.get_pot())
        community_cards = self.last_info.get('community_cards', self.env.get_community_cards())
        current_bets = self.last_info.get('current_bets', self.env.get_current_bets())
        human_hand = self.env.get_player_hand(self.human_player_seat) # Use helper for agent hand
        opponent_actions = self.last_info.get('opponent_actions', []) # Get opponent actions from info

        # Update Pot and Current Bet (Max bet this stage)
        self.pot_label.config(text=f"Pot: ${pot:.2f}")
        max_bet_this_stage = max(current_bets.values()) if current_bets else 0
        self.current_bet_label.config(text=f"Current Bet: ${max_bet_this_stage:.2f}")

        # Update Turn Indicator (handled in _process_game_turn)

        # Update Seat Information
        active_players_this_round = self.last_info.get('active_players', self.env.active_players_in_round) # Get active players if available
        for i in range(NUM_PLAYERS):
            stack = stacks.get(i, 0)
            status_text = f"Type: {self.seat_configs.get(i, 'N/A').title()}"
            if i == self.human_player_seat: status_text += " (You)"

            is_folded = (i not in active_players_this_round and i in self.env.hands) # Check if had hand but not active
            is_all_in = (stack <= 0 and i in active_players_this_round) # Check if active but no stack

            if self.seat_configs.get(i) == 'empty': status_text += " (Empty)"; stack = 0
            elif is_folded: status_text += " (Folded)"
            elif is_all_in: status_text += " (All-In)"

            if i in self.seat_status_labels: self.seat_status_labels[i].config(text=status_text)
            if i in self.seat_stack_labels: self.seat_stack_labels[i].config(text=f"Stack: ${stack:.2f}")

            # Update last action labels (clear first, then update based on info)
            # Only update opponents based on opponent_actions list for this step
            is_opponent = (i != self.human_player_seat)
            action_updated = False
            if is_opponent:
                 for opp_id, opp_action_str in opponent_actions:
                      if opp_id == i:
                           action_color = "gray" # Default
                           if opp_action_str == 'fold': action_color = 'red'
                           elif 'bet' in opp_action_str or opp_action_str == 'all_in': action_color = 'orange'
                           elif opp_action_str == 'call': action_color = 'blue'
                           elif opp_action_str == 'check': action_color = 'purple'
                           if i in self.seat_action_labels:
                                self.seat_action_labels[i].config(text=f"Action: {opp_action_str.title()}", foreground=action_color)
                                action_updated = True
                           break # Process only first action found for this opponent in the list
            # If opponent didn't act this step, maybe keep previous action or clear? Clear for now.
            # if is_opponent and not action_updated and i in self.seat_action_labels:
            #      self.seat_action_labels[i].config(text="Action: -", foreground="gray")
            # Human action label updated implicitly when buttons pressed/disabled

            # Clear showdown cards unless round is over (handled in _handle_round_end)
            if not self.env.round_over and i in self.seat_showdown_card_labels:
                 self.seat_showdown_card_labels[i].config(text="")


        # Update Community Cards
        for idx, label in enumerate(self.community_card_labels):
            if idx < len(community_cards):
                label.config(text=render_card(community_cards[idx]))
            else:
                label.config(text="")

        # Update Player Hand Display
        if self.human_player_seat != -1:
            self.player_hand_frame.config(text=f"Your Hand (Seat {self.human_player_seat + 1})")
            for idx, label in enumerate(self.player_card_labels):
                if human_hand and idx < len(human_hand):
                    label.config(text=render_card(human_hand[idx]))
                else:
                    label.config(text="") # Should have 2 cards

        self.root.update_idletasks() # Force UI update


    def _handle_round_end(self, final_info):
        """ (Prompt UI-7) Handles end of round display and prompts. """
        print("\n--- Round Ended (UI) ---")
        self.env.round_over = True # Ensure flag is set

        # Disable action buttons
        for btn in self.action_buttons.values(): btn.config(state=tk.DISABLED)
        self.turn_label.config(text="Turn: Round Over", foreground="darkred")

        # Extract info for display
        winners = final_info.get('winners', [])
        scores = final_info.get('scores', {})
        final_stacks = final_info.get('final_stacks', {})
        showdown_hands = final_info.get('showdown_hands', {}) # Hands involved in showdown
        agent_reward = final_info.get('agent_reward', 0) # Agent's reward for the round

        # --- Display Showdown Hands ---
        print(f"Debug Showdown Hands: {showdown_hands}")
        for pid, hand_list in showdown_hands.items():
             if hand_list and pid in self.seat_showdown_card_labels:
                  hand_str = render_hand(hand_list)
                  self.seat_showdown_card_labels[pid].config(text=f"Cards: {hand_str}")
             elif pid in self.seat_showdown_card_labels: # Should have hand if in showdown_hands
                  self.seat_showdown_card_labels[pid].config(text="Cards: Error")


        self._update_ui() # Update UI one last time with showdown cards etc.

        # --- Prepare results message ---
        result_message = f"Round Over!\nYour Reward: ${agent_reward:.2f}\n\n"
        pot_this_round = sum(self.env.round_total_bets.values()) # Recalculate pot for display
        result_message += f"Final Pot: ${pot_this_round:.2f}\n" # Display calculated pot

        if not winners: result_message += "No winner determined.\n"
        else:
             win_amount = pot_this_round / len(winners) if winners else 0
             result_message += f"Winner(s): Seat(s) {[w+1 for w in winners]} (${win_amount:.2f} each)\n"
             # Display scores/hands of showdown players
             result_message += "Showdown:\n"
             sorted_showdown_players = sorted(showdown_hands.keys())
             for pid in sorted_showdown_players:
                  hand_str = render_hand(showdown_hands.get(pid, ["?", "?"]))
                  score_desc = self._describe_hand_score(scores.get(pid)) # Use helper
                  result_message += f"  Seat {pid+1}: {hand_str} ({score_desc})\n"


        result_message += "\nFinal Stacks:\n"
        for pid in range(NUM_PLAYERS):
             if self.seat_configs.get(pid) != 'empty':
                 result_message += f"  Seat {pid+1}: ${final_stacks.get(pid, 0):.2f}\n"

        # --- Prompt for Next Action ---
        # Use 'after' to ensure UI updates before showing messagebox
        self.root.after(200, lambda msg=result_message: self._prompt_next_round(msg))


    def _prompt_next_round(self, result_message):
        """Shows the results message box and handles the response."""
        # Check for game over before prompting
        active_players_with_chips = [p for p, s in self.env.get_stacks(include_all=True).items() if s > 0 and self.seat_configs.get(p) != 'empty']
        if len(active_players_with_chips) <= 1:
             winner_id = active_players_with_chips[0] if active_players_with_chips else -1
             messagebox.showinfo("Game Over", f"Game Over! Seat {winner_id + 1} wins the match!\nReconfigure to play again.")
             self._reconfigure()
             return # Don't prompt for next round

        # Game not over, prompt user
        response = messagebox.askyesnocancel("Round Over", result_message + "\nStart a new round?")

        if response is True: # Yes
            self._start_new_round()
        elif response is False: # No
            self._reconfigure()
        else: # Cancel or closed box
            self.root.quit()


    def _describe_hand_score(self, score_tuple):
         """ Converts the hand score tuple from evaluate_hand into a readable string. """
         # (Code remains same as original main_ui.py)
         if isinstance(score_tuple, str): return score_tuple
         if not isinstance(score_tuple, tuple) or len(score_tuple) != 2: return "Unknown Hand"
         rank_map_rev = {i: r for i, r in enumerate(RANKS, start=2)}; rank_map_rev[14] = 'A'
         try:
             rank_val, tiebreaker = score_tuple; desc = ""
             if rank_val == 10: desc = "Royal Flush"
             elif rank_val == 9: desc = "Straight Flush"
             elif rank_val == 8: desc = "Four of a Kind"
             elif rank_val == 7: desc = "Full House"
             elif rank_val == 6: desc = "Flush"
             elif rank_val == 5: desc = "Straight"
             elif rank_val == 4: desc = "Three of a Kind"
             elif rank_val == 3: desc = "Two Pair"
             elif rank_val == 2: desc = "One Pair"
             elif rank_val == 1: desc = "High Card"
             else: return "Unknown Hand Rank"
             # Add details from tiebreaker (simplified)
             if rank_val in [9, 5, 10] and tiebreaker: high_card = rank_map_rev.get(tiebreaker[0], '?'); desc += f" ({high_card} high)"
             elif rank_val == 8 and tiebreaker: quad_rank = rank_map_rev.get(tiebreaker[0], '?'); desc += f" ({quad_rank}s)"
             elif rank_val == 7 and len(tiebreaker) >= 2: trip_rank = rank_map_rev.get(tiebreaker[0], '?'); pair_rank = rank_map_rev.get(tiebreaker[1], '?'); desc += f" ({trip_rank}s / {pair_rank}s)"
             elif rank_val == 4 and tiebreaker: trip_rank = rank_map_rev.get(tiebreaker[0], '?'); desc += f" ({trip_rank}s)"
             elif rank_val == 3 and len(tiebreaker) >= 2: p1 = rank_map_rev.get(tiebreaker[0], '?'); p2 = rank_map_rev.get(tiebreaker[1], '?'); desc += f" ({p1}s & {p2}s)"
             elif rank_val == 2 and tiebreaker: pair_rank = rank_map_rev.get(tiebreaker[0], '?'); desc += f" ({pair_rank}s)"
             elif rank_val == 1 and tiebreaker: high_card = rank_map_rev.get(tiebreaker[0], '?'); desc += f" ({high_card} high)"
         except Exception as e: print(f"Error describing score {score_tuple}: {e}"); return "Hand Score Error"
         return desc


    def _reconfigure(self):
        """Closes the game window and re-opens the configuration window."""
        if self.game_window: self.game_window.destroy(); self.game_window = None
        if self.env: self.env.close(); self.env = None # Close env
        # Reset game elements
        self.agent_model = None; self.seat_configs = {}; self.current_encoded_state = None; self.human_player_seat = -1; self.last_info = {}
        # Clear UI element storage
        self.seat_frames = {}; self.seat_status_labels = {}; self.seat_action_labels = {}; self.seat_stack_labels = {}; self.seat_showdown_card_labels = {}
        self.player_card_labels = []; self.community_card_labels = []; self.action_buttons = {}
        self._create_config_window()


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    # Optional: Apply theme
    # try: ttk.Style(root).theme_use('clam')
    # except tk.TclError: print("Themes not available.")
    app = PokerApp(root)
    root.mainloop()
