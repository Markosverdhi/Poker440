# filename: package/main_ui.py
"""
Main UI for Poker Game using Gymnasium-compliant Environment (Tournament Structure).

MODIFIED (Round Transition Fix v2):
- Refined _process_game_turn logic when current_player_id is None to ensure
  _step_env(-1) is called reliably to trigger the start of the next round.

MODIFIED (Refactoring):
- Removed local `_load_model` and `get_opponent_policy` functions.
- Imported `load_agent_model` and `get_opponent_policy` from `utils.py`.
- Updated `_start_game` to use imported functions.

MODIFIED (Empty Seat Handling):
- Updated _start_game to pass seat_configs to the TrainFullPokerEnv constructor.
- Updated _update_ui to visually gray out and clear info for 'Empty' seats.

MODIFIED (Layout & Showdown Fixes):
- Moved Pot display below player hand in the center.
- Added a dedicated multi-line label (`showdown_overview_label`) below the pot
  for displaying round winner/hand summary text.
- Modified `_display_round_results` to display ALL dealt hands and populate overview.
- Modified `_update_ui` to clear overview label when round is not over.

MODIFIED (Fix Human ID Detection):
- Corrected the loop in `_start_game` to properly identify the human player's
  seat index by removing an erroneous semicolon and ensuring correct indentation.

MODIFIED (UI Size Adjustment & Error Fix):
- Increased `seat_height` in `_create_game_window` to make player info boxes taller.
- Added a None check for `current_player_id` in `_handle_ai_action` to prevent TypeError.

MODIFIED (Community Card Debugging):
- Added a print statement in `_update_ui` to log the community cards being received
  from the environment's info dictionary just before rendering. (Can be commented out later)
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import os
import torch
import random
import json
import numpy as np
import time # For potential delays

# --- Attempt to import necessary custom modules ---
try:
    # Assuming these modules are in the same directory or accessible via PYTHONPATH
    from Back_End.envs import TrainFullPokerEnv, evaluate_hand # Use the updated envs.py
    # *** Import refactored functions from utils ***
    from Back_End.utils import encode_obs_eval, load_agent_model, get_opponent_policy
    from Back_End.constants import NEW_STATE_DIM
    from .card_utils import render_card_ascii, render_hand_for_labels, render_community_cards_for_labels, render_hand
    from .seat_config import SeatConfigManager
except ImportError as e:
    # Provide a more informative error message if imports fail
    print(f"ERROR: Critical modules not found or import failed. Ensure 'envs.py', 'utils.py', 'card_utils.py', 'seat_config.py', 'models.py' are available and updated.")
    print(f"Import Error Details: {e}")
    # Use tkinter to show the error if possible, otherwise exit
    try:
        root = tk.Tk()
        root.withdraw() # Hide the main window
        messagebox.showerror("Import Error", f"Critical modules not found or failed to import. Ensure required files are available and updated.\n\nDetails: {e}\n\nApplication will now exit.")
        root.destroy()
    except tk.TclError:
        pass # If tkinter itself fails, just exit
    exit(1) # Exit with an error code

# Assuming models.py is available and updated for NEW_STATE_DIM
try:
    from Back_End.models import BestPokerModel
except ImportError:
    print("ERROR: 'models.py' not found. Ensure it is available.")
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Import Error", "Module 'models.py' not found. Application will now exit.")
        root.destroy()
    except tk.TclError:
        pass
    exit(1)


# --- Constants ---
NUM_PLAYERS = 6           # Total number of seats at the table
STATE_DIM = NEW_STATE_DIM # Expected dimension of the encoded state from utils.py
MONOSPACE_FONT = ("Consolas", 10) # Preferred font for card rendering
EMPTY_SEAT_COLOR = "gray" # Color for text in empty seats

# --- REMOVED get_opponent_policy function (now imported from utils) ---

class PokerApp:
    """ Main class for the Tkinter Poker Application (Tournament Adapted) """
    def __init__(self, root):
        """ Initialize the application, set up variables, and create the config window. """
        self.root = root
        self.root.withdraw() # Hide the main Tkinter window initially

        # Game state variables
        self.env = None                 # The poker environment instance
        self.agent_model = None         # Loaded AI model (if used)
        self.seat_configs = {}          # Dictionary mapping seat index to player type ('human', 'model', etc.)
        self.checkpoint_path = tk.StringVar(value="checkpoints/final_agent_model.pt") # Path to AI model checkpoint
        self.current_encoded_state = None # Last encoded state received from the environment
        self.human_player_seat = -1     # Index of the human player's seat
        self.action_list = []           # List of possible action strings from the env
        self.num_actions = 0            # Number of possible actions
        self._action_string_to_idx = {} # Mapping from action string to action index
        self.last_info = {}             # Last info dictionary received from env.step() or env.reset()
        self.tournament_running = False # Flag indicating if a tournament is active
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Determine device
        print(f"Using device: {self.device}")

        # UI Window references
        self.game_window = None         # Reference to the main game Toplevel window
        self.config_window = None       # Reference to the configuration Toplevel window

        # UI Widget references (organized by type)
        self.seat_frames = {}           # Frames for each player seat area
        self.seat_status_labels = {}    # Labels showing player type/status (e.g., "Model (BTN)")
        self.seat_action_labels = {}    # Labels showing the last action taken by a player
        self.seat_stack_labels = {}     # Labels showing player stack size and current bet
        self.seat_showdown_card_labels = {} # Labels within seat frames to show cards at showdown
        self.player_card_labels = []    # Labels for the human player's hand cards
        self.community_card_labels = [] # Labels for the community cards on the table
        self.pot_label = None           # Label displaying the current pot size
        self.turn_label = None          # Label indicating whose turn it is
        self.action_buttons = {}        # Dictionary mapping action strings to action buttons
        self.player_hand_frame = None   # Frame containing the human player's card labels
        self.status_bar = None          # Label at the bottom for status messages
        self.showdown_overview_label = None # Label in the center for round results summary

        # Configuration helper
        self.seat_config_manager = SeatConfigManager(num_players=NUM_PLAYERS) # Assuming SeatConfigManager exists

        # Start by showing the configuration window
        self._create_config_window()

    # --- Configuration and Setup Methods ---

    def _validate_checkpoint_path(self, suffix_or_path):
        """ Validates the checkpoint path, prepending a default directory if only a filename is given. """
        if not suffix_or_path:
            return None
        # If it looks like just a filename (no directory separators)
        if "/" not in suffix_or_path and "\\" not in suffix_or_path:
            default_dir = "checkpoints"
            if os.path.isdir(default_dir):
                return os.path.join(default_dir, suffix_or_path)
            else:
                # If default dir doesn't exist, return the original path and let load fail later
                print(f"Warning: Default checkpoint directory '{default_dir}' not found.")
                return suffix_or_path
        # Otherwise, assume it's a full or relative path
        return suffix_or_path

    # --- REMOVED _load_model function (now imported from utils) ---

    def _create_config_window(self):
        """ Creates the initial configuration window for setting up the game. """
        if self.config_window and self.config_window.winfo_exists():
            self.config_window.lift() # Bring existing window to front
            return

        self.config_window = tk.Toplevel(self.root)
        self.config_window.title("Poker Game Configuration")
        # Ensure closing this window exits the app if the game hasn't started
        self.config_window.protocol("WM_DELETE_WINDOW", self.root.quit)

        main_frame = ttk.Frame(self.config_window, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(main_frame, text="Configure Seats:", font="-weight bold").grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")

        # Create dropdown menus for each seat configuration
        self.seat_vars = []
        options = self.seat_config_manager.get_options() # Get available types ('model', 'random', etc.)
        # Ensure 'human' is always an option, handle potential duplicates if already in options
        if 'human' not in options: options.insert(0, 'human')
        else: options.remove('human'); options.insert(0, 'human') # Move to front

        for i in range(NUM_PLAYERS):
            ttk.Label(main_frame, text=f"Seat {i+1}:").grid(row=i+1, column=0, sticky=tk.W, padx=5, pady=2)
            # Default first seat to human, others to model
            default_val = 'human' if i == 0 else 'model'
            if default_val not in options: default_val = options[0] # Fallback if default isn't valid

            var = tk.StringVar(value=default_val)
            # Use unique list of options for the dropdown
            dropdown = ttk.OptionMenu(main_frame, var, default_val, *options)
            dropdown.grid(row=i+1, column=1, sticky="ew", padx=5, pady=2)
            self.seat_vars.append(var)

        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(row=NUM_PLAYERS + 1, column=0, columnspan=2, sticky="ew", pady=10)

        # Checkpoint path entry
        ttk.Label(main_frame, text="Model Checkpoint:").grid(row=NUM_PLAYERS + 2, column=0, sticky=tk.W, padx=5, pady=2)
        checkpoint_entry = ttk.Entry(main_frame, textvariable=self.checkpoint_path, width=40)
        checkpoint_entry.grid(row=NUM_PLAYERS + 2, column=1, sticky="ew", padx=5, pady=2)

        # Start game button
        start_button = ttk.Button(main_frame, text="Start Game", command=self._start_game)
        start_button.grid(row=NUM_PLAYERS + 3, column=0, columnspan=2, pady=(15, 5))

        self.config_window.resizable(False, False)
        self.config_window.update_idletasks() # Ensure window size is calculated
        # Center the window (optional)
        # x = self.root.winfo_screenwidth() // 2 - self.config_window.winfo_width() // 2
        # y = self.root.winfo_screenheight() // 2 - self.config_window.winfo_height() // 2
        # self.config_window.geometry(f'+{x}+{y}')

    def _start_game(self):
        """ Validates configuration, initializes the environment and model, and starts the game loop. """
        # --- ** FIX APPLIED HERE (Human ID Detection) ** ---
        selected_types = [var.get() for var in self.seat_vars]

        # Validate exactly one human player
        human_count = selected_types.count('human')
        if human_count == 0:
            messagebox.showerror("Configuration Error", "No seat assigned as 'human'. Please select one seat for the human player.")
            return
        if human_count > 1:
            messagebox.showerror("Configuration Error", "More than one seat assigned as 'human'. Please select only one seat for the human player.")
            return

        # Store seat configurations and find the human player's seat index
        self.seat_configs = {}
        self.human_player_seat = -1 # Initialize
        for i, seat_type in enumerate(selected_types):
            self.seat_configs[i] = seat_type # Store the config type for seat i
            if seat_type == "human":
                self.human_player_seat = i # Found the human player

        print(f"Seat Configurations: {self.seat_configs}")
        print(f"Human Player assigned to Seat Index: {self.human_player_seat} (Seat {self.human_player_seat + 1})")

        # Safety check (should not be needed due to validation above, but good practice)
        if self.human_player_seat == -1:
            messagebox.showerror("Internal Error", "Failed to identify human player seat index after configuration.")
            return
        # --- ** END FIX ** ---

        # --- Initialize Environment ---
        try:
            # *** MODIFICATION: Pass seat_config to environment ***
            self.env = TrainFullPokerEnv(
                num_players=NUM_PLAYERS,
                agent_id=self.human_player_seat,
                render_mode="human",
                seat_config=self.seat_configs # Pass the configuration
            )
            self.action_list = self.env.action_list
            self.num_actions = self.env.action_space.n
            self._action_string_to_idx = {s: i for i, s in enumerate(self.action_list)}
            print(f"Environment action list: {self.action_list}")

            # Validate observation space dimension
            if self.env.observation_space.shape[0] != STATE_DIM:
                messagebox.showerror("Configuration Error", f"Environment observation space dimension ({self.env.observation_space.shape[0]}) does not match the expected STATE_DIM ({STATE_DIM}). Check 'utils.py' and environment definition.")
                self.env.close()
                self.env = None
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create poker environment: {e}")
            self.env = None
            return

        # --- Load AI Model (if needed) ---
        self.agent_model = None
        needs_model = any(stype == "model" or stype == "variable" for stype in self.seat_configs.values())
        if needs_model:
            checkpoint_input = self.checkpoint_path.get()
            full_checkpoint_path = self._validate_checkpoint_path(checkpoint_input)

            if not full_checkpoint_path: # Check if path is invalid/empty after validation
                 messagebox.showerror("Configuration Error", f"Model checkpoint path is invalid or empty.")
                 self.env.close(); self.env = None; return

            # *** Use imported load_agent_model ***
            self.agent_model = load_agent_model(full_checkpoint_path, self.num_actions, self.device)
            if not self.agent_model:
                # load_agent_model prints errors, just need to stop
                self.env.close()
                self.env = None
                return
        else:
            print("No 'model' or 'variable' opponents selected, AI model not loaded.")

        # --- Set Opponent Policies in Environment ---
        for i in range(NUM_PLAYERS):
            # Skip human player and empty seats
            seat_type = self.seat_configs.get(i)
            if i == self.human_player_seat or seat_type == 'empty':
                continue

            # *** Use imported get_opponent_policy ***
            policy_func = get_opponent_policy(
                opponent_type=seat_type,
                agent_model=self.agent_model,
                action_list=self.action_list,
                num_actions=self.num_actions,
                device=self.device
            )
            try:
                self.env.set_opponent_policy(i, policy_func)
                print(f"Set Seat {i+1} (Index {i}) policy to: {seat_type}")
            except Exception as e:
                 messagebox.showerror("Error", f"Failed to set policy for opponent at seat {i+1}: {e}")
                 self.env.close(); self.env = None; return


        # --- Transition to Game Window ---
        if self.config_window:
            self.config_window.destroy()
            self.config_window = None

        self._create_game_window() # Build the main game UI
        self.tournament_running = True
        if self.status_bar: self.status_bar.config(text="Starting new tournament...")

        # --- Reset Environment and Start Game Loop ---
        try:
            # Reset the environment to get the initial state and info
            # Env reset now uses the seat_config passed during init
            self.current_encoded_state, self.last_info = self.env.reset()
            # Check for immediate errors after reset
            if isinstance(self.last_info, dict) and self.last_info.get("error"):
                messagebox.showerror("Error", f"Environment reset failed: {self.last_info['error']}")
                self._reconfigure() # Go back to config screen
                return

            self._update_ui() # Update UI with initial state
            # Schedule the first game turn processing
            self.root.after(100, self._process_game_turn)
        except Exception as e:
            messagebox.showerror("Error", f"Failed during initial environment reset: {e}")
            self._reconfigure() # Go back to config if reset fails

    def _create_game_window(self):
        """ Creates the main game window with all UI elements. """
        if self.game_window and self.game_window.winfo_exists():
            self.game_window.lift() # Bring existing window to front
            return

        self.game_window = tk.Toplevel(self.root)
        self.game_window.title("Poker Game (Tournament Mode)")
        self.game_window.geometry("950x800") # Adjust size as needed
        self.game_window.protocol("WM_DELETE_WINDOW", self.root.quit) # Closing game exits app

        # --- Styling ---
        style = ttk.Style()
        try:
            # Attempt to use a specific theme
            style.theme_use('clam') # Or 'alt', 'default', 'classic'
        except tk.TclError:
            print("Note: 'clam' theme not available, using default.")

        try:
            # Configure custom styles for different labels
            style.configure("Card.TLabel", font=MONOSPACE_FONT, padding=2, anchor="center", justify="center", borderwidth=1, relief="solid")
            style.configure("PlayerHand.TLabel", font=(MONOSPACE_FONT[0], 12, "bold"), padding=3, anchor="center", justify="center", borderwidth=2, relief="solid")
            style.configure("Showdown.TLabel", font=MONOSPACE_FONT, padding=1, anchor="nw", justify="left", foreground="darkblue") # For cards shown under seats
            style.configure("Overview.TLabel", padding=(5, 3), anchor="nw", justify="left", foreground="black", background="#f0f0f0", relief="groove", borderwidth=1) # For round summary
            style.configure("Bold.TLabel", font="-weight bold")
            style.configure("Turn.TLabel", font=("", 12, "bold")) # Style for the turn indicator
            style.configure("Pot.TLabel", font=("", 12, "bold")) # Style for the pot label
            # Style for empty seat labels
            style.configure("Empty.TLabel", foreground=EMPTY_SEAT_COLOR)

        except tk.TclError as e:
            print(f"Warning: Failed to set custom font '{MONOSPACE_FONT}'. Using default styles. Error: {e}")
            # Provide basic fallbacks if font fails
            style.configure("Card.TLabel", padding=2, anchor="center", justify="center", borderwidth=1, relief="solid")
            style.configure("PlayerHand.TLabel", font="-size 12 -weight bold", padding=3, anchor="center", justify="center", borderwidth=2, relief="solid")
            style.configure("Showdown.TLabel", padding=1, anchor="nw", justify="left", foreground="darkblue")
            style.configure("Overview.TLabel", padding=(5, 3), anchor="nw", justify="left", foreground="black", background="#f0f0f0", relief="groove", borderwidth=1)
            style.configure("Bold.TLabel", font="-weight bold")
            style.configure("Turn.TLabel", font="-size 12 -weight bold")
            style.configure("Pot.TLabel", font="-size 12 -weight bold")
            style.configure("Empty.TLabel", foreground=EMPTY_SEAT_COLOR)


        # --- Main Layout Frames ---
        top_frame = ttk.Frame(self.game_window, padding=5)
        top_frame.pack(fill=tk.X, side=tk.TOP)

        middle_frame = ttk.Frame(self.game_window, padding=10)
        middle_frame.pack(fill=tk.BOTH, expand=True)

        bottom_frame = ttk.Frame(self.game_window, padding=10)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # --- Top Frame Content (Turn Indicator) ---
        self.turn_label = ttk.Label(top_frame, text="Turn: -", style="Turn.TLabel", foreground="black")
        self.turn_label.pack(side=tk.LEFT, padx=20, pady=5)

        # --- Middle Frame Content (Seats and Table) ---
        # Create frames for left seats, center table, and right seats
        left_seats_frame = ttk.Frame(middle_frame, padding=5)
        left_seats_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), anchor='n')

        table_frame = ttk.Frame(middle_frame, padding=10) # Center area
        table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, anchor='center')

        right_seats_frame = ttk.Frame(middle_frame, padding=5)
        right_seats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0), anchor='n')

        # Create individual seat frames and labels
        self.seat_frames = {}
        self.seat_status_labels = {}
        self.seat_action_labels = {}
        self.seat_stack_labels = {}
        self.seat_showdown_card_labels = {}
        seat_width = 170 # Adjust width as needed
        # --- MODIFICATION: Increased seat height ---
        seat_height = 140 # Increased from 110 to make boxes taller

        for i in range(NUM_PLAYERS):
            # Determine parent frame (left or right)
            parent_frame = left_seats_frame if i < (NUM_PLAYERS / 2) else right_seats_frame

            # Create the main frame for the seat
            seat_frame = ttk.LabelFrame(parent_frame, text=f"Seat {i+1}", padding=10, width=seat_width, height=seat_height)
            seat_frame.pack(pady=5, fill=tk.X, anchor='n')
            seat_frame.pack_propagate(False) # Prevent frame from shrinking to content
            self.seat_frames[i] = seat_frame

            # Create labels within the seat frame
            status_label = ttk.Label(seat_frame, text="Status: -", wraplength=seat_width-20)
            status_label.pack(anchor=tk.NW, fill=tk.X)
            self.seat_status_labels[i] = status_label

            stack_label = ttk.Label(seat_frame, text="Stack: $0")
            stack_label.pack(anchor=tk.NW, fill=tk.X)
            self.seat_stack_labels[i] = stack_label

            action_label = ttk.Label(seat_frame, text="Last Action: -", foreground="gray", wraplength=seat_width-20)
            action_label.pack(anchor=tk.NW, fill=tk.X)
            self.seat_action_labels[i] = action_label

            # Label specifically for showing cards at showdown (initially empty)
            # Increased height allows more space for this label
            showdown_label = ttk.Label(seat_frame, text="", style="Showdown.TLabel", justify="left", wraplength=seat_width-20)
            showdown_label.pack(anchor=tk.NW, pady=(5,0), fill=tk.X, expand=True) # Allow vertical expansion
            self.seat_showdown_card_labels[i] = showdown_label

        # --- Center Table Elements (Community Cards, Player Hand, Pot) ---
        community_frame = ttk.LabelFrame(table_frame, text="Community Cards", padding=10)
        community_frame.pack(pady=10, anchor='center')
        self.community_card_labels = []
        for _ in range(5): # 5 community cards max
            lbl = ttk.Label(community_frame, text=" ", style="Card.TLabel", width=7, anchor="center")
            lbl.pack(side=tk.LEFT, padx=3)
            self.community_card_labels.append(lbl)

        # Player Hand Frame (for human player)
        self.player_hand_frame = ttk.LabelFrame(table_frame, text="Your Hand (Seat ?)", padding=10)
        self.player_hand_frame.pack(pady=10, anchor='center')
        self.player_card_labels = []
        for _ in range(2): # 2 hole cards
            lbl = ttk.Label(self.player_hand_frame, text=" ", style="PlayerHand.TLabel", width=7, anchor="center")
            lbl.pack(side=tk.LEFT, padx=5)
            self.player_card_labels.append(lbl)

        # Pot Label
        self.pot_label = ttk.Label(table_frame, text="Pot: $0", style="Pot.TLabel")
        self.pot_label.pack(pady=(10, 5), anchor='center')

        # Showdown Overview Label (for winner summary)
        self.showdown_overview_label = ttk.Label(table_frame, text="", style="Overview.TLabel", wraplength=450, justify="left")
        self.showdown_overview_label.pack(pady=(5,10), fill=tk.X, anchor='center', expand=False) # Don't expand vertically

        # --- Bottom Frame Content (Action Buttons, Controls, Status Bar) ---
        action_frame = ttk.Frame(bottom_frame)
        action_frame.pack(pady=(0,10)) # Pack action buttons first

        self.action_buttons = {}
        # Use action_list from env if available, otherwise default
        button_actions = self.action_list if self.action_list else ['fold', 'call', 'check', 'bet_small', 'bet_big', 'all_in']
        # Define a consistent order for buttons
        action_order = ['fold', 'check', 'call', 'bet_small', 'bet_big', 'all_in']
        display_actions = [a for a in action_order if a in button_actions]

        for action in display_actions:
            btn_text = action.replace('_', ' ').title() # Format text (e.g., 'bet_small' -> 'Bet Small')
            btn = ttk.Button(action_frame, text=btn_text, width=10, state=tk.DISABLED,
                             command=lambda a=action: self._handle_human_action(a))
            btn.pack(side=tk.LEFT, padx=5)
            self.action_buttons[action] = btn

        # Control buttons (New Game, Exit)
        control_frame = ttk.Frame(bottom_frame)
        control_frame.pack(pady=(5,0))

        reconfig_button = ttk.Button(control_frame, text="New Game / Reconfigure", command=self._reconfigure)
        reconfig_button.pack(side=tk.LEFT, padx=10)

        exit_button = ttk.Button(control_frame, text="Exit Application", command=self.root.quit)
        exit_button.pack(side=tk.LEFT, padx=10)

        # Status Bar
        self.status_bar = ttk.Label(bottom_frame, text="Welcome! Configure seats and start the game.", relief=tk.SUNKEN, anchor=tk.W, padding=2)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(10,0)) # Pack status bar last at the bottom

    # --- Game Flow Logic ---

    def _process_game_turn(self):
        """ Checks whose turn it is and either enables human actions or triggers AI action. """
        if not self.env or not self.tournament_running:
            # print("Debug: Process game turn called but env/tournament not ready.") # Reduced verbosity
            return

        # Check if the tournament ended in the environment's state
        # Use last_info as env state might not be updated yet if tournament ended in last step
        if self.last_info.get('terminated', False) or self.last_info.get('truncated', False) or \
           (hasattr(self.env, 'tournament_over') and self.env.tournament_over):
            print("Debug: Tournament detected as over during turn processing.")
            # Ensure UI update happens before potential blocking message box
            # Pass the *last* info dict received from the step that ended the tournament
            self.root.after(50, lambda info=self.last_info: self._handle_tournament_end(info))
            return

        # Get current player ID from the environment
        current_player_id = None
        try:
            # Added try-except in case env state is unstable
            current_player_id = self.env.current_player_id if hasattr(self.env, 'current_player_id') else None
        except Exception as e:
            print(f"Error accessing env.current_player_id: {e}")
            # Potentially handle error state, e.g., reconfigure
            self.root.after(50, self._reconfigure)
            return


        # Update turn indicator label and highlight active player frame
        if current_player_id is not None and self.seat_configs.get(current_player_id) != 'empty':
            is_human_turn = (current_player_id == self.human_player_seat)
            seat_type_str = self.seat_configs.get(current_player_id, '?').title()
            turn_text = f"Turn: Seat {current_player_id + 1}" + (" (You)" if is_human_turn else f" ({seat_type_str})")
            fg_color = "blue" if is_human_turn else "black"
            if self.turn_label.winfo_exists(): self.turn_label.config(text=turn_text, foreground=fg_color)

            # Highlight the active player's frame
            for i, frame in self.seat_frames.items():
                 if frame.winfo_exists():
                     # Only highlight if not empty
                     is_current = (i == current_player_id and self.seat_configs.get(i) != 'empty')
                     frame.config(relief="sunken" if is_current else ("flat" if self.seat_configs.get(i) == 'empty' else "groove"))
        else:
            # Handle case where current_player_id is None or points to an empty seat
            if self.turn_label.winfo_exists(): self.turn_label.config(text="Turn: -", foreground="black")
            for i, frame in self.seat_frames.items():
                 if frame.winfo_exists():
                     frame.config(relief="flat" if self.seat_configs.get(i) == 'empty' else "groove")


        # Check if it's the human player's turn
        if current_player_id == self.human_player_seat:
            if self.status_bar.winfo_exists(): self.status_bar.config(text="Your turn. Choose an action.")
            try:
                # Get legal actions specifically for the human agent
                legal_actions = self.env.get_legal_actions_for_agent()
                # print(f"Debug: Human legal actions: {legal_actions}") # Reduced verbosity

                # *** Add check for no legal actions when it's human turn (e.g., after fold) ***
                if not legal_actions:
                     print("Debug: Human turn, but no legal actions found. Triggering step.")
                     # Treat as if current_player_id was None to advance state
                     current_player_id = None # Force into the 'else' block below
                     # Fall through to the else block...
                else:
                     # Enable corresponding buttons
                     for action, button in self.action_buttons.items():
                          if button.winfo_exists():
                              button.config(state=tk.NORMAL if action in legal_actions else tk.DISABLED)

            except AttributeError:
                 print("Error: Environment does not have 'get_legal_actions_for_agent' method.")
                 messagebox.showerror("Error", "Environment is missing the required 'get_legal_actions_for_agent' method.")
                 for button in self.action_buttons.values():
                      if button.winfo_exists(): button.config(state=tk.DISABLED) # Disable all
            except Exception as e:
                print(f"Error getting legal actions for human: {e}")
                messagebox.showerror("Error", f"Could not get legal actions: {e}")
                for button in self.action_buttons.values():
                     if button.winfo_exists(): button.config(state=tk.DISABLED) # Disable all

        # Check AI turn *after* potential modification of current_player_id above
        if current_player_id is not None and current_player_id != self.human_player_seat and self.seat_configs.get(current_player_id) != 'empty': # It's an AI player's turn
            if self.status_bar.winfo_exists(): self.status_bar.config(text=f"Waiting for Seat {current_player_id + 1}...")
            # Disable all human action buttons
            for button in self.action_buttons.values():
                 if button.winfo_exists(): button.config(state=tk.DISABLED)

            # Schedule the AI action after a short delay (for visual feedback)
            ai_delay = 300 # milliseconds
            self.root.after(ai_delay, self._handle_ai_action)

        elif current_player_id is None: # Covers case where it was None initially, or set to None above
             # current_player_id is None (between rounds or end)
             print("Debug: current_player_id is None. Triggering step for potential new round start.")
             if self.status_bar.winfo_exists(): self.status_bar.config(text="Starting next round...")
             # Disable buttons while processing
             for button in self.action_buttons.values():
                  if button.winfo_exists(): button.config(state=tk.DISABLED)
             # Call _step_env to let env handle the round transition
             self.root.after(100, lambda: self._step_env(-1)) # Short delay

        elif self.seat_configs.get(current_player_id) == 'empty': # Points to empty seat
             # Env should handle skipping, trigger next step processing via AI handler path
             print(f"Debug: Current player {current_player_id} is empty seat. Triggering step.")
             # Use handle_ai_action which calls _step_env(-1)
             self.root.after(50, self._handle_ai_action)


    def _handle_ai_action(self):
        """ Triggers the environment to process the AI opponent's turn. """
        # Initial guard clause
        if not self.tournament_running or not hasattr(self.env, 'current_player_id'):
            print("Debug: AI action handler called but tournament stopped or env invalid.")
            return

        # Get current player ID
        current_player_id = None
        try:
             current_player_id = self.env.current_player_id
        except Exception as e:
             print(f"Error accessing env.current_player_id in _handle_ai_action: {e}")
             self.root.after(50, self._reconfigure) # Go to config on error
             return

        if current_player_id is None:
            print("Warning: _handle_ai_action called when current_player_id is None. Skipping AI step.")
            # If ID is None, _process_game_turn should handle triggering the next step.
            # Avoid calling _step_env directly here if ID is None.
            return

        if current_player_id == self.human_player_seat:
            print("Debug: AI action handler called but it's human's turn. Skipping.")
            return

        # Skip AI action if seat is empty and trigger next step
        if self.seat_configs.get(current_player_id) == 'empty':
             print(f"Debug: Skipping AI action for empty Seat {current_player_id + 1}")
             # Call _step_env with dummy action to advance past empty seat
             self._step_env(-1)
             return


        print(f"Debug: Handling AI action for Seat {current_player_id + 1}")
        # The environment's step function handles getting the action from the opponent's policy when action=-1
        self._step_env(-1) # Use -1 to signal the env should use the internal policy for the current player

    def _handle_human_action(self, action_str):
        """ Handles the button press for a human player's action. """
        current_player_id = self.env.current_player_id if hasattr(self.env, 'current_player_id') else None
        # Verify it's actually the human's turn
        if not self.env or not self.tournament_running or current_player_id != self.human_player_seat:
            print(f"Warning: Human action '{action_str}' received, but it's not the human's turn (Current: {current_player_id}).")
            return

        print(f"Human (Seat {self.human_player_seat + 1}) chose action: {action_str}")

        # Convert action string to the index the environment expects
        action_idx = self._action_string_to_idx.get(action_str, -1)
        if action_idx == -1:
            messagebox.showerror("Internal Error", f"Invalid action mapping for '{action_str}'. Cannot proceed.")
            return

        # Disable buttons immediately to prevent double-clicks
        for btn in self.action_buttons.values():
            if btn.winfo_exists(): btn.config(state=tk.DISABLED)

        if self.status_bar.winfo_exists(): self.status_bar.config(text=f"You chose '{action_str}'. Processing...")
        self.root.update_idletasks() # Force UI update

        # Step the environment with the chosen action index
        self._step_env(action_idx)

    def _step_env(self, action_idx):
        """ Steps the environment with the given action index and handles the result. """
        if not self.tournament_running or not self.env:
            print("Debug: Step env called but tournament/env not running.")
            return

        try:
            # Perform the step in the environment
            # Action_idx is the integer index for human, or -1 for AI (env uses its policy)
            next_encoded_state, reward, terminated, truncated, info = self.env.step(action_idx)

            # Store the results
            self.current_encoded_state = next_encoded_state
            self.last_info = info if isinstance(info, dict) else {} # Ensure info is a dict
            # Store terminated/truncated flags in last_info for checks in _process_game_turn
            self.last_info['terminated'] = terminated
            self.last_info['truncated'] = truncated
            done = terminated or truncated # Check if episode/round/tournament ended

            # --- Log key info ---
            last_action_info = info.get('last_action', {})
            # Find the player ID from the keys of last_action_info if it exists
            last_player_id = next(iter(last_action_info.keys())) if last_action_info else None
            last_action_str = last_action_info.get(last_player_id, "N/A") if last_player_id is not None else "N/A"
            # print(f"Debug: env.step result - Reward: {reward}, Done: {done}, Stage: {info.get('stage', 'N/A')}, Last Action ({last_player_id}): {last_action_str}") # Reduced verbosity
            if 'error' in info: print(f"ERROR in env info: {info['error']}")

            # --- Update UI based on new state ---
            self._update_ui() # Update stacks, cards, pot, etc.

            # --- Handle Round End ---
            round_over = info.get('round_over', False)
            delay_ms = 200 # Default delay before next turn/action

            if round_over:
                print("Debug: Round detected as over.")
                self._display_round_results(info) # Show winner, hands, etc.
                delay_ms = 2000 # Longer pause after showing round results

            # --- Handle Tournament End or Continue ---
            if done:
                print(f"Debug: Game is done (Terminated: {terminated}, Truncated: {truncated}). Handling tournament end.")
                # Use root.after to ensure UI updates before blocking with message box
                # Pass the info dict from the step that ended the game
                self.root.after(100, lambda info_end=info: self._handle_tournament_end(info_end))
            else:
                # Schedule the next turn processing after the delay
                self.root.after(delay_ms, self._process_game_turn)

        except Exception as e:
            # Catch errors during the environment step
            print(f"CRITICAL ERROR during env.step(): {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback
            messagebox.showerror("Environment Error", f"A critical error occurred during game progression:\n{e}\n\nReturning to configuration.")
            self.tournament_running = False
            # Use root.after to avoid issues if error occurs during tk callback
            self.root.after(50, self._reconfigure)

    # --- UI Update and Display Methods ---

    def _update_ui(self):
        """ Updates all UI elements based on the latest `self.last_info` dictionary. """
        if not self.game_window or not self.game_window.winfo_exists() or not self.env or not self.last_info:
            # print("Debug: Update UI skipped - window/env/info not ready.")
            return

        # Extract relevant information from the last_info dictionary
        stacks = self.last_info.get('stacks', {})
        pot = self.last_info.get('pot', 0)
        community_cards = self.last_info.get('community_cards', [])
        active_players = self.last_info.get('active_players', []) # Players still in the current hand
        # Get active players who are all-in (need to check stack == 0)
        all_in_players = [p for p in active_players if stacks.get(p, 0) == 0]
        # Folded players might not be in active_players, need separate tracking if env provides it
        # Assuming env doesn't explicitly track folded, derive from !active_players and stack > 0? Risky.
        # Let's rely on active_players and all_in_players for status for now.
        current_bets = self.last_info.get('current_bets', {}) # Bets in the current street
        last_actions = self.last_info.get('last_action', {}) # Map player_id -> last action string
        button_pos = self.last_info.get('button_pos', -1) # Dealer button position
        current_stage = self.last_info.get('stage', 'Unknown') # e.g., 'preflop', 'flop', 'turn', 'river'

        # Get human player's hand (might be empty if player folded or not dealt yet)
        human_hand = []
        if self.human_player_seat != -1:
             try:
                 # Use a method assumed to exist in the env
                 human_hand = self.env.get_player_hand(self.human_player_seat)
             except AttributeError:
                 print("Warning: env does not have get_player_hand method.")
             except Exception as e:
                 print(f"Error getting human hand: {e}")

        # --- Update Center Table ---
        # Pot Label
        if self.pot_label and self.pot_label.winfo_exists():
            self.pot_label.config(text=f"Pot: ${pot:.2f}")

        # *** DEBUGGING PRINT STATEMENT ADDED HERE ***
        # This will print the list of community cards the UI received from the environment's info dictionary.
        # Compare this output in your console to what you expect to see on the board.
        # print(f"DEBUG: Updating UI - Community Cards received: {community_cards}") # Keep for debugging if needed

        # Community Cards
        rendered_community = render_community_cards_for_labels(community_cards) # Util function formats cards
        for i, label in enumerate(self.community_card_labels):
            if label.winfo_exists():
                label.config(text=rendered_community[i]) # Update label text

        # Player Hand (Human)
        if self.human_player_seat != -1 and self.seat_configs.get(self.human_player_seat) != 'empty':
            if self.player_hand_frame.winfo_exists():
                self.player_hand_frame.config(text=f"Your Hand (Seat {self.human_player_seat + 1}) - {current_stage.title()}")
            rendered_hand = render_hand_for_labels(human_hand) # Util function formats cards
            for i, label in enumerate(self.player_card_labels):
                if label.winfo_exists():
                    label.config(text=rendered_hand[i])
        elif self.player_hand_frame.winfo_exists(): # Hide if human seat is empty
             self.player_hand_frame.config(text="Player Hand") # Reset title
             for label in self.player_card_labels:
                  if label.winfo_exists(): label.config(text="")


        # Clear showdown labels if the round is NOT over
        round_just_ended = self.last_info.get('round_over', False)
        if not round_just_ended:
            # Clear individual showdown labels under each seat
            for i in range(NUM_PLAYERS):
                if i in self.seat_showdown_card_labels and self.seat_showdown_card_labels[i].winfo_exists():
                    self.seat_showdown_card_labels[i].config(text="") # Clear text
            # Clear the central overview label
            if self.showdown_overview_label and self.showdown_overview_label.winfo_exists():
                self.showdown_overview_label.config(text="")

        # --- Update Seat Information ---
        for i in range(NUM_PLAYERS):
            # Skip if seat frame doesn't exist (shouldn't happen)
            if i not in self.seat_frames or not self.seat_frames[i].winfo_exists():
                continue

            seat_type = self.seat_configs.get(i, 'N/A') # Get original type ('empty', 'model', etc.)
            stack = stacks.get(i, 0)
            current_bet = current_bets.get(i, 0)

            # Check if seat is empty and update display accordingly
            if seat_type == 'empty':
                 if i in self.seat_status_labels: self.seat_status_labels[i].config(text="--- EMPTY ---", style="Empty.TLabel")
                 if i in self.seat_stack_labels: self.seat_stack_labels[i].config(text="", style="Empty.TLabel")
                 if i in self.seat_action_labels: self.seat_action_labels[i].config(text="", style="Empty.TLabel")
                 if i in self.seat_showdown_card_labels: self.seat_showdown_card_labels[i].config(text="", style="Empty.TLabel")
                 self.seat_frames[i].config(relief="flat") # Make empty seats less prominent
                 continue # Skip rest of update for empty seat

            # --- Update for Non-Empty Seats ---
            # Reset style in case it was empty before
            if i in self.seat_status_labels: self.seat_status_labels[i].config(style="TLabel")
            if i in self.seat_stack_labels: self.seat_stack_labels[i].config(style="TLabel")
            if i in self.seat_action_labels: self.seat_action_labels[i].config(style="TLabel", foreground="gray") # Reset color
            if i in self.seat_showdown_card_labels: self.seat_showdown_card_labels[i].config(style="Showdown.TLabel")
            # Reset relief based on whose turn it is (handled in _process_game_turn)
            # self.seat_frames[i].config(relief="groove") # Reset relief


            # Determine player status text
            status_text = f"Type: {seat_type.title()}"
            player_status = ""
            is_player_active_in_hand = i in active_players # Still eligible to win pot
            has_stack = stack > 0

            if i == self.human_player_seat: status_text += " (You)"
            if i == button_pos: status_text += " (BTN)"

            # Refined Status Check
            if not has_stack: player_status = " (Out)"
            elif i in all_in_players: player_status = " (All-In)"
            elif not is_player_active_in_hand and current_stage != 'prehand' and current_stage != 'showdown':
                 # If hand is in progress, they have chips, but aren't active -> Folded
                 player_status = " (Folded)"


            # Update Status Label
            if i in self.seat_status_labels and self.seat_status_labels[i].winfo_exists():
                self.seat_status_labels[i].config(text=status_text + player_status)

            # Update Stack and Bet Label
            stack_bet_text = f"Stack: ${stack:.2f}"
            if current_bet > 0:
                stack_bet_text += f" (Bet: ${current_bet:.2f})"
            if i in self.seat_stack_labels and self.seat_stack_labels[i].winfo_exists():
                self.seat_stack_labels[i].config(text=stack_bet_text)

            # Update Last Action Label
            if i in self.seat_action_labels and self.seat_action_labels[i].winfo_exists():
                action_str = last_actions.get(i, "-")
                # Only show action if player acted this round or is involved
                if action_str != "-":
                     self.seat_action_labels[i].config(text=f"Last Action: {action_str.replace('_', ' ').title()}", foreground="black")
                else:
                     # Clear action if player hasn't acted or is waiting
                     self.seat_action_labels[i].config(text="Last Action: -", foreground="gray")

        # Force Tkinter to process pending UI updates immediately
        # self.root.update_idletasks() # Use cautiously, can sometimes cause issues

    def _display_round_results(self, round_info):
        """ Displays the results of a completed round (showdown or everyone folds). """
        print("--- Displaying Round Results ---")
        if not isinstance(round_info, dict):
             print("Error: Invalid round_info received.")
             return

        winners = round_info.get('winners', []) # List of winner player indices
        showdown_hands_info = round_info.get('showdown_hands', {}) # Dict: {pid: {'hand': [...], 'desc': '...'}}
        round_reward = round_info.get('round_reward', 0.0) # Reward for the human player this round
        final_pot = round_info.get('final_pot', self.last_info.get('pot', 0)) # Pot size at end of round

        # Get all hands dealt at the start of the round (needed even if players folded)
        # Ensure this attribute exists and is populated correctly in your envs.py
        all_dealt_hands = getattr(self.env, 'hands', {})
        if not all_dealt_hands:
             print("Warning: `env.hands` attribute not found or empty in _display_round_results. Cannot show folded hands.")


        # --- Update Showdown Labels under each Seat ---
        # This loop iterates through ALL players who were dealt hands initially.
        for pid, hand_list in all_dealt_hands.items():
            # Skip empty seats
            if self.seat_configs.get(pid) == 'empty': continue

            if pid in self.seat_showdown_card_labels and self.seat_showdown_card_labels[pid].winfo_exists():
                # Format hand string using card_utils - this shows the cards regardless of fold status
                hand_str = render_hand(hand_list, separator=" ") if hand_list else "N/A"

                # Get hand description (e.g., "Pair of Kings") ONLY if available from showdown_info
                # We don't evaluate folded hands here to avoid extra computation/complexity,
                # but we DO show their dealt cards via hand_str.
                desc_str = ""
                if pid in showdown_hands_info:
                    desc_str = showdown_hands_info[pid].get('desc', '')

                # Construct display text - includes cards for everyone, description only for shown hands
                display_text = f"Cards: {hand_str}"
                if desc_str:
                    display_text += f"\n({desc_str})"

                # Update the label under the player's seat frame
                self.seat_showdown_card_labels[pid].config(text=display_text)
            elif pid not in self.seat_showdown_card_labels:
                 # Only warn if it's not an empty seat
                 if self.seat_configs.get(pid) != 'empty':
                      print(f"Warning: No showdown label found for player {pid}")

        # --- Update Central Overview Label ---
        overview_text = f"Round Over! Final Pot: ${final_pot:.2f}\n"
        if winners:
            win_amount = final_pot / len(winners) if winners else 0
            # Convert indices to seat numbers (1-based)
            winner_seats = [w + 1 for w in winners if self.seat_configs.get(w) != 'empty']
            overview_text += f"Winner(s): Seat(s) {', '.join(map(str, winner_seats))} (${win_amount:.2f} each)\n"

            # Try to get the description of the winning hand(s)
            win_desc = ""
            # Check if the first winner showed their hand
            if winners and winners[0] in showdown_hands_info:
                 win_desc = showdown_hands_info[winners[0]].get('desc', '')
            # If multiple winners, maybe list all descriptions? For now, just first.

            if win_desc:
                 overview_text += f"Winning Hand: {win_desc}\n"
        else:
            # This might happen if everyone folds except one person before showdown
            # Find the single remaining active player
            active_in_round = self.last_info.get('active_players', [])
            # Filter out empty seats from active_in_round
            active_playing = [p for p in active_in_round if self.seat_configs.get(p) != 'empty']

            if len(active_playing) == 1:
                 last_man_standing = active_playing[0]
                 overview_text += f"Winner: Seat {last_man_standing + 1} (Opponents Folded)\n"
            else:
                 overview_text += "No winner determined (e.g., error or unusual fold scenario).\n"


        # Add human player's reward for the round
        if self.human_player_seat != -1 and self.seat_configs.get(self.human_player_seat) != 'empty':
            reward_color = "green" if round_reward > 0 else "red" if round_reward < 0 else "black"
            # Simple text version, coloring requires more complex handling (e.g., tags in Text widget)
            overview_text += f"Your Round Reward: ${round_reward:.2f}"

        # Update the central overview label
        if self.showdown_overview_label and self.showdown_overview_label.winfo_exists():
            self.showdown_overview_label.config(text=overview_text)

        # Update status bar with a concise result
        status_msg = overview_text.split('\n')[1] if winners or len(self.last_info.get('active_players', [])) == 1 else "Round Over."
        if self.status_bar.winfo_exists(): self.status_bar.config(text=status_msg)
        # self.root.update_idletasks() # Ensure UI updates before the pause

    def _handle_tournament_end(self, final_info):
        """ Handles the end of the tournament, displays results, and prompts for a new game. """
        print("\n--- Tournament Ended (UI Handling) ---")
        self.tournament_running = False # Stop the game loop

        # Disable action buttons and update turn label
        for btn in self.action_buttons.values():
            if btn.winfo_exists(): btn.config(state=tk.DISABLED)
        if self.turn_label.winfo_exists(): self.turn_label.config(text="Tournament Over", foreground="darkred")

        # Determine the winner based on final stacks (considering only non-empty seats)
        final_stacks = final_info.get('stacks', {})
        winner_id = -1 # -1: Undetermined, -2: No winner
        max_stack = -1
        playing_players = [i for i, t in self.seat_configs.items() if t != 'empty']


        # Find players with chips remaining among playing players
        active_players = [p for p in playing_players if final_stacks.get(p, 0) > 0]

        if len(active_players) == 1:
            winner_id = active_players[0] # Single player remaining is the winner
        elif not active_players and playing_players: # No active players, but some were playing
             print("Warning: Tournament ended with no players having stacks > 0. Finding max stack among playing players.")
             max_stack = -float('inf')
             for pid in playing_players:
                  stack = final_stacks.get(pid, 0)
                  if stack > max_stack:
                       max_stack = stack
                       winner_id = pid
             if max_stack <= 0 : winner_id = -2 # Truly no winner if max stack is <= 0
        elif not playing_players: # No playing players at all
             winner_id = -2
             print("Warning: Tournament ended with no playing players configured.")
        else: # Multiple players remain active (e.g., truncated), declare highest stack winner
            print(f"Tournament ended with multiple players active: {active_players}. Declaring highest stack winner.")
            for pid in active_players:
                 if final_stacks.get(pid, 0) > max_stack:
                     max_stack = final_stacks.get(pid, 0)
                     winner_id = pid

        # --- Construct Result Message ---
        result_message = "Tournament Over!\n\n"
        if winner_id >= 0:
            win_msg = f"Seat {winner_id + 1} ({self.seat_configs.get(winner_id, 'N/A').title()}) wins!"
            if winner_id == self.human_player_seat:
                win_msg += " Congratulations!"
            result_message += win_msg + "\n"
        elif winner_id == -2:
             result_message += "Tournament ended unexpectedly with no winner.\n"
        else: # Should only happen if multiple players finish with exact same highest stack (unlikely)
             result_message += "Tournament ended. Highest stack wins.\n" # Generic message

        # Add human player elimination status
        is_human_playing = self.human_player_seat != -1 and self.seat_configs.get(self.human_player_seat) != 'empty'
        if is_human_playing and final_stacks.get(self.human_player_seat, 0) <= 0 and winner_id != self.human_player_seat:
             result_message += "You were eliminated.\n"

        # List final stacks for non-empty seats
        result_message += "\nFinal Stacks:\n"
        for pid in range(NUM_PLAYERS):
             if self.seat_configs.get(pid) != 'empty': # Only show non-empty seats
                 stack = final_stacks.get(pid, 0)
                 result_message += f"  Seat {pid+1}: ${stack:.2f}\n"

        print(result_message) # Log final results

        # --- Prompt for New Game ---
        # Use root.after to ensure the message box appears after the current event loop cycle
        self.root.after(200, lambda msg=result_message: self._prompt_new_game(msg))

    def _prompt_new_game(self, result_message):
        """ Shows a message box with results and asks to play again or reconfigure. """
        # Ensure game window still exists before showing message box relative to it
        parent_window = self.game_window if (self.game_window and self.game_window.winfo_exists()) else self.root

        try:
            response = messagebox.askyesno(
                "Tournament Over",
                result_message + "\nStart a new tournament with the same configuration?",
                parent=parent_window # Make message box appear over game window or root
            )
        except tk.TclError as e:
             print(f"Error showing messagebox (window might be destroyed): {e}")
             # Default to reconfigure if message box fails
             self._reconfigure()
             return


        if response is True:
            # Restart game with same config
            print("Restarting game with the same configuration...")
            # Need to clean up game window before starting again
            if self.game_window and self.game_window.winfo_exists():
                 self.game_window.destroy()
            self.game_window = None

            if self.env:
                try: self.env.close()
                except Exception as e: print(f"Error closing env before restart: {e}")
                self.env = None
            # Reset necessary state variables but keep configs and model path
            self.current_encoded_state = None; self.last_info = {}; self.tournament_running = False
            # Clear UI references that will be recreated
            self.seat_frames = {}; self.seat_status_labels = {}; self.seat_action_labels = {}; self.seat_stack_labels = {}; self.seat_showdown_card_labels = {}
            self.player_card_labels = []; self.community_card_labels = []; self.action_buttons = {}
            self.pot_label = None; self.turn_label = None; self.player_hand_frame = None; self.status_bar = None; self.showdown_overview_label = None

            # Call _start_game again, which will use existing self.seat_configs
            # Need to ensure model is reloaded if needed, _start_game handles this
            self.agent_model = None # Clear loaded model reference
            self._start_game()
        else:
            # Go back to configuration screen
            print("Returning to configuration screen.")
            self._reconfigure()

    def _reconfigure(self):
        """ Cleans up the current game state and returns to the configuration window. """
        print("Reconfiguring game...")
        self.tournament_running = False # Ensure game loop stops

        # Destroy the game window if it exists
        if self.game_window and self.game_window.winfo_exists():
            self.game_window.destroy()
        self.game_window = None

        # Close the environment if it exists
        if self.env:
            try:
                self.env.close()
                print("Environment closed.")
            except Exception as e:
                print(f"Error closing environment during reconfiguration: {e}")
        self.env = None

        # Reset game-related state variables
        self.agent_model = None # Force model reload if needed
        # Keep self.seat_configs from previous run? No, reconfigure means start fresh.
        # self.seat_configs = {} # Resetting here might lose defaults, let config window handle it.
        self.current_encoded_state = None
        self.human_player_seat = -1
        self.last_info = {}
        self.action_list = []
        self.num_actions = 0
        self._action_string_to_idx = {}

        # Clear UI widget references
        self.seat_frames = {}
        self.seat_status_labels = {}
        self.seat_action_labels = {}
        self.seat_stack_labels = {}
        self.seat_showdown_card_labels = {}
        self.player_card_labels = []
        self.community_card_labels = []
        self.action_buttons = {}
        self.pot_label = None
        self.turn_label = None
        self.player_hand_frame = None
        self.status_bar = None
        self.showdown_overview_label = None

        # Re-create the configuration window
        self._create_config_window()


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw() # Keep the root window hidden

    # Attempt to apply a modern theme
    try:
        style = ttk.Style(root)
        # Try themes in order of preference
        available_themes = style.theme_names()
        preferred_themes = ['clam', 'alt', 'default'] # Add more like 'vista', 'xpnative' if needed
        for theme in preferred_themes:
            if theme in available_themes:
                 try:
                      style.theme_use(theme)
                      print(f"Using theme: {theme}")
                      break
                 except tk.TclError:
                      print(f"Failed to apply theme: {theme}")
        else:
             print("Preferred themes not found or failed to apply, using system default.")

    except tk.TclError:
        print("Warning: Failed to initialize ttk themes.")

    # Create and run the application instance
    app = PokerApp(root)
    root.mainloop() # Start the Tkinter event loop
    print("Application exited.")
