# main_ui.py
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import os
import torch
import random
import json # Needed for potentially loading detailed log data if extended
import numpy as np # Added for argmax/argsort

# Import from existing codebase
from envs import BaseFullPokerEnv, evaluate_hand #
from models import BestPokerModel #
from utils import encode_obs_eval #
# Import optional helper modules
from seat_config import SeatConfigManager
from human_action_handler import HumanActionHandler
from card_utils import render_card, render_hand, SUITS_UNICODE, RANKS #

# --- Constants ---
NUM_PLAYERS = 6
ACTION_LIST = ['fold', 'call', 'check', 'bet_small', 'bet_big', 'all_in']
STATE_DIM = 52 + 1 + (NUM_PLAYERS - 1) * 52 # 313 for full encoding
DEFAULT_CHECKPOINT_PREFIX = "checkpoints/checkpoint_"
DEFAULT_STACK = 10000

class PokerApp:
    """
    Main class for the Tkinter Poker Application.
    Manages the configuration and game windows, game state, and interactions.
    """
    def __init__(self, root):
        self.root = root
        self.root.withdraw() # Hide the main root window initially

        self.env = BaseFullPokerEnv(num_players=NUM_PLAYERS) #
        self.agent_model = None # Loaded based on checkpoint
        self.opponent_policies = {}
        self.seat_configs = {} # Stores type ('player', 'model', etc.) for each seat index (0-5)
        self.checkpoint_path = tk.StringVar(value="95.pt") # Default example suffix

        self.game_window = None
        self.config_window = None
        self.seat_config_manager = SeatConfigManager(num_players=NUM_PLAYERS)
        self.human_action_handler = HumanActionHandler(ACTION_LIST)

        # UI Elements (will be created in game window)
        self.seat_status_labels = {}
        self.seat_action_labels = {}
        self.seat_stack_labels = {}
        self.seat_showdown_card_labels = {} # NEW: Labels to show cards at showdown
        self.player_card_labels = []
        self.community_card_labels = []
        self.pot_label = None # Will be created in table_frame now
        self.current_bet_label = None # Remains in top_frame
        self.turn_label = None # Remains in top_frame
        self.action_buttons = {}

        # Game state tracking
        self.current_obs = None
        self.human_player_seat = -1 # Seat index of the human player
        self.last_round_active_players = [] # Store players active at showdown

        self._create_config_window()

    def _validate_checkpoint_path(self, suffix_or_path):
        """Ensures checkpoint path starts with the prefix if only a suffix is given."""
        if not suffix_or_path:
            return None # No checkpoint specified
        if not suffix_or_path.startswith("checkpoints/") and not os.path.dirname(suffix_or_path):
             # Prepend prefix only if it's just a filename/suffix
            return DEFAULT_CHECKPOINT_PREFIX + suffix_or_path
        return suffix_or_path # Assume full path provided

    def _load_model(self, path):
        """Loads the PyTorch model from the specified checkpoint path."""
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", f"Checkpoint not found: {path}")
            return None

        try:
            model = BestPokerModel(input_dim=STATE_DIM, num_actions=len(ACTION_LIST)) #
            # Load with strict=False as required
            checkpoint = torch.load(path, map_location=torch.device('cpu')) # Load to CPU for UI app

            # Handle potential dictionary nesting (e.g., if saved as state_dict)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict):
                 state_dict = checkpoint
            else:
                 messagebox.showerror("Error", "Invalid checkpoint format.")
                 return None

            # Basic key cleaning (e.g., remove 'module.' prefix if saved from DataParallel)
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                cleaned_state_dict[k.replace('module.', '')] = v

            model.load_state_dict(cleaned_state_dict, strict=False) #
            model.eval() # Set model to evaluation mode
            print(f"Successfully loaded model from {path}")
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load checkpoint {path}: {e}")
            return None

    def _get_opponent_policy(self, opponent_type, model):
        """Creates a policy function for non-human opponents."""
        if opponent_type == "model":
            if not model: return self._get_opponent_policy("random", None) # Fallback if model failed
            def policy_fn(obs, legal_actions):
                state = encode_obs_eval(obs, use_half_encoding=False) #
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_values = model(state_tensor)
                # Choose best legal action
                q_values_np = q_values.squeeze().numpy()
                sorted_actions = np.argsort(q_values_np)[::-1] # Indices from highest Q to lowest
                for action_idx in sorted_actions:
                    action_str = ACTION_LIST[action_idx]
                    if action_str in legal_actions:
                        return action_str
                # Fallback if no predicted action is legal (should be rare)
                if 'fold' in legal_actions: return 'fold'
                return random.choice(legal_actions) if legal_actions else 'fold'
            return policy_fn
        elif opponent_type == "random":
            def policy_fn(obs, legal_actions):
                return random.choice(legal_actions) if legal_actions else "fold"
            return policy_fn
        elif opponent_type == "variable":
            if not model: return self._get_opponent_policy("random", None) # Fallback
            def policy_fn(obs, legal_actions):
                if random.random() < 0.5: # Use model
                     state = encode_obs_eval(obs, use_half_encoding=False) #
                     state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                     with torch.no_grad():
                         q_values = model(state_tensor)
                     # Choose best legal action
                     q_values_np = q_values.squeeze().numpy()
                     sorted_actions = np.argsort(q_values_np)[::-1]
                     for action_idx in sorted_actions:
                         action_str = ACTION_LIST[action_idx]
                         if action_str in legal_actions:
                             return action_str
                     # Fallback
                     if 'fold' in legal_actions: return 'fold'
                     return random.choice(legal_actions) if legal_actions else 'fold'
                else: # Use random
                    return random.choice(legal_actions) if legal_actions else "fold"
            return policy_fn
        elif opponent_type == "empty":
             # Empty seats always fold immediately if asked, but shouldn't be asked
             def policy_fn(obs, legal_actions):
                 return "fold"
             return policy_fn
        else: # Default to random
             return self._get_opponent_policy("random", None)


    def _create_config_window(self):
        """Creates the initial configuration window."""
        if self.config_window:
            self.config_window.lift()
            return

        self.config_window = tk.Toplevel(self.root)
        self.config_window.title("Poker Game Configuration")
        self.config_window.protocol("WM_DELETE_WINDOW", self.root.quit) # Exit app if config closed

        main_frame = ttk.Frame(self.config_window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(main_frame, text="Configure Seats:", font="-weight bold").grid(row=0, column=0, columnspan=4, pady=(0, 10))

        self.seat_vars = []
        options = self.seat_config_manager.get_options() # Use manager
        for i in range(NUM_PLAYERS):
            ttk.Label(main_frame, text=f"Seat {i+1}:").grid(row=i+1, column=0, sticky=tk.W, padx=5)
            default_val = self.seat_config_manager.get_default(i) # Use manager
            var = tk.StringVar(value=default_val)

            # Using OptionMenu (dropdown)
            dropdown = ttk.OptionMenu(main_frame, var, default_val, *options)
            dropdown.grid(row=i+1, column=1, sticky=(tk.W, tk.E), padx=5)
            self.seat_vars.append(var)

        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(row=NUM_PLAYERS + 1, column=0, columnspan=4, sticky="ew", pady=10)

        # Checkpoint Entry
        ttk.Label(main_frame, text="Checkpoint Suffix/Path:").grid(row=NUM_PLAYERS + 2, column=0, sticky=tk.W, padx=5)
        checkpoint_entry = ttk.Entry(main_frame, textvariable=self.checkpoint_path, width=30)
        checkpoint_entry.grid(row=NUM_PLAYERS + 2, column=1, columnspan=3, sticky=(tk.W, tk.E), padx=5)

        # Start Game Button
        start_button = ttk.Button(main_frame, text="Start Game", command=self._start_game)
        start_button.grid(row=NUM_PLAYERS + 3, column=0, columnspan=4, pady=(10, 0))

        self.config_window.resizable(False, False)


    def _start_game(self):
        """Validates configuration, loads model, creates game window, and starts the first round."""
        # 1. Validate Seat Configuration using Manager
        is_valid, message = self.seat_config_manager.validate_config(self.seat_vars)
        if not is_valid:
            messagebox.showerror("Configuration Error", message)
            return

        # 2. Store Seat Configuration & Find Human Player
        self.seat_configs = {}
        self.human_player_seat = -1
        for i, var in enumerate(self.seat_vars):
             seat_type = var.get()
             self.seat_configs[i] = seat_type
             if seat_type == "player":
                 self.human_player_seat = i
        print("Seat Configurations:", self.seat_configs)

        # 3. Validate and Load Checkpoint
        checkpoint_input = self.checkpoint_path.get()
        full_checkpoint_path = self._validate_checkpoint_path(checkpoint_input)

        needs_model = any(stype in ["model", "variable"] for stype in self.seat_configs.values())
        if needs_model or full_checkpoint_path:
             if not full_checkpoint_path:
                 messagebox.showerror("Configuration Error", "A checkpoint path/suffix is required if using 'model' or 'variable' opponents.")
                 return
             self.agent_model = self._load_model(full_checkpoint_path)
             if not self.agent_model and needs_model:
                 messagebox.showerror("Error", "Model loading failed, but is required. Cannot start.")
                 return
             elif not self.agent_model:
                 print("Warning: Model loading failed, but no 'model'/'variable' players selected. Proceeding without AI model.")
        else:
             self.agent_model = None
             print("No model players selected and no checkpoint specified. Running without AI model.")


        # 4. Set Opponent Policies (used directly in game loop now)
        self.opponent_policies = {}
        for i in range(NUM_PLAYERS):
             if i == self.human_player_seat: continue
             seat_type = self.seat_configs[i]
             # Store the policy function directly
             self.opponent_policies[i] = self._get_opponent_policy(seat_type, self.agent_model)


        # 5. Close Config Window & Create Game Window
        if self.config_window:
            self.config_window.destroy()
            self.config_window = None
        self._create_game_window()

        # 6. Start the first round
        self._start_new_round()

    def _create_game_window(self):
        """Creates the main game window with all UI elements."""
        if self.game_window:
            self.game_window.lift()
            return

        self.game_window = tk.Toplevel(self.root)
        self.game_window.title("Poker Game")
        self.game_window.geometry("900x700") # Increased height slightly for pot/showdown cards
        self.game_window.protocol("WM_DELETE_WINDOW", self.root.quit) # Exit app if game window closed

        # Main Layout Frames
        top_frame = ttk.Frame(self.game_window, padding=5)
        top_frame.pack(fill=tk.X)
        middle_frame = ttk.Frame(self.game_window, padding=10)
        middle_frame.pack(fill=tk.BOTH, expand=True)
        bottom_frame = ttk.Frame(self.game_window, padding=10)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM) # Ensure bottom frame stays at bottom

        # Top Frame: Game Info (Bet, Turn) - Pot moved
        self.current_bet_label = ttk.Label(top_frame, text="Current Bet: $0", font="-weight bold")
        self.current_bet_label.pack(side=tk.LEFT, padx=20)
        self.turn_label = ttk.Label(top_frame, text="Turn: -", font="-weight bold", foreground="blue")
        self.turn_label.pack(side=tk.RIGHT, padx=20)

        # Middle Frame: Seats and Table
        left_seats_frame = ttk.Frame(middle_frame, padding=5)
        left_seats_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), anchor='n') # Anchor north
        table_frame = ttk.Frame(middle_frame, padding=10)
        table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_seats_frame = ttk.Frame(middle_frame, padding=5)
        right_seats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0), anchor='n') # Anchor north

        # Seat Status Panels
        self.seat_status_labels = {}
        self.seat_action_labels = {}
        self.seat_stack_labels = {}
        self.seat_showdown_card_labels = {} # Initialize dict for showdown labels
        seat_frames = {}
        for i in range(NUM_PLAYERS):
            parent_frame = left_seats_frame if i < 3 else right_seats_frame
            seat_frame = ttk.LabelFrame(parent_frame, text=f"Seat {i+1}", padding=10)
            seat_frame.pack(pady=5, fill=tk.X, anchor='n') # Anchor north
            seat_frames[i] = seat_frame

            status_label = ttk.Label(seat_frame, text=f"Type: {self.seat_configs.get(i, 'N/A')}")
            status_label.pack(anchor=tk.W)
            stack_label = ttk.Label(seat_frame, text="Stack: $0")
            stack_label.pack(anchor=tk.W)
            action_label = ttk.Label(seat_frame, text="Last Action: -", foreground="gray", wraplength=100) # Allow wrap
            action_label.pack(anchor=tk.W)

            # NEW: Label for showing cards at showdown
            showdown_label = ttk.Label(seat_frame, text="", foreground="darkgreen", font="-weight bold")
            showdown_label.pack(anchor=tk.W, pady=(5,0)) # Add padding top

            self.seat_status_labels[i] = status_label
            self.seat_stack_labels[i] = stack_label
            self.seat_action_labels[i] = action_label
            self.seat_showdown_card_labels[i] = showdown_label # Store reference

        # Table Frame: Community Cards, Player Hand, and Pot
        community_frame = ttk.LabelFrame(table_frame, text="Community Cards", padding=10)
        community_frame.pack(pady=10, anchor='center') # Center community cards
        self.community_card_labels = []
        for _ in range(5):
            lbl = ttk.Label(community_frame, text="", font=("Courier", 14), relief="ridge", width=7, anchor="center", padding=5)
            lbl.pack(side=tk.LEFT, padx=3)
            self.community_card_labels.append(lbl)

        player_hand_frame = ttk.LabelFrame(table_frame, text="Your Hand (Seat ?)", padding=10)
        player_hand_frame.pack(pady=10, anchor='center') # Center player hand
        self.player_hand_frame = player_hand_frame # To update title later
        self.player_card_labels = []
        for _ in range(2):
            lbl = ttk.Label(player_hand_frame, text="", font=("Courier", 16, "bold"), relief="solid", width=7, anchor="center", padding=5)
            lbl.pack(side=tk.LEFT, padx=5)
            self.player_card_labels.append(lbl)

        # NEW: Pot Label moved under player hand
        self.pot_label = ttk.Label(table_frame, text="Pot: $0", font=("-weight bold", 14)) # Larger font
        self.pot_label.pack(pady=(15, 5), anchor='center') # Add padding and center


        # Bottom Frame: Action Buttons
        action_frame = ttk.Frame(bottom_frame)
        action_frame.pack(pady=(0,10)) # Add padding below buttons
        self.action_buttons = {}
        button_actions = ['fold', 'check', 'call', 'bet_small', 'bet_big', 'all_in'] # Order for display
        for action in button_actions:
            btn = ttk.Button(action_frame, text=action.replace('_', ' ').title(), width=10, state=tk.DISABLED,
                             command=lambda a=action: self._handle_human_action(a))
            btn.pack(side=tk.LEFT, padx=5)
            self.action_buttons[action] = btn

        # Reconfigure/Exit buttons
        control_frame = ttk.Frame(bottom_frame)
        control_frame.pack(pady=(5,0))
        reconfig_button = ttk.Button(control_frame, text="Reconfigure Seats", command=self._reconfigure)
        reconfig_button.pack(side=tk.LEFT, padx=10)
        exit_button = ttk.Button(control_frame, text="Exit Application", command=self.root.quit)
        exit_button.pack(side=tk.LEFT, padx=10)


    def _update_ui(self):
        """Updates all UI elements based on the current environment state."""
        if not self.game_window or not self.env: return

        # Update Pot (now in table_frame) and Current Bet (top_frame)
        self.pot_label.config(text=f"Pot: ${self.env.pot}")
        self.current_bet_label.config(text=f"Current Bet: ${self.env.current_max_bet}")

        # Update Turn Indicator
        current_player_id = self.env.current_player
        if current_player_id is not None and not self.env.round_over:
             turn_text = f"Turn: Seat {current_player_id + 1}"
             is_human_turn = (current_player_id == self.human_player_seat)
             self.turn_label.config(text=turn_text, foreground="blue" if is_human_turn else "black")
             # Highlight the current player's seat frame (optional - change label weight)
             for idx, frame in self.seat_status_labels.items():
                 is_current = (idx == current_player_id)
                 font_weight = "bold" if is_current else "normal"
                 self.seat_status_labels[idx].config(font=f"-weight {font_weight}")
                 self.seat_stack_labels[idx].config(font=f"-weight {font_weight}")
                 # Optionally change action label weight too
                 # self.seat_action_labels[idx].config(font=f"-weight {font_weight}")

        elif self.env.round_over:
             self.turn_label.config(text="Turn: Showdown", foreground="black")
             # Reset highlight
             for idx in self.seat_status_labels:
                 self.seat_status_labels[idx].config(font="-weight normal")
                 self.seat_stack_labels[idx].config(font="-weight normal")
                 # self.seat_action_labels[idx].config(font=f"-weight normal")
        else:
             self.turn_label.config(text="Turn: -", foreground="black")
              # Reset highlight
             for idx in self.seat_status_labels:
                 self.seat_status_labels[idx].config(font="-weight normal")
                 self.seat_stack_labels[idx].config(font="-weight normal")
                 # self.seat_action_labels[idx].config(font=f"-weight normal")


        # Update Seat Information (Stacks, Status) - Action updated in _process_turn/_handle_human_action
        for i in range(NUM_PLAYERS):
            stack = self.env.stacks.get(i, 0)
            status_text = f"Type: {self.seat_configs.get(i, 'N/A').title()}"
            is_folded = (i not in self.env.active_players and self.env.stage != "showdown" and stack > 0)
            is_all_in = (stack <= 0 and self.seat_configs.get(i) != 'empty' and not is_folded)

            if self.seat_configs.get(i) == 'empty':
                status_text += " (Folded)"
                self.seat_stack_labels[i].config(text="Stack: $0")
                # Keep action label clear for empty seats unless showing cards
                if not self.env.round_over: self.seat_action_labels[i].config(text="Last Action: -")
            elif is_folded:
                status_text += " (Folded)"
                self.seat_stack_labels[i].config(text=f"Stack: ${stack}")
                # Action label color might be set elsewhere, ensure it's red if folded now
                self.seat_action_labels[i].config(foreground="red")
            elif is_all_in:
                 status_text += " (All-In)"
                 self.seat_stack_labels[i].config(text="Stack: $0")
                 # Action label color might be set elsewhere
            else:
                 self.seat_stack_labels[i].config(text=f"Stack: ${stack}")

            self.seat_status_labels[i].config(text=status_text)

            # Clear showdown cards unless the round is actually over
            if not self.env.round_over:
                self.seat_showdown_card_labels[i].config(text="")


        # Update Community Cards
        community = self.env.community_cards
        for i in range(5):
            if i < len(community):
                card_str = render_card(community[i]) # Use card_utils
                self.community_card_labels[i].config(text=card_str)
            else:
                self.community_card_labels[i].config(text="") # Clear unused labels

        # Update Player Hand Display
        if self.human_player_seat != -1:
            self.player_hand_frame.config(text=f"Your Hand (Seat {self.human_player_seat + 1})")
            hand = self.env.hands.get(self.human_player_seat, [])
            for i in range(2):
                if i < len(hand):
                    card_str = render_card(hand[i]) # Use card_utils
                    self.player_card_labels[i].config(text=card_str)
                else:
                    self.player_card_labels[i].config(text="") # Should not happen in poker

        # Update Action Buttons State
        if current_player_id == self.human_player_seat and not self.env.round_over:
            legal_actions = self.env._get_legal_actions(self.human_player_seat) #
            #print(f"Human turn (Seat {self.human_player_seat+1}). Legal actions: {legal_actions}") # Debug
            for action, button in self.action_buttons.items():
                is_legal = (action in legal_actions)
                # Handle check/call enabling logic more robustly
                can_check = (self.env.current_bets.get(self.human_player_seat, 0) == self.env.current_max_bet)
                can_call = (self.env.current_bets.get(self.human_player_seat, 0) < self.env.current_max_bet)

                if action == 'check' and not can_check:
                     button.config(state=tk.DISABLED)
                elif action == 'call' and not can_call:
                     button.config(state=tk.DISABLED)
                elif is_legal:
                    button.config(state=tk.NORMAL)
                else:
                    button.config(state=tk.DISABLED)
        else:
            # Disable all buttons if not human's turn or round over
            for button in self.action_buttons.values():
                button.config(state=tk.DISABLED)

        self.root.update_idletasks() # Force UI update


    def _start_new_round(self):
        """Resets the environment and starts the game loop for a new round."""
        print("\n--- Starting New Round ---")
        # Clear showdown cards from previous round
        for i in range(NUM_PLAYERS):
            if i in self.seat_showdown_card_labels:
                self.seat_showdown_card_labels[i].config(text="")

        self.current_obs = self.env.reset()
        self.env.round_over = False # Explicitly reset round over flag
        self.last_round_active_players = [] # Clear list of players from last showdown

        # Set stacks to 0 for 'empty' seats before the round starts
        for seat_id, seat_type in self.seat_configs.items():
            if seat_type == 'empty':
                self.env.stacks[seat_id] = 0
                # Ensure they are not in active players list if env logic didn't catch it
                if seat_id in self.env.active_players:
                    try: self.env.active_players.remove(seat_id)
                    except ValueError: pass # Already removed
                # Also ensure they are not in players_to_act queue
                if seat_id in self.env.players_to_act:
                     try: self.env.players_to_act.remove(seat_id)
                     except ValueError: pass

        print(f"Initial Stacks: {self.env.stacks}")
        print(f"Dealer: {self.env.dealer}, SB: {self.env.small_blind}, BB: {self.env.big_blind}")
        print(f"Initial Active Players: {self.env.active_players}")
        print(f"Initial Players to Act Queue: {self.env.players_to_act}")
        print(f"Player to Act First: {self.env.current_player}")


        # Clear previous round actions display
        for i in range(NUM_PLAYERS):
             self.seat_action_labels[i].config(text="Last Action: -", foreground="gray")


        self._update_ui()
        # Start the turn processing after a short delay
        self.root.after(100, self._process_turn)


    def _process_turn(self):
        """Handles the logic for the current player's turn."""
        if self.env.round_over:
            # This check might be redundant if _handle_round_end is called correctly,
            # but serves as a safeguard.
            # Let _handle_round_end manage the end-of-round state.
            # Avoid calling _handle_round_end multiple times.
            if not self.last_round_active_players: # Check if end logic already ran
                print("Debug: _process_turn called when round_over is True, but end logic hasn't run yet. Calling _handle_round_end.")
                self._handle_round_end()
            else:
                # End logic has already run (or is in progress), do nothing here.
                print("Debug: _process_turn called when round_over is True and end logic likely ran/running.")
            return

        current_player_id = self.env.current_player
        if current_player_id is None:
             # Betting round likely ended, or only one player left.
             print("No player to act, checking stage progression...")
             self._progress_to_next_stage()
             return

        player_type = self.seat_configs.get(current_player_id)

        # Additional check: Ensure player is active and has chips OR is the designated player to act
        # This handles cases where env logic might lag in removing players, OR if an all-in player still needs to "act" (pass turn)
        is_active = current_player_id in self.env.active_players
        has_chips = self.env.stacks.get(current_player_id, 0) > 0
        is_in_action_queue = current_player_id in self.env.players_to_act

        # --- MODIFICATION START ---
        # Skip turn if player is NOT in the action queue, OR if they are empty/folded/busted AND somehow still current_player
        # (The env should ideally handle this, but this adds robustness)
        should_skip = False
        if not is_in_action_queue:
            print(f"Warning: Player {current_player_id+1} is current_player but not in players_to_act queue. Skipping.")
            should_skip = True
        elif player_type == 'empty':
            # Empty seats should never be the current player if env logic is correct, but handle defensively.
            print(f"Skipping empty Seat {current_player_id+1}.")
            should_skip = True
        elif not is_active and has_chips: # Folded but still has chips? Should be removed from active_players.
             print(f"Skipping folded player {current_player_id+1}")
             should_skip = True
        elif not has_chips and is_active: # All-in - check if they still need to act
             # All-in players might remain 'active' but have no more decisions.
             # The environment's _update_players_to_act should handle removing them
             # from the *betting* queue once their all-in is processed.
             # If they are still the current player, it means the turn needs to pass.
             print(f"Player {current_player_id+1} is All-In. Passing turn.")
             # We'll treat this like a normal turn progression below, but they won't get options.
             # No 'skip' needed here, proceed to normal logic, but expect limited/no legal actions.
             pass # Proceed to regular turn logic below
        elif not has_chips and not is_active: # Busted and inactive
             print(f"Skipping busted inactive player {current_player_id+1}")
             should_skip = True

        if should_skip:
            # Remove the player from the queue if they are somehow still in it
            if current_player_id in self.env.players_to_act:
                try:
                    self.env.players_to_act.remove(current_player_id)
                    print(f"Manually removed skipped player {current_player_id+1} from action queue.")
                except ValueError:
                    pass # Already gone

            # Check if the queue became empty after removal
            if not self.env.players_to_act:
                 # If removing the skipped player empties the queue, the betting round ends.
                 self._progress_to_next_stage()
            else:
                 # Queue still has players, set the next player from queue and schedule _process_turn
                 self.env.current_player = self.env.players_to_act[0] # Get next player
                 print(f"After skipping, next player set to: {self.env.current_player + 1}")
                 self.root.after(50, self._process_turn) # Try next turn quickly
            return # End processing for the skipped player
        # --- MODIFICATION END ---


        print(f"\nTurn: Seat {current_player_id + 1} ({player_type.title()})")
        self._update_ui() # Update UI to show whose turn it is, enable/disable buttons

        if player_type == "player":
            # Human player's turn - UI buttons are enabled/disabled by _update_ui
            legal_actions = self.env._get_legal_actions(current_player_id)
            if not legal_actions:
                 # This can happen if player is all-in but still technically active
                 print(f"Human player {current_player_id+1} has no legal actions (likely All-In). Auto-passing turn.")
                 # Simulate a 'pass' action - simply advance the turn.
                 self.root.after(100, lambda p=current_player_id: self._process_action_and_continue(p, "pass_all_in")) # Use a dummy action
            else:
                 print("Waiting for human action...")
                 # Action is triggered by button click (_handle_human_action)
                 pass
        else:
            # AI / Random / Variable player's turn
            policy = self.opponent_policies.get(current_player_id)
            if not policy: # Fallback
                 print(f"Warning: Policy not found for Seat {current_player_id + 1}, defaulting to random.")
                 policy = self._get_opponent_policy("random", None)

            obs_for_policy = self.env._get_obs(current_player_id)
            legal_actions = self.env._get_legal_actions(current_player_id)

            if not legal_actions:
                 # This can happen if player is all-in but still technically active
                 print(f"Seat {current_player_id+1} ({player_type.title()}) has no legal actions (likely All-In). Passing turn.")
                 action_str = "pass_all_in" # Use a dummy action
                 # Update the action label immediately
                 self.seat_action_labels[current_player_id].config(text="Action: All-In", foreground="orange")
                 self.root.update_idletasks()
                 # Process the dummy action after a short delay to advance turn
                 delay_ms = 750
                 self.root.after(delay_ms, lambda p=current_player_id, a=action_str: self._process_action_and_continue(p, a))

            else:
                 action_str = policy(obs_for_policy, legal_actions)

                 # Ensure the chosen action is valid
                 if action_str not in legal_actions:
                     print(f"Warning: Policy for Seat {current_player_id+1} chose illegal action '{action_str}'. Legal: {legal_actions}. Forcing valid action.")
                     # Prioritize call/check if possible, else fold
                     if 'call' in legal_actions: action_str = 'call'
                     elif 'check' in legal_actions: action_str = 'check'
                     elif 'fold' in legal_actions: action_str = 'fold'
                     else: action_str = legal_actions[0] # Should not happen if legal_actions is not empty

                 print(f"Seat {current_player_id + 1} ({player_type.title()}) chose action: {action_str}")

                 # Update the action label immediately (Coloring moved to _process_action_and_continue)
                 self.seat_action_labels[current_player_id].config(text=f"Action: {action_str.title()}", foreground="gray") # Temp color
                 self.root.update_idletasks()

                 # Process the action in the environment after a short delay
                 delay_ms = 750
                 self.root.after(delay_ms, lambda p=current_player_id, a=action_str: self._process_action_and_continue(p, a))


        print(f"\nTurn: Seat {current_player_id + 1} ({player_type.title()})")
        self._update_ui() # Update UI to show whose turn it is, enable/disable buttons

        if player_type == "player":
            # Human player's turn - UI buttons are enabled/disabled by _update_ui
            print("Waiting for human action...")
            # Action is triggered by button click (_handle_human_action)
            pass
        else:
            # AI / Random / Variable player's turn
            # Get the stored policy function
            policy = self.opponent_policies.get(current_player_id)
            if not policy: # Fallback
                 print(f"Warning: Policy not found for Seat {current_player_id + 1}, defaulting to random.")
                 policy = self._get_opponent_policy("random", None)

            obs_for_policy = self.env._get_obs(current_player_id)
            legal_actions = self.env._get_legal_actions(current_player_id)

            if not legal_actions:
                 # This case *should* be covered by the active/stack check above, but as a failsafe:
                 print(f"Warning: No legal actions for active player {current_player_id+1}. Forcing fold.")
                 action_str = "fold"
            else:
                 action_str = policy(obs_for_policy, legal_actions)

                 # Ensure the chosen action is valid
                 if action_str not in legal_actions:
                     print(f"Warning: Policy for Seat {current_player_id+1} chose illegal action '{action_str}'. Legal: {legal_actions}. Forcing valid action.")
                     # Prioritize call/check if possible, else fold
                     if 'call' in legal_actions: action_str = 'call'
                     elif 'check' in legal_actions: action_str = 'check'
                     elif 'fold' in legal_actions: action_str = 'fold'
                     else: action_str = legal_actions[0] # Should not happen


            print(f"Seat {current_player_id + 1} ({player_type.title()}) chose action: {action_str}")

            # Update the action label immediately
            self.seat_action_labels[current_player_id].config(text=f"Action: {action_str.title()}", foreground="green")
            self.root.update_idletasks()

            # Process the action in the environment after a short delay
            delay_ms = 750
            self.root.after(delay_ms, lambda p=current_player_id, a=action_str: self._process_action_and_continue(p, a))


    def _process_action_and_continue(self, player_id, action_str):
         """ Processes the action in the env and schedules the next turn """
         # Update action label for the player who acted
         # Determine color based on action type
         action_color = "black"
         if action_str == 'fold': action_color = 'red'
         elif 'bet' in action_str or action_str == 'all_in': action_color = 'orange'
         elif action_str == 'call': action_color = 'blue'
         elif action_str == 'check': action_color = 'purple'
         self.seat_action_labels[player_id].config(text=f"Action: {action_str.title()}", foreground=action_color)

         # Core action processing using env's method
         self.env._process_action(player_id, action_str)

         # Determine who is next to act based on the environment's logic
         self.env._update_players_to_act(player_id) # Pass the player who just acted

         # Check if the betting round is over
         if not self.env.players_to_act:
              self._progress_to_next_stage()
         else:
              # Continue to the next player's turn
              self.root.after(100, self._process_turn)


    def _handle_human_action(self, action_str):
        """Callback for human player action buttons."""
        human_seat = self.human_player_seat
        if self.env.current_player != human_seat or self.env.round_over:
            print("Warning: Human action button clicked, but not human's turn or round over.")
            return

        legal_actions = self.env._get_legal_actions(human_seat) #
        if action_str not in legal_actions:
            print(f"Warning: Human chose illegal action '{action_str}'. Legal: {legal_actions}")
            # Re-enable valid buttons if needed, though UI update should handle this
            self._update_ui()
            return

        print(f"Human (Seat {human_seat + 1}) chose action: {action_str}")

        # Disable buttons immediately after click
        for btn in self.action_buttons.values():
             btn.config(state=tk.DISABLED)

        # Process the action and continue the game loop
        # Action label update is handled within _process_action_and_continue now
        self._process_action_and_continue(human_seat, action_str)

    def _progress_to_next_stage(self):
         """ Advances the game to the next stage (Flop, Turn, River, Showdown) """
         # Check if only one player remains active
         if len(self.env.active_players) <= 1:
              print("Only one active player left. Ending round.")
              self.env.stage = "showdown" # Proceed directly to finalize
              # Ensure round_over flag is set before calling handle_round_end
              if not self.env.round_over:
                   self._handle_round_end()
              return

         print(f"Betting round ended for stage: {self.env.stage}")
         # Use the environment's stage progression logic
         self.env._progress_stage()
         print(f"Advanced to stage: {self.env.stage}")

         if self.env.stage == "showdown":
             if not self.env.round_over: # Ensure not called multiple times
                 self._handle_round_end()
         else:
              # Reset last action labels for the new betting round for active players
              for i in self.env.active_players:
                   self.seat_action_labels[i].config(text="Last Action: -", foreground="gray")
              # Start the next turn process
              self.root.after(100, self._process_turn)


    def _handle_round_end(self):
        """Handles the end of a round, evaluates hands, displays results, shows cards, and prompts."""
        if self.env.round_over: # Prevent multiple calls
            # print("Debug: _handle_round_end called while already round_over.") # Optional debug
            return

        print("\n--- Round Ended ---")
        self.env.round_over = True # Set flag immediately
        self.last_round_active_players = self.env.active_players[:] # Store who made it to the end
        print(f"Debug Showdown: Stored last_round_active_players = {self.last_round_active_players}") # DEBUG

        # Disable action buttons and update turn label
        for btn in self.action_buttons.values():
            btn.config(state=tk.DISABLED)
        self.turn_label.config(text="Turn: Showdown")

        # --- Winner Calculation Logic (as before) ---
        winners = []
        win_amount = 0
        scores = {}
        final_stacks = self.env.stacks.copy()
        pot_to_distribute = self.env.pot

        if len(self.last_round_active_players) > 0:
            if len(self.last_round_active_players) == 1: # Won by default
                winners = self.last_round_active_players[:]
                scores[winners[0]] = "Won by default"
            else: # Showdown evaluation
                # ... (evaluation logic as before, populating final_scores and scores) ...
                final_scores = {}
                print(f"Debug: Evaluating showdown for players: {self.last_round_active_players}")
                # Ensure board is complete for evaluation before looping
                while len(self.env.community_cards) < 5:
                    if not self.env.deck: break
                    self.env.community_cards.append(self.env.deck.pop())

                for pid in self.last_round_active_players:
                    if pid in self.env.hands:
                        full_hand = self.env.hands[pid] + self.env.community_cards
                        score_tuple = evaluate_hand(full_hand) # Use envs.evaluate_hand
                        final_scores[pid] = score_tuple
                        scores[pid] = score_tuple # Store tuple
                    else:
                         scores[pid] = "Error: No hand found"
                         final_scores[pid] = (-1, []) # Assign a losing score

                best_score_tuple = None
                for pid, score_tuple in final_scores.items():
                     if isinstance(score_tuple, tuple):
                          if best_score_tuple is None or score_tuple > best_score_tuple:
                              best_score_tuple = score_tuple
                              winners = [pid]
                          elif score_tuple == best_score_tuple:
                              # Avoid adding duplicates if score calculated multiple times somehow
                              if pid not in winners:
                                 winners.append(pid)

            # Calculate winnings
            if winners:
                win_amount = pot_to_distribute / len(winners)
                for pid in winners:
                    final_stacks[pid] = final_stacks.get(pid, 0) + win_amount
            print(f"Debug Showdown: Determined Winners = {[w+1 for w in winners]}") # DEBUG
            # print(f"Final Scores/Status: {scores}") # Optional debug
        else:
             print("Warning: Round ended with no active players.")
        # --- End Winner Calculation ---

        # --- CARD DISPLAY LOGIC WITH DEBUGGING ---
        # ===> PRINT 1: Check available hands <===
        print(f"Debug Showdown: Current env.hands before display loop: {self.env.hands}")

        for pid in range(NUM_PLAYERS):
            label_to_update = self.seat_showdown_card_labels.get(pid)
            if not label_to_update: continue

            hand = self.env.hands.get(pid, []) # Get hand info

            is_winner = pid in winners
            was_active_at_showdown = pid in self.last_round_active_players
            is_human = pid == self.human_player_seat
            # Check if more than one player was left at the point showdown was determined
            showdown_occurred = len(self.last_round_active_players) > 1

            should_show_cards = False
            if is_winner:
                should_show_cards = True
            elif showdown_occurred and was_active_at_showdown and not is_human:
                should_show_cards = True

            # ===> PRINT 2: Check conditions for each player <===
            print(f"Debug Showdown Loop (pid={pid}): hand={hand}, is_winner={is_winner}, was_active={was_active_at_showdown}, is_human={is_human}, showdown_occurred={showdown_occurred} -> should_show={should_show_cards}")

            # Update the label
            if should_show_cards:
                if hand: # Check specifically if hand data exists
                    hand_str = render_hand(hand)
                    label_to_update.config(text=f"Cards: {hand_str}")
                else:
                    label_to_update.config(text="Cards: N/A")
                    # This warning indicates the logic decided to show cards, but data was missing
                    print(f"Warning: Condition met for pid={pid}, but no hand found in env.hands!")
            else:
                # Clear label for players whose cards should not be shown
                label_to_update.config(text="")
        # --- END OF CARD DISPLAY LOGIC ---

        self._update_ui() # Update UI
        # print(f"Debug Showdown: UI updated after setting card labels.") # Optional debug

        # --- Display results in a popup (as before) ---
        # ... (message box logic remains the same) ...
        result_message = f"Round Over!\nPot: ${pot_to_distribute:.0f}\n\n"

        if not winners:
             result_message += "No winner determined (Error or unexpected state).\n"
        # ... (rest of message formatting) ...
        elif len(self.last_round_active_players) == 1 and len(winners) == 1:
             winner_id = winners[0]
             # ... message for win by default ...
             winner_hand = self.env.hands.get(winner_id, [])
             if winner_hand:
                 result_message += f"Winner's Hand: {render_hand(winner_hand)}\n"

        else: # Showdown results
             # ... message for showdown results ...
             result_message += "Showdown Results:\n"
             sorted_showdown_players = sorted(self.last_round_active_players)
             for pid in sorted_showdown_players:
                 hand_str = render_hand(self.env.hands.get(pid, ["?", "?"])) # Use stored hand
                 score_desc = self._describe_hand_score(scores.get(pid))
                 result_message += f"  Seat {pid+1}: {hand_str} ({score_desc})"
                 if pid in winners:
                     result_message += f" -> Wins ${win_amount:.2f}\n"
                 else:
                     result_message += "\n"


        result_message += "\nFinal Stacks (Approximate):\n"
        for pid in range(NUM_PLAYERS):
             if self.seat_configs.get(pid) != 'empty':
                result_message += f"  Seat {pid+1}: ${final_stacks.get(pid, 0):.0f}\n"

        response = messagebox.askquestion("Round Over", result_message + "\nStart a new round?",
                                          icon='info', type='yesnocancel')

        if response == 'yes':
            self._start_new_round()
        elif response == 'no':
            self._reconfigure()
        else: # Cancel or closed box
            self.root.quit()

        self._update_ui() # Update UI to show final community cards & showdown hands
        print(f"Debug Showdown: UI updated after setting card labels. self.env.round_over = {self.env.round_over}") # DEBUG PRINT

        # Determine winners using env's evaluation logic
        winners = []
        win_amount = 0
        scores = {}
        final_stacks = self.env.stacks.copy() # Copy stacks before potential modification
        pot_to_distribute = self.env.pot

        if len(self.last_round_active_players) > 0:
            if len(self.last_round_active_players) == 1: # Won by default
                winners = self.last_round_active_players[:]
                scores[winners[0]] = "Won by default"
            else: # Showdown evaluation
                final_scores = {}
                for pid in self.last_round_active_players:
                    if pid in self.env.hands:
                        full_hand = self.env.hands[pid] + self.env.community_cards
                        final_scores[pid] = evaluate_hand(full_hand) # Use envs.evaluate_hand
                        scores[pid] = final_scores[pid] # Store tuple
                    else:
                         scores[pid] = "Error: No hand"

                best_score_tuple = None
                for pid, score_tuple in final_scores.items():
                     # Ensure score_tuple is valid before comparison
                     if isinstance(score_tuple, tuple):
                          if best_score_tuple is None or score_tuple > best_score_tuple:
                               best_score_tuple = score_tuple
                               winners = [pid]
                          elif score_tuple == best_score_tuple:
                               winners.append(pid)

            # Calculate winnings and update *temporary* final_stacks for display
            if winners:
                 win_amount = pot_to_distribute / len(winners)
                 for pid in winners:
                      final_stacks[pid] = final_stacks.get(pid, 0) + win_amount
            print(f"Winners: {[w+1 for w in winners]}, Amount each: ${win_amount:.2f}")
            print(f"Final Scores: {scores}")

            # --- Display results in a popup ---
            result_message = f"Round Over!\nPot: ${pot_to_distribute}\n\n"
            if len(self.last_round_active_players) == 1:
                 winner_id = winners[0]
                 result_message += f"Seat {winner_id + 1} ({self.seat_configs[winner_id].title()}) wins by default.\n"
                 result_message += f"Wins: ${win_amount:.2f}\n"
                 # Show hand if it's the human player winning by default
                 if winner_id == self.human_player_seat:
                      hand_str = render_hand(self.env.hands.get(winner_id, [])) # Use card_utils
                      result_message += f"Your Hand: {hand_str}\n"
            else:
                 result_message += "Showdown Results:\n"
                 for pid in sorted(self.last_round_active_players):
                      hand_str = render_hand(self.env.hands.get(pid, ["?", "?"])) # Use card_utils
                      score_desc = self._describe_hand_score(scores.get(pid))
                      result_message += f"  Seat {pid+1}: {hand_str} ({score_desc})"
                      if pid in winners:
                           result_message += f" -> Wins ${win_amount:.2f}\n"
                      else:
                           result_message += "\n"

            result_message += "\nFinal Stacks (Approximate):\n" # Env reset handles actual stacks
            for pid in range(NUM_PLAYERS):
                 result_message += f"  Seat {pid+1}: ${final_stacks.get(pid, 0):.0f}\n"

            # Prompt for next action using askquestion
            response = messagebox.askquestion("Round Over", result_message + "\nStart a new round?",
                                             icon='info', type='yesnocancel')

            if response == 'yes':
                 self._start_new_round()
            elif response == 'no':
                 self._reconfigure()
            else: # Cancel or closed box
                 self.root.quit()
        else:
             messagebox.showinfo("Round Over", "Round ended unexpectedly (no active players).\nStarting new round.")
             self._start_new_round()


    def _describe_hand_score(self, score_tuple):
         """ Converts the hand score tuple from evaluate_hand into a readable string. """
         if isinstance(score_tuple, str): return score_tuple # Handle errors/defaults
         if not isinstance(score_tuple, tuple) or len(score_tuple) != 2: return "Unknown Hand"

         rank_map_rev = {i: r for i, r in enumerate(RANKS, start=2)}
         rank_map_rev[14] = 'A' # Ace high for display

         try:
             rank_val, tiebreaker = score_tuple
             desc = ""
             if rank_val == 9: desc = "Straight Flush"
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
             if rank_val in [9, 5]: # Straight Flush / Straight
                  high_card = rank_map_rev.get(tiebreaker, '?')
                  desc += f" ({high_card} high)"
             elif rank_val == 8: # 4 of a Kind
                  quad_rank = rank_map_rev.get(tiebreaker[0], '?')
                  desc += f" ({quad_rank}s)"
             elif rank_val == 7: # Full House
                  trip_rank = rank_map_rev.get(tiebreaker[0], '?')
                  pair_rank = rank_map_rev.get(tiebreaker[1], '?')
                  desc += f" ({trip_rank}s / {pair_rank}s)"
             # Add more details for other ranks if desired (e.g., kickers)
             elif rank_val == 2: # Pair
                  pair_rank = rank_map_rev.get(tiebreaker[0], '?')
                  desc += f" ({pair_rank}s)"
             elif rank_val == 1: # High Card
                  high_card = rank_map_rev.get(tiebreaker[0], '?') if tiebreaker else '?'
                  desc += f" ({high_card} high)"


         except Exception as e:
             print(f"Error describing score {score_tuple}: {e}")
             return "Hand Score Error"

         return desc


    def _reconfigure(self):
        """Closes the game window and re-opens the configuration window."""
        if self.game_window:
            self.game_window.destroy()
            self.game_window = None
        # Reset game elements
        self.env = BaseFullPokerEnv(num_players=NUM_PLAYERS) # New env instance
        self.agent_model = None
        self.opponent_policies = {}
        # Clear UI element storage
        self.seat_status_labels = {}
        self.seat_action_labels = {}
        self.seat_stack_labels = {}
        self.seat_showdown_card_labels = {} # Clear new labels too
        self.player_card_labels = []
        self.community_card_labels = []
        self.action_buttons = {}
        self.last_round_active_players = []

        self._create_config_window()


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    # Optional: Apply a theme for better visuals if available (e.g., 'clam', 'alt', 'default', 'classic')
    # try:
    #     style = ttk.Style(root)
    #     style.theme_use('clam') # Or another theme
    # except tk.TclError:
    #     print("Themes not available on this system.")
    app = PokerApp(root)
    root.mainloop()
