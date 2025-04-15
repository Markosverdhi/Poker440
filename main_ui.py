# filename: package/main_ui.py
"""
Main UI for Poker Game using Gymnasium-compliant Environment (Tournament Structure).

MODIFIED (ASCII Card Display):
- Uses card_utils.render_card_ascii for card display.
- Sets a monospace font for card labels for better alignment.
- Configures card labels (height, justify) for multi-line display.
- Clears showdown labels properly in _update_ui.
- Displays all player cards (if available in info['showdown_hands'])
  in the showdown labels during _display_round_results.
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import os
import torch
import random
import json
import numpy as np
import time # For potential delays

# Import Gymnasium-compliant environment and updated utils
try:
    from envs import TrainFullPokerEnv, evaluate_hand
    from utils import encode_obs_eval, NEW_STATE_DIM
    # Use updated card_utils for rendering
    from card_utils import render_card_ascii, render_hand_for_labels, render_community_cards_for_labels, render_hand
    from seat_config import SeatConfigManager
except ImportError as e:
    print(f"ERROR: Ensure envs.py, utils.py, card_utils.py, seat_config.py are available: {e}")
    exit()

# Assuming models.py is available and updated for NEW_STATE_DIM
try:
    from models import BestPokerModel
except ImportError:
    print("ERROR: Ensure models.py is available.")
    exit()


# --- Constants ---
NUM_PLAYERS = 6
STATE_DIM = NEW_STATE_DIM
# Define a monospace font for card rendering
MONOSPACE_FONT = ("Consolas", 10) # Or "Courier New", "Lucida Console"

# --- Helper: Opponent Policy Creation (Unchanged) ---
def get_opponent_policy(opponent_type, agent_model, action_list, num_actions):
    action_index_to_str = {i: s for i, s in enumerate(action_list)}
    if opponent_type == "model":
        if not agent_model: print("Warning: 'model' opponent selected but no model loaded. Using 'random'."); return get_opponent_policy("random", None, action_list, num_actions)
        def policy_fn(obs_dict):
            if not isinstance(obs_dict, dict): return 'fold'
            legal_actions = obs_dict.get('legal_actions', []);
            if not legal_actions: return 'fold'
            try: state = encode_obs_eval(obs_dict)
            except Exception as e: print(f"Error encoding opponent obs: {e}. Folding."); return 'fold'
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0); agent_model.eval()
            with torch.no_grad(): q_values = agent_model(state_tensor)
            q_values_np = q_values.squeeze().cpu().numpy(); sorted_indices = np.argsort(q_values_np)[::-1]
            for action_idx in sorted_indices:
                if 0 <= action_idx < num_actions:
                    action_str = action_index_to_str.get(action_idx)
                    if action_str and action_str in legal_actions: return action_str
            if 'check' in legal_actions: return 'check'
            if 'call' in legal_actions: return 'call'
            if 'fold' in legal_actions: return 'fold'
            return random.choice(legal_actions) if legal_actions else 'fold'
        return policy_fn
    elif opponent_type == "random":
        def policy_fn(obs_dict):
            if not isinstance(obs_dict, dict): return 'fold'
            legal = obs_dict.get("legal_actions", []); return random.choice(legal) if legal else "fold"
        return policy_fn
    elif opponent_type == "variable":
        model_policy = get_opponent_policy("model", agent_model, action_list, num_actions); random_policy = get_opponent_policy("random", None, action_list, num_actions)
        def policy_fn(obs_dict): return model_policy(obs_dict) if random.random() < 0.5 else random_policy(obs_dict)
        return policy_fn
    else: print(f"Warning: Unknown opponent type '{opponent_type}'. Using 'random'."); return get_opponent_policy("random", None, action_list, num_actions)


class PokerApp:
    """ Main class for the Tkinter Poker Application (Tournament Adapted) """
    def __init__(self, root):
        self.root = root
        self.root.withdraw()

        self.env = None
        self.agent_model = None
        self.seat_configs = {}
        self.checkpoint_path = tk.StringVar(value="checkpoints/final_agent_model.pt")

        self.game_window = None
        self.config_window = None
        self.seat_config_manager = SeatConfigManager(num_players=NUM_PLAYERS)

        # UI Elements
        self.seat_frames = {}
        self.seat_status_labels = {}
        self.seat_action_labels = {}
        self.seat_stack_labels = {}
        self.seat_showdown_card_labels = {} # Label to show opponent cards at showdown
        self.player_card_labels = [] # Labels for human player's hand
        self.community_card_labels = [] # Labels for community cards
        self.pot_label = None
        self.turn_label = None
        self.action_buttons = {}
        self.player_hand_frame = None
        self.status_bar = None

        # Game state tracking
        self.current_encoded_state = None
        self.human_player_seat = -1
        self.action_list = []
        self.num_actions = 0
        self._action_string_to_idx = {}
        self.last_info = {}
        self.tournament_running = False

        self._create_config_window()

    # --- Configuration and Setup ---

    def _validate_checkpoint_path(self, suffix_or_path):
        # (Unchanged)
        if not suffix_or_path: return None
        if "/" not in suffix_or_path and "\\" not in suffix_or_path:
             default_dir = "checkpoints";
             if os.path.isdir(default_dir): return os.path.join(default_dir, suffix_or_path)
        return suffix_or_path

    def _load_model(self, path, state_dim, num_actions):
        # (Unchanged from previous fix)
        if not path or not os.path.exists(path): messagebox.showerror("Error", f"Checkpoint not found: {path}"); return None
        try:
            model = BestPokerModel(num_actions=num_actions) # Uses NEW_STATE_DIM internally
            checkpoint_data = torch.load(path, map_location=torch.device('cpu'))
            state_dict = None
            if isinstance(checkpoint_data, dict): state_dict = checkpoint_data.get('agent_state_dict', checkpoint_data.get('state_dict', checkpoint_data))
            else: state_dict = checkpoint_data
            if not state_dict: raise TypeError("Could not find state_dict in checkpoint")
            cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(cleaned_state_dict, strict=False); model.eval()
            print(f"Successfully loaded model from {path} (Input Dim: {model.input_dim})")
            return model
        except Exception as e: messagebox.showerror("Error", f"Failed to load checkpoint {path}: {e}"); return None

    def _create_config_window(self):
        # (Unchanged)
        if self.config_window: self.config_window.lift(); return
        self.config_window = tk.Toplevel(self.root); self.config_window.title("Poker Game Configuration"); self.config_window.protocol("WM_DELETE_WINDOW", self.root.quit)
        main_frame = ttk.Frame(self.config_window, padding="10"); main_frame.grid(row=0, column=0, sticky="nsew")
        ttk.Label(main_frame, text="Configure Seats:", font="-weight bold").grid(row=0, column=0, columnspan=2, pady=(0, 10))
        self.seat_vars = []
        options = self.seat_config_manager.get_options();
        if 'human' not in options: options.insert(0,'human')
        for i in range(NUM_PLAYERS):
            ttk.Label(main_frame, text=f"Seat {i+1}:").grid(row=i+1, column=0, sticky=tk.W, padx=5)
            default_val = 'human' if i == 0 else 'model'; var = tk.StringVar(value=default_val); dropdown = ttk.OptionMenu(main_frame, var, default_val, *options); dropdown.grid(row=i+1, column=1, sticky="ew", padx=5); self.seat_vars.append(var)
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(row=NUM_PLAYERS + 1, column=0, columnspan=2, sticky="ew", pady=10)
        ttk.Label(main_frame, text="Model Checkpoint:").grid(row=NUM_PLAYERS + 2, column=0, sticky=tk.W, padx=5)
        checkpoint_entry = ttk.Entry(main_frame, textvariable=self.checkpoint_path, width=40); checkpoint_entry.grid(row=NUM_PLAYERS + 2, column=1, sticky="ew", padx=5)
        start_button = ttk.Button(main_frame, text="Start Game", command=self._start_game); start_button.grid(row=NUM_PLAYERS + 3, column=0, columnspan=2, pady=(10, 0))
        self.config_window.resizable(False, False)

    def _start_game(self):
        # (Unchanged from previous fix)
        selected_types = [var.get() for var in self.seat_vars]; human_count = selected_types.count('human')
        if human_count == 0: messagebox.showerror("Config Error", "No seat assigned as 'human'."); return
        if human_count > 1: messagebox.showerror("Config Error", "More than one seat assigned as 'human'."); return
        self.seat_configs = {}; self.human_player_seat = -1
        for i, seat_type in enumerate(selected_types): self.seat_configs[i] = seat_type;
        if seat_type == "human": self.human_player_seat = i
        print(f"Seat Configurations: {self.seat_configs}, Human Seat (Agent ID): {self.human_player_seat}")
        try:
             self.env = TrainFullPokerEnv(num_players=NUM_PLAYERS, agent_id=self.human_player_seat, render_mode="human")
             self.action_list = self.env.action_list; self.num_actions = self.env.action_space.n; self._action_string_to_idx = {s: i for i, s in enumerate(self.action_list)}
             if self.env.observation_space.shape[0] != STATE_DIM: messagebox.showerror("Config Error", f"Env obs space ({self.env.observation_space.shape[0]}) != utils STATE_DIM ({STATE_DIM})."); self.env.close(); self.env = None; return
        except Exception as e: messagebox.showerror("Error", f"Failed to create environment: {e}"); self.env = None; return
        self.agent_model = None; needs_model = any(stype == "model" for stype in self.seat_configs.values())
        if needs_model:
             checkpoint_input = self.checkpoint_path.get(); full_checkpoint_path = self._validate_checkpoint_path(checkpoint_input)
             if not full_checkpoint_path or not os.path.exists(full_checkpoint_path): messagebox.showerror("Config Error", f"Model checkpoint not found: {full_checkpoint_path}"); self.env.close(); self.env = None; return
             self.agent_model = self._load_model(full_checkpoint_path, STATE_DIM, self.num_actions)
             if not self.agent_model: self.env.close(); self.env = None; return
        else: print("No 'model' opponents selected.")
        for i in range(NUM_PLAYERS):
             if i == self.human_player_seat: continue
             seat_type = self.seat_configs[i]; policy_func = get_opponent_policy(seat_type, self.agent_model, self.action_list, self.num_actions); self.env.set_opponent_policy(i, policy_func); print(f"Set Seat {i+1} policy to: {seat_type}")
        if self.config_window: self.config_window.destroy(); self.config_window = None
        self._create_game_window()
        self.tournament_running = True; self.status_bar.config(text="Starting tournament...")
        try:
            self.current_encoded_state, self.last_info = self.env.reset()
            if self.last_info.get("error"): messagebox.showerror("Error", f"Environment reset failed: {self.last_info['error']}"); self._reconfigure(); return
            self._update_ui(); self.root.after(100, self._process_game_turn)
        except Exception as e: messagebox.showerror("Error", f"Failed during environment reset: {e}"); self._reconfigure()


    def _create_game_window(self):
        """Creates the main game window UI elements."""
        if self.game_window: self.game_window.lift(); return
        self.game_window = tk.Toplevel(self.root); self.game_window.title("Poker Game (Tournament Mode)"); self.game_window.geometry("950x800"); self.game_window.protocol("WM_DELETE_WINDOW", self.root.quit)

        # Style
        style = ttk.Style()
        style.configure("Card.TLabel", font=MONOSPACE_FONT, padding=2, anchor="center", justify="center")
        style.configure("PlayerHand.TLabel", font=(MONOSPACE_FONT[0], 12, "bold"), padding=3, anchor="center", justify="center") # Slightly larger for player hand
        style.configure("Showdown.TLabel", font=MONOSPACE_FONT, padding=1, anchor="nw", justify="left", foreground="darkblue") # Smaller for showdown
        style.configure("Bold.TLabel", font="-weight bold")

        # Frames
        top_frame = ttk.Frame(self.game_window, padding=5); top_frame.pack(fill=tk.X)
        middle_frame = ttk.Frame(self.game_window, padding=10); middle_frame.pack(fill=tk.BOTH, expand=True)
        bottom_frame = ttk.Frame(self.game_window, padding=10); bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # Top elements
        self.pot_label = ttk.Label(top_frame, text="Pot: $0", style="Bold.TLabel", font=("", 12)); self.pot_label.pack(side=tk.LEFT, padx=20)
        self.turn_label = ttk.Label(top_frame, text="Turn: -", style="Bold.TLabel", font=("", 12), foreground="blue"); self.turn_label.pack(side=tk.RIGHT, padx=20)

        # Middle elements (Seats and Table)
        left_seats_frame = ttk.Frame(middle_frame, padding=5); left_seats_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), anchor='n')
        table_frame = ttk.Frame(middle_frame, padding=10); table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_seats_frame = ttk.Frame(middle_frame, padding=5); right_seats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0), anchor='n')

        self.seat_frames = {}; self.seat_status_labels = {}; self.seat_action_labels = {}; self.seat_stack_labels = {}; self.seat_showdown_card_labels = {}
        seat_width = 160 # Increased width slightly
        for i in range(NUM_PLAYERS):
            parent_frame = left_seats_frame if i < 3 else right_seats_frame
            seat_frame = ttk.LabelFrame(parent_frame, text=f"Seat {i+1}", padding=10, width=seat_width); seat_frame.pack(pady=5, fill=tk.X, anchor='n'); seat_frame.pack_propagate(False); self.seat_frames[i] = seat_frame
            status_txt = f"Type: {self.seat_configs.get(i, 'N/A').title()}" + (" (You)" if i == self.human_player_seat else "")
            status_label = ttk.Label(seat_frame, text=status_txt); status_label.pack(anchor=tk.W); self.seat_status_labels[i] = status_label
            stack_label = ttk.Label(seat_frame, text="Stack: $0"); stack_label.pack(anchor=tk.W); self.seat_stack_labels[i] = stack_label
            action_label = ttk.Label(seat_frame, text="Last Action: -", foreground="gray", wraplength=seat_width-20); action_label.pack(anchor=tk.W); self.seat_action_labels[i] = action_label
            # Showdown label - uses monospace font via style
            showdown_label = ttk.Label(seat_frame, text="", style="Showdown.TLabel", height=6); # Height for ASCII card
            showdown_label.pack(anchor=tk.W, pady=(5,0)); self.seat_showdown_card_labels[i] = showdown_label

        # Community Cards Frame
        community_frame = ttk.LabelFrame(table_frame, text="Community Cards", padding=10); community_frame.pack(pady=10, anchor='center')
        self.community_card_labels = []
        for _ in range(5):
            # Use styled label for community cards
            lbl = ttk.Label(community_frame, text="", style="Card.TLabel", relief="ridge", width=7, height=5); # Width/Height for ASCII card
            lbl.pack(side=tk.LEFT, padx=3); self.community_card_labels.append(lbl)

        # Player Hand Frame
        self.player_hand_frame = ttk.LabelFrame(table_frame, text="Your Hand (Seat ?)", padding=10); self.player_hand_frame.pack(pady=10, anchor='center')
        self.player_card_labels = []
        for _ in range(2):
             # Use styled label for player cards
             lbl = ttk.Label(self.player_hand_frame, text="", style="PlayerHand.TLabel", relief="solid", width=7, height=5); # Width/Height for ASCII card
             lbl.pack(side=tk.LEFT, padx=5); self.player_card_labels.append(lbl)

        # Bottom elements (Actions, Controls, Status)
        action_frame = ttk.Frame(bottom_frame); action_frame.pack(pady=(0,10))
        self.action_buttons = {}
        button_actions = self.action_list if self.action_list else ['fold', 'call', 'check', 'bet_small', 'bet_big', 'all_in']
        for action in button_actions:
            btn_text = action.replace('_', ' ').title(); btn = ttk.Button(action_frame, text=btn_text, width=10, state=tk.DISABLED, command=lambda a=action: self._handle_human_action(a)); btn.pack(side=tk.LEFT, padx=5); self.action_buttons[action] = btn
        control_frame = ttk.Frame(bottom_frame); control_frame.pack(pady=(5,0))
        reconfig_button = ttk.Button(control_frame, text="New Game / Reconfigure", command=self._reconfigure); reconfig_button.pack(side=tk.LEFT, padx=10)
        exit_button = ttk.Button(control_frame, text="Exit Application", command=self.root.quit); exit_button.pack(side=tk.LEFT, padx=10)
        self.status_bar = ttk.Label(bottom_frame, text="Welcome!", relief=tk.SUNKEN, anchor=tk.W, padding=2); self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5,0))

    # --- Game Flow Logic (Unchanged from previous fix) ---
    def _process_game_turn(self):
        # (Unchanged)
        if not self.env or not self.tournament_running: return
        if hasattr(self.env, 'tournament_over') and self.env.tournament_over: self._handle_tournament_end(self.last_info); return
        current_player_id = self.env.current_player_id if hasattr(self.env, 'current_player_id') else None
        if current_player_id is not None:
             turn_text = f"Turn: Seat {current_player_id + 1}" + (" (You)" if current_player_id == self.human_player_seat else "")
             fg_color = "blue" if current_player_id == self.human_player_seat else "black"; self.turn_label.config(text=turn_text, foreground=fg_color)
             for i, frame in self.seat_frames.items(): frame.config(relief="sunken" if i == current_player_id else "flat")
        else: self.turn_label.config(text="Turn: -", foreground="black")
        if current_player_id == self.human_player_seat:
            try: legal_actions = self.env.get_legal_actions_for_agent()
            except Exception as e: print(f"Error getting legal actions: {e}"); messagebox.showerror("Error", f"Could not get legal actions: {e}"); legal_actions = []
            self.status_bar.config(text="Your turn.")
            for action, button in self.action_buttons.items(): button.config(state=tk.NORMAL if action in legal_actions else tk.DISABLED)
        else:
            self.status_bar.config(text=f"Waiting for Seat {(current_player_id + 1) if current_player_id is not None else '?' }...")
            for button in self.action_buttons.values(): button.config(state=tk.DISABLED)
            self.root.after(200, lambda: self._handle_ai_action()) # Slightly longer delay for AI

    def _handle_ai_action(self):
        # (Unchanged)
        if not self.tournament_running or not hasattr(self.env, 'current_player_id') or self.env.current_player_id == self.human_player_seat: return
        self._step_env(-1)

    def _handle_human_action(self, action_str):
        # (Unchanged)
        if not self.env or not self.tournament_running or not hasattr(self.env, 'current_player_id') or self.env.current_player_id != self.human_player_seat: return
        print(f"Human (Seat {self.human_player_seat + 1}) chose action: {action_str}")
        action_idx = self._action_string_to_idx.get(action_str, -1)
        if action_idx == -1: messagebox.showerror("Internal Error", f"Invalid action mapping for {action_str}"); return
        for btn in self.action_buttons.values(): btn.config(state=tk.DISABLED)
        self.status_bar.config(text=f"You chose {action_str}. Processing...")
        self._step_env(action_idx)

    def _step_env(self, action_idx):
        # (Unchanged)
        if not self.tournament_running: return
        try:
            next_encoded_state, reward, terminated, truncated, info = self.env.step(action_idx)
            self.current_encoded_state = next_encoded_state; self.last_info = info; done = terminated or truncated
            self._update_ui()
            delay_ms = 200 # Base delay
            if info.get('round_over', False): self._display_round_results(info); delay_ms = 1500 # Longer delay after round results
            if done: self._handle_tournament_end(info)
            else: self.root.after(delay_ms, self._process_game_turn)
        except Exception as e:
            print(f"Error during env.step(): {e}"); messagebox.showerror("Environment Error", f"An error occurred: {e}")
            self.tournament_running = False; self._reconfigure()

    def _update_ui(self):
        """ Updates all UI elements based on env state/last_info. """
        if not self.game_window or not self.env or not self.last_info: return

        # Get state from last info dict or env helpers
        stacks = self.last_info.get('stacks', {})
        pot = self.last_info.get('pot', 0)
        community_cards = self.last_info.get('community_cards', [])
        # current_bets = self.last_info.get('current_bets', {}) # Maybe display this per player?
        active_players = self.last_info.get('active_players', [])
        stage = self.last_info.get('stage', 'N/A')
        button_pos = self.last_info.get('button_pos', -1)
        human_hand = self.env.get_player_hand(self.human_player_seat) if self.human_player_seat != -1 else []

        # Update Pot
        self.pot_label.config(text=f"Pot: ${pot:.2f}")

        # Clear previous round's showdown hands UNLESS round just ended
        round_just_ended = self.last_info.get('round_over', False)
        if not round_just_ended:
             for i in range(NUM_PLAYERS):
                  if i in self.seat_showdown_card_labels:
                       self.seat_showdown_card_labels[i].config(text="") # Clear old cards

        # Update Seat Information
        for i in range(NUM_PLAYERS):
            stack = stacks.get(i, 0)
            current_bet = self.last_info.get('current_bets', {}).get(i, 0) # Get current bet for display
            status_text = f"Type: {self.seat_configs.get(i, 'N/A').title()}"
            player_status = ""
            is_player_active = i in active_players

            if i == self.human_player_seat: status_text += " (You)"
            if not is_player_active and i in self.env.hands: player_status = " (Folded)"
            elif stack <= 0 and is_player_active: player_status = " (All-In)"
            elif not is_player_active: player_status = " (Out)"
            if self.seat_configs.get(i) == 'empty': status_text = "Seat Empty"; stack = 0; player_status=""
            if i == button_pos: status_text += " (BTN)"

            if i in self.seat_status_labels: self.seat_status_labels[i].config(text=status_text + player_status)
            # Display stack and current bet
            stack_bet_text = f"Stack: ${stack:.2f}"
            if current_bet > 0: stack_bet_text += f" (Bet: ${current_bet:.2f})"
            if i in self.seat_stack_labels: self.seat_stack_labels[i].config(text=stack_bet_text)

            # Clear action label if player folded/out this turn? Optional.
            if not is_player_active and i in self.seat_action_labels:
                 self.seat_action_labels[i].config(text="Last Action: -", foreground="gray")


        # Update Community Cards using ASCII art renderer
        rendered_community = render_community_cards_for_labels(community_cards)
        for i, label in enumerate(self.community_card_labels):
            label.config(text=rendered_community[i])

        # Update Player Hand Display using ASCII art renderer
        if self.human_player_seat != -1:
            self.player_hand_frame.config(text=f"Your Hand (Seat {self.human_player_seat + 1})")
            rendered_hand = render_hand_for_labels(human_hand)
            for i, label in enumerate(self.player_card_labels):
                label.config(text=rendered_hand[i])

        self.root.update_idletasks() # Force UI update


    def _display_round_results(self, round_info):
        """ Displays results of a completed round, including showdown hands. """
        print("--- Displaying Round Results ---") # Debug
        winners = round_info.get('winners', [])
        showdown_hands = round_info.get('showdown_hands', {}) # Dict: pid -> {'hand':[], 'score':S, 'desc':D}
        round_reward = round_info.get('round_reward', 0.0)
        final_pot = round_info.get('final_pot', self.last_info.get('pot', 0))

        # Display Showdown Hands in Seat Frames using simple text render
        for pid, hand_data in showdown_hands.items():
             if pid in self.seat_showdown_card_labels:
                  # Use simple render_hand for compactness in showdown label
                  hand_str = render_hand(hand_data.get('hand', ["?", "?"]), separator=" ")
                  desc_str = hand_data.get('desc', 'Unknown')
                  # Showdown label now uses monospace font via style
                  self.seat_showdown_card_labels[pid].config(text=f"Showdown:\n{hand_str}\n({desc_str})")

        # Update status bar with winner info
        if winners:
             win_amount = final_pot / len(winners) if winners else 0
             winner_seats = [w + 1 for w in winners]
             status_msg = f"Round Over. Winner(s): Seat(s) {winner_seats} (${win_amount:.2f} each)."
             if self.human_player_seat != -1: status_msg += f" Your Round Reward: ${round_reward:.2f}"
             self.status_bar.config(text=status_msg)
        else:
             self.status_bar.config(text="Round Over. No winner determined?")

        self.root.update_idletasks() # Ensure UI updates


    def _handle_tournament_end(self, final_info):
        # (Unchanged)
        print("\n--- Tournament Ended (UI) ---"); self.tournament_running = False
        for btn in self.action_buttons.values(): btn.config(state=tk.DISABLED); self.turn_label.config(text="Tournament Over", foreground="darkred")
        final_stacks = final_info.get('stacks', {}); winner_id = -1; max_stack = -1; active_players = [p for p,s in final_stacks.items() if s > 0]
        if len(active_players) == 1: winner_id = active_players[0]
        elif not active_players: winner_id = -2
        else:
             for pid, stack in final_stacks.items():
                  if stack > max_stack: max_stack = stack; winner_id = pid
        result_message = "Tournament Over!\n\n"
        if winner_id >= 0: win_msg = f"Seat {winner_id + 1} wins!";
        if winner_id == self.human_player_seat: win_msg += " Congratulations!"; result_message += win_msg + "\n"
        elif self.human_player_seat != -1 and final_stacks.get(self.human_player_seat, 0) <= 0: result_message += "You were eliminated.\n"
        else: result_message += "Tournament ended.\n"
        result_message += "\nFinal Stacks:\n"
        for pid in range(NUM_PLAYERS):
             if self.seat_configs.get(pid) != 'empty': result_message += f"  Seat {pid+1}: ${final_stacks.get(pid, 0):.2f}\n"
        self.root.after(200, lambda msg=result_message: self._prompt_new_game(msg))

    def _prompt_new_game(self, result_message):
        # (Unchanged)
        response = messagebox.askyesno("Tournament Over", result_message + "\nStart a new tournament with the same configuration?")
        if response is True: self._start_game()
        else: self._reconfigure()

    def _reconfigure(self):
        # (Unchanged)
        self.tournament_running = False
        if self.game_window: self.game_window.destroy(); self.game_window = None
        if self.env:
             try: self.env.close()
             except Exception as e: print(f"Error closing env: {e}")
             self.env = None
        self.agent_model = None; self.seat_configs = {}; self.current_encoded_state = None; self.human_player_seat = -1; self.last_info = {}
        self.seat_frames = {}; self.seat_status_labels = {}; self.seat_action_labels = {}; self.seat_stack_labels = {}; self.seat_showdown_card_labels = {}
        self.player_card_labels = []; self.community_card_labels = []; self.action_buttons = {}
        self._create_config_window()


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    try: style = ttk.Style(root); style.theme_use('clam') # Or 'vista', 'xpnative', 'default'
    except tk.TclError: print("Themes not available or theme failed.")
    app = PokerApp(root)
    root.mainloop()
