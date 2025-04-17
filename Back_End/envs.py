import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import Counter, defaultdict

# Assuming card_utils provides these (or define them here)
# from card_utils import SUITS_UNICODE, RANKS
SUITS = ['H', 'D', 'C', 'S']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
DECK = [r + s for s in SUITS for r in RANKS]

try:
    # Use the improved evaluator
    from .improved_hand_evaluator import evaluate_hand as evaluate_hand_improved
except ImportError:
    print("Warning: Could not import improved_hand_evaluator. Using placeholder.")
    # Fallback placeholder if import fails
    def evaluate_hand_improved(hole_cards, community_cards):
        # Extremely simplified placeholder
        score = len(hole_cards) + len(community_cards)
        return score, f"Placeholder Score: {score}"


class BaseFullPokerEnv(gym.Env):
    """
    Base Poker Environment for 6 players (Full Deck) - Tournament Structure.

    Observation Space:
        A flattened vector representing cards, pot, stacks, bets, position etc.
        Size defined by NEW_STATE_DIM in utils.py (e.g., 333).
        Encoding handled by utils.encode_obs.

    Action Space:
        Discrete space representing poker actions.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # *** THIS IS THE CORRECTED __init__ SIGNATURE ***
    def __init__(self, num_players=6, agent_id=0, starting_stack=10000, small_blind=50, big_blind=100, render_mode=None, seat_config=None): # Added seat_config
        super().__init__()

        self.num_players = num_players
        self.agent_id = agent_id # The ID of the RL agent player
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind

        # Store seat configuration to know which seats are active
        self.seat_config = seat_config if seat_config else {i: 'model' for i in range(num_players)} # Default if not provided
        if agent_id is not None and agent_id in self.seat_config:
             # Ensure agent's seat type in the config reflects the agent_id passed
             # Overwrite if necessary, assuming agent_id is the source of truth for the agent's seat
             self.seat_config[agent_id] = 'agent'

        # --- State Attributes ---
        self.deck = []
        self.stacks = {} # Player ID -> Stack size (initialized in reset)
        self.hands = {} # Player ID -> List of hole cards
        self.community_cards = []
        self.pot = 0
        self.current_bets = {} # Player ID -> Bet amount in current round
        self.round_total_bets = {} # Player ID -> Total bet amount in this hand
        # Store initial stacks at the start of a round/agent turn for reward calculation
        self.initial_stacks_this_round = {}
        self.active_players_in_round = set() # Players still active (not folded) in the current hand
        self.current_player_id = None
        self.stage = None # 'preflop', 'flop', 'turn', 'river', 'showdown'
        self.button_pos = 0 # Index of the player with the button
        self.min_raise = big_blind
        self.last_raiser = None # ID of the last player who raised/bet
        self.tournament_over = False
        self.round_over = False # Flag specific to the current hand/round
        self.players_acted_this_round = set() # Track players who acted in the current betting round

        # Opponent policies (set externally)
        self.opponent_policies = {} # Player ID -> policy function

        # --- Action Space Definition ---
        # Example actions - adjust as needed
        self.action_list = ['fold', 'call', 'check', 'bet_small', 'bet_big', 'all_in']
        self.action_space = spaces.Discrete(len(self.action_list))
        self._action_to_string = {i: s for i, s in enumerate(self.action_list)}
        self._string_to_action = {s: i for i, s in enumerate(self.action_list)}

        # --- Observation Space Definition ---
        # Import the dimension defined in utils.py
        try:
            # Ensure relative import works if envs.py is in Back_End
            from .constants import NEW_STATE_DIM
            self.observation_space_dim = NEW_STATE_DIM
        except ImportError:
            print("Warning: Could not import NEW_STATE_DIM from .constants. Using default 333.")
            self.observation_space_dim = 333 # Fallback, ensure constants.py is accessible

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.observation_space_dim,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        # Add rendering elements if needed (e.g., pygame)


    def set_opponent_policy(self, player_id, policy_func):
        """ Sets the policy function for a specific opponent """
        if player_id != self.agent_id and self.seat_config.get(player_id) != 'empty': # Only set for non-empty opponents
            self.opponent_policies[player_id] = policy_func

    def _get_playing_players(self):
        """ Returns a list of player IDs who are not 'empty'. """
        return [i for i, seat_type in self.seat_config.items() if seat_type != 'empty']

    def _get_active_players(self, include_all_in=True):
        """ Returns a list of player IDs who have chips > 0 and are not 'empty'. """
        playing_players = self._get_playing_players()
        active = []
        for i in playing_players:
            if self.stacks.get(i, 0) > 0:
                 active.append(i)
            elif include_all_in and i in self.active_players_in_round and self.stacks.get(i, 0) <= 0:
                 # Include players who are all-in but still in the hand
                 active.append(i)
        return active

    def _get_next_active_player_id(self, start_id):
        """ Finds the next player ID in rotation who is still active in the round and not empty. """
        if not self.active_players_in_round: return None # No active players left

        playing_players = self._get_playing_players()
        if not playing_players: return None # No playing players

        # Find index of start_id in the playing player list (circular)
        try:
            current_start_id = start_id if start_id is not None else self.button_pos # Handle None case
            start_idx = playing_players.index(current_start_id)
        except ValueError: # start_id not in playing_players (e.g., was empty or invalid)
            # Fallback: find the first active player after the button among playing players
            current_id = self.button_pos
            for _ in range(self.num_players + 1): # Limit loop iterations
                current_id = (current_id + 1) % self.num_players
                if current_id in playing_players and current_id in self.active_players_in_round:
                    # Check if they can act (have chips)
                    if self.stacks.get(current_id, 0) > 0:
                        return current_id
            return None # No suitable player found

        # Iterate through playing players circularly from start_idx + 1
        num_playing = len(playing_players)
        for i in range(1, num_playing + 1): # Check each player once
            current_idx = (start_idx + i) % num_playing
            current_id = playing_players[current_idx]
            # Must be active in the current hand AND have chips remaining to act
            if current_id in self.active_players_in_round and self.stacks.get(current_id, 0) > 0:
                 return current_id

        # If loop completes without finding next player (e.g., all remaining are all-in), return None.
        return None # Indicates action might be closed or no one can act

    def _deal_cards(self):
        """ Deals hole cards and community cards based on stage """
        playing_players = self._get_playing_players() # Only deal to non-empty seats

        if self.stage == 'preflop':
            self.hands = {}
            # Only deal to players with chips at the start of the hand
            active_at_start = [p for p in playing_players if self.initial_stacks_this_round.get(p, 0) > 0]
            for _ in range(2): # Deal 2 hole cards
                for player_id in active_at_start:
                    if player_id not in self.hands: self.hands[player_id] = []
                    if self.deck: self.hands[player_id].append(self.deck.pop())
            # print(f"Dealt Hands: {self.hands}") # Debug
        elif self.stage == 'flop':
            if len(self.community_cards) == 0: # Only deal flop if no community cards yet
                if self.deck: self.deck.pop() # Burn card
                self.community_cards = [self.deck.pop() for _ in range(3) if self.deck]
                # print(f"Dealt Flop: {self.community_cards}") # Debug
        elif self.stage == 'turn':
             if len(self.community_cards) == 3: # Only deal turn if flop is out
                if self.deck: self.deck.pop() # Burn card
                if self.deck: self.community_cards.append(self.deck.pop())
                # print(f"Dealt Turn: {self.community_cards}") # Debug
        elif self.stage == 'river':
             if len(self.community_cards) == 4: # Only deal river if turn is out
                if self.deck: self.deck.pop() # Burn card
                if self.deck: self.community_cards.append(self.deck.pop())
                # print(f"Dealt River: {self.community_cards}") # Debug

    def _start_new_round(self):
        """ Initializes a new hand/round within the tournament. """
        self.round_over = False
        # Get players who are playing AND have chips
        active_tournament_players = self._get_active_players(include_all_in=False)
        if len(active_tournament_players) <= 1:
            print("Tournament ended: Not enough active players to start a new round.")
            self.tournament_over = True
            return # Don't start round if tournament is over

        # print(f"\n--- Starting Round ---") # Less verbose
        self.deck = DECK[:]
        random.shuffle(self.deck)
        self.community_cards = []
        self.pot = 0
        self.current_bets = {i: 0 for i in self._get_playing_players()} # Only track playing players
        self.round_total_bets = {i: 0 for i in self._get_playing_players()}
        self.active_players_in_round = set(active_tournament_players) # Players starting the round
        self.last_raiser = None
        self.stage = 'preflop'
        self.initial_stacks_this_round = self.stacks.copy() # Store stacks at round start
        self.players_acted_this_round = set() # Reset who has acted

        # --- Determine positions (handle small number of players & empty seats) ---
        playing_players_sorted = sorted(self._get_playing_players())
        num_playing = len(playing_players_sorted)

        current_button_idx = -1
        if self.button_pos in playing_players_sorted:
            current_button_idx = playing_players_sorted.index(self.button_pos)
        else: # Button was on an empty seat, find previous playing player
             temp_pos = self.button_pos
             for _ in range(self.num_players):
                  temp_pos = (temp_pos - 1 + self.num_players) % self.num_players
                  if temp_pos in playing_players_sorted:
                       current_button_idx = playing_players_sorted.index(temp_pos)
                       break
             if current_button_idx == -1: current_button_idx = 0 # Default if none found

        # Find next playing player for button
        next_button_idx = (current_button_idx + 1) % num_playing
        self.button_pos = playing_players_sorted[next_button_idx]

        # Find SB and BB among active players (in the round) relative to new button
        active_list_sorted = sorted(list(self.active_players_in_round))
        num_active_in_round = len(active_list_sorted)

        # Ensure there are enough active players to assign blinds
        if num_active_in_round < 2:
             print("Error: Less than 2 active players to assign blinds. Ending tournament.")
             self.tournament_over = True
             return

        button_idx_in_active = -1
        try:
             button_idx_in_active = active_list_sorted.index(self.button_pos)
        except ValueError: # Button player might not be active (e.g., just busted but tournament continues)
             # Find first active player index as reference
             button_idx_in_active = 0 # Default to first active player

        if num_active_in_round == 2: # Heads up posting logic
             sb_player = active_list_sorted[button_idx_in_active]
             bb_player = active_list_sorted[(button_idx_in_active + 1) % num_active_in_round]
             self.current_player_id = sb_player # Button/SB acts first preflop HU
        else: # 3+ players
             sb_player = active_list_sorted[(button_idx_in_active + 1) % num_active_in_round]
             bb_player = active_list_sorted[(button_idx_in_active + 2) % num_active_in_round]
             next_player_idx = (button_idx_in_active + 3) % num_active_in_round
             self.current_player_id = active_list_sorted[next_player_idx] # UTG

        # Post blinds
        sb_amount = min(self.small_blind, self.stacks.get(sb_player, 0))
        bb_amount = min(self.big_blind, self.stacks.get(bb_player, 0))
        self._post_bet(sb_player, sb_amount)
        self._post_bet(bb_player, bb_amount)
        self.last_raiser = bb_player # BB is the initial "raiser"
        self.min_raise = self.big_blind # Initial min raise amount
        self.players_acted_this_round.add(sb_player) # Blinds count as acting
        self.players_acted_this_round.add(bb_player)

        # Deal cards
        self._deal_cards()
        # print(f"Button: {self.button_pos}, SB: {sb_player}, BB: {bb_player}, Start Player: {self.current_player_id}") # Debug


    def reset(self, seed=None, options=None):
        """ Resets the environment to start a new tournament. """
        super().reset(seed=seed)
        print("\n===== RESETTING TOURNAMENT =====")

        # Initialize tournament state - ONLY for non-empty seats
        self.stacks = {}
        playing_players = self._get_playing_players()
        if not playing_players:
             # Handle error: No players configured to play
             print("ERROR: No playing players found in seat configuration during reset.")
             # Return a default state indicating immediate end?
             self.tournament_over = True
             # Return format needs observation and info
             zero_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
             error_info = self._get_info(error="No playing players configured.")
             return zero_obs, error_info


        for i in range(self.num_players):
            if i in playing_players:
                self.stacks[i] = self.starting_stack
            else:
                self.stacks[i] = 0 # Ensure empty seats have 0 stack

        # Random initial button among playing players
        self.button_pos = random.choice(playing_players) if playing_players else 0

        self.tournament_over = False
        self.round_over = True # Ensure a new round starts

        # Start the first round
        self._start_new_round()

        # Get initial observation and info
        # Handle case where tournament might end immediately (e.g., only 1 player)
        if self.tournament_over:
             observation = self._get_obs(self.agent_id) # Get obs even if over
             info = self._get_info(round_over=True) # Indicate round is also over
             info['error'] = "Tournament ended immediately on reset."
             info['terminated'] = True # Explicitly set terminated flag
        else:
             observation = self._get_obs(self.agent_id)
             info = self._get_info()


        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _post_bet(self, player_id, amount):
        """ Handles posting a bet/blind/call """
        if player_id not in self.stacks or self.seat_config.get(player_id) == 'empty': return # Safety check
        actual_bet = min(amount, self.stacks.get(player_id, 0)) # Cannot bet more than stack
        self.stacks[player_id] = self.stacks.get(player_id, 0) - actual_bet
        self.current_bets[player_id] = self.current_bets.get(player_id, 0) + actual_bet
        self.round_total_bets[player_id] = self.round_total_bets.get(player_id, 0) + actual_bet
        self.pot += actual_bet
        # print(f"Player {player_id} posts {actual_bet}. Stack: {self.stacks[player_id]}") # Debug

    def _get_legal_actions(self, player_id):
        """ Determines legal actions for the current player. """
        legal = []
        # Ignore if player isn't playing this tournament
        if self.seat_config.get(player_id) == 'empty':
             return []

        current_player_stack = self.stacks.get(player_id, 0)
        # If player is not active in round or has no stack, no actions are legal
        if player_id not in self.active_players_in_round or current_player_stack <= 0:
            return []

        max_bet_on_table = 0
        if self.current_bets:
             # Consider only bets from players still active in the round
             active_bets = {p: b for p, b in self.current_bets.items() if p in self.active_players_in_round}
             if active_bets:
                  max_bet_on_table = max(active_bets.values())

        player_current_bet = self.current_bets.get(player_id, 0)
        amount_to_call = max_bet_on_table - player_current_bet

        # --- Fold ---
        # Fold is always legal unless player is all-in and action is closed? Usually allowed.
        legal.append('fold')

        # --- Call / Check ---
        if amount_to_call <= 0:
            # No bet to call, Check is legal
            legal.append('check')
        else:
            # There is a bet to call
            if current_player_stack > amount_to_call:
                # Can afford the full call
                legal.append('call')
            elif current_player_stack > 0:
                 # Cannot afford full call, but has chips -> must go all-in to "call"
                 if 'all_in' not in legal: legal.append('all_in')


        # --- Bet / Raise / All-in ---
        # Can only bet/raise if stack > amount_to_call (need chips beyond the call amount)
        if current_player_stack > amount_to_call:
            # Determine minimum legal raise amount
            min_raise_delta = max(self.min_raise, self.big_blind) # Must raise at least BB or last raise size
            min_total_bet = max_bet_on_table + min_raise_delta # Minimum total bet required for a raise

            # Can the player afford the minimum raise?
            can_min_raise = current_player_stack >= (min_total_bet - player_current_bet)

            if can_min_raise:
                 # Add simplified bet sizes if min raise is possible
                 # TODO: Define bet_small/bet_big more precisely (e.g., fractions of pot)
                 legal.append('bet_small') # e.g., Pot/2 or fixed amount
                 legal.append('bet_big')   # e.g., Pot or fixed amount
                 if 'all_in' not in legal: legal.append('all_in')
            else:
                 # Cannot afford min raise, only option beyond call/fold is all-in
                 if 'all_in' not in legal: legal.append('all_in')


        # --- Cleanup and Finalization ---
        final_legal = set(legal)

        # Ensure check/call exclusivity
        if 'call' in final_legal and 'check' in final_legal:
            if amount_to_call > 0: final_legal.remove('check')
            else: final_legal.remove('call')

        # If 'all_in' is the only betting action possible besides check/call/fold, ensure it's there
        # (Handled above)

        # If player's stack equals exactly the call amount, 'call' and 'all_in' might both appear.
        # Remove 'call' if stack == amount_to_call > 0, only 'all_in' is the true action.
        if amount_to_call > 0 and current_player_stack == amount_to_call:
             if 'call' in final_legal: final_legal.remove('call')
             if 'all_in' not in final_legal: final_legal.add('all_in')


        return sorted(list(final_legal))


    def _execute_action(self, player_id, action_str):
        """ Executes the chosen action for the player """
        # Get legal actions again for safety, though should be checked by caller
        legal_actions = self._get_legal_actions(player_id)
        is_legal = action_str in legal_actions

        # Handle cases where player might not be active anymore or action is illegal
        if not is_legal or player_id not in self.active_players_in_round:
             # If player folded previously or action is illegal, treat as check if possible, else fold.
             current_legal = self._get_legal_actions(player_id) # Re-check current legal actions
             if 'check' in current_legal:
                  action_str = 'check'
                  print(f"Warning: Player {player_id} invalid action '{action_str}'. Checking instead.")
             else:
                  action_str = 'fold'
                  print(f"Warning: Player {player_id} invalid action '{action_str}'. Folding instead.")
                  if player_id in self.active_players_in_round: # Ensure removal if folding
                       self.active_players_in_round.remove(player_id)
                  self.players_acted_this_round.add(player_id) # Mark as acted (folded)
                  return # Stop processing action if folded


        player_stack = self.stacks.get(player_id, 0)
        max_bet_on_table = 0
        if self.current_bets:
             active_bets = {p: b for p, b in self.current_bets.items() if p in self.active_players_in_round}
             if active_bets: max_bet_on_table = max(active_bets.values())

        player_current_bet = self.current_bets.get(player_id, 0)
        amount_to_call = max_bet_on_table - player_current_bet

        # print(f"Player {player_id} (Stack: {player_stack}) takes action: {action_str}") # Debug

        bet_amount = 0 # Amount added *this action*
        is_raise = False # Flag if action was a bet/raise

        if action_str == 'fold':
            if player_id in self.active_players_in_round:
                 self.active_players_in_round.remove(player_id)
        elif action_str == 'check':
            # Legal only if amount_to_call is 0
            pass # No change in bets
        elif action_str == 'call':
            call_amount = min(amount_to_call, player_stack) # Can't call more than stack
            self._post_bet(player_id, call_amount)
            bet_amount = call_amount
        elif action_str == 'all_in':
            bet_amount = player_stack
            self._post_bet(player_id, bet_amount)
            total_player_bet = player_current_bet + bet_amount
            if total_player_bet > max_bet_on_table: # Check if the all-in constitutes a raise
                 # Check if it's a valid raise (at least min_raise OR putting player all-in)
                 raise_delta = total_player_bet - max_bet_on_table
                 # Allow raise if delta >= min_raise OR if the player is now all-in (stack == 0)
                 if raise_delta >= self.min_raise or self.stacks.get(player_id, 0) == 0:
                      is_raise = True
                      self.last_raiser = player_id
                      self.min_raise = max(self.min_raise, raise_delta) # Update min raise for next player
                 else: # All-in is less than min raise - doesn't reopen betting fully
                      is_raise = False # Treat as a call that doesn't reopen action
                      print(f"Debug: Player {player_id} all-in ({bet_amount}) is less than min raise delta ({self.min_raise}). Not reopening betting.")

        else: # Handle bets (bet_small, bet_big)
            is_raise = True
            if action_str == 'bet_small':
                # Simplified: Bet half pot (ensure min raise/bet size)
                base_bet_size = int(self.pot * 0.5)
            elif action_str == 'bet_big':
                # Simplified: Bet full pot (ensure min raise/bet size)
                base_bet_size = int(self.pot)
            else: # Should not happen if action validation is correct
                 print(f"Error: Unknown bet action '{action_str}'. Folding.")
                 if player_id in self.active_players_in_round: self.active_players_in_round.remove(player_id)
                 self.players_acted_this_round.add(player_id)
                 return

            # Calculate the required increase to make a valid bet/raise
            min_increase = 0
            if max_bet_on_table > player_current_bet: # Facing a bet (raising)
                min_raise_delta = max(self.min_raise, self.big_blind)
                required_total_bet = max_bet_on_table + min_raise_delta
                min_increase = required_total_bet - player_current_bet
            else: # Opening bet
                min_increase = self.big_blind # Must bet at least BB

            # Determine actual increase: max of desired size and min required, capped by stack
            desired_increase = base_bet_size
            actual_increase = max(min_increase, desired_increase)
            actual_increase = min(player_stack, actual_increase) # Cap by stack

            self._post_bet(player_id, actual_increase)
            bet_amount = actual_increase
            self.last_raiser = player_id
            # Update min_raise based on the actual increase relative to the previous max bet
            current_total_bet = player_current_bet + actual_increase
            self.min_raise = current_total_bet - max_bet_on_table # New minimum raise delta

        # Mark player as having acted in this betting round
        self.players_acted_this_round.add(player_id)

        # If a raise occurred, reset the 'acted' set for the new betting round dynamics
        # Only players who haven't acted *since the last raise* need to act again.
        if is_raise:
            # print(f"Debug: Raise by {player_id}. Resetting players_acted_this_round.") # Debug
            self.players_acted_this_round = {player_id} # Only the raiser has acted in this new "sub-round"


    def _check_betting_round_over(self):
        """ Checks if the current betting round should end. """
        active_in_round = list(self.active_players_in_round)
        if len(active_in_round) <= 1:
            # print("Debug: Betting round over - <= 1 player active.") # Debug
            return True # Betting ends if only one player left

        # Find players who can still act (have chips > 0)
        players_can_act = [p for p in active_in_round if self.stacks.get(p, 0) > 0]
        if not players_can_act:
             # print("Debug: Betting round over - all remaining players are all-in.") # Debug
             return True # Betting ends if all remaining players are all-in

        # Check if all players who *can* act have matched the highest bet AND have acted in this round
        max_bet = 0
        if self.current_bets:
             active_bets = {p: b for p, b in self.current_bets.items() if p in active_in_round}
             if active_bets: max_bet = max(active_bets.values())


        all_acted_and_matched = True
        for p_id in players_can_act:
            # Player must have acted since the last raise (or posted blind) AND bet matches max bet
            if p_id not in self.players_acted_this_round or self.current_bets.get(p_id, 0) < max_bet:
                all_acted_and_matched = False
                # print(f"Debug: Betting continues - Player {p_id} hasn't acted ({p_id in self.players_acted_this_round}) or matched bet ({self.current_bets.get(p_id, 0)} < {max_bet}).") # Debug
                break

        if all_acted_and_matched:
            # Special case: Preflop, action back to BB who hasn't raised and can check
            is_bb = False # Determine if current player is BB
            active_list_sorted = sorted(list(self.active_players_in_round))
            num_active_in_round = len(active_list_sorted)
            if num_active_in_round < 2: return True # Should not happen here, but safety

            button_idx_in_active = -1
            try: button_idx_in_active = active_list_sorted.index(self.button_pos)
            except ValueError: button_idx_in_active = 0 # Default if button not active

            bb_player = -1
            if num_active_in_round == 2:
                 bb_player = active_list_sorted[(button_idx_in_active + 1) % num_active_in_round]
            else: # 3+ players
                 bb_player = active_list_sorted[(button_idx_in_active + 2) % num_active_in_round]

            # Check if action is on BB, they were the last raiser (blind), and bet is BB amount
            if self.stage == 'preflop' and \
               self.current_player_id == bb_player and \
               self.last_raiser == bb_player and \
               max_bet == self.big_blind:
                   # print("Debug: Betting continues - Preflop action on BB to check/raise.") # Debug
                   return False # BB still needs to act

            # print("Debug: Betting round over - all active players acted and matched.") # Debug
            return True

        return False # Default: betting continues


    def _get_winners(self):
        """ Determines the winner(s) at showdown using the imported evaluator. """
        active_showdown = list(self.active_players_in_round) # Players involved in showdown
        if not active_showdown: return [], {} # No active players

        # If only one player left, they win by default
        if len(active_showdown) == 1:
            winner_id = active_showdown[0]
            # Hand isn't relevant if only one player left, but return structure consistent
            hand_info = {'hand': self.hands.get(winner_id, []), 'score': 0, 'desc': "Default Win"}
            return [winner_id], {winner_id: hand_info}

        best_score = -1
        winners = []
        showdown_hands = {} # Store hands for info dict

        # print(f"Showdown between: {active_showdown}") # Debug
        # print(f"Community Cards: {self.community_cards}") # Debug

        for player_id in active_showdown:
            hole = self.hands.get(player_id, [])
            if not hole:
                 # print(f"Player {player_id} has no hand for showdown?") # Debug
                 continue # Skip players with no hand

            # Use the imported improved hand evaluator
            try:
                 score, desc = evaluate_hand_improved(hole, self.community_cards)
                 showdown_hands[player_id] = {'hand': hole, 'score': score, 'desc': desc}
                 # print(f"Player {player_id} Showdown: Hand={hole} -> Score={score} ({desc})") # Debug

                 if score > best_score:
                     best_score = score
                     winners = [player_id]
                 elif score == best_score:
                     winners.append(player_id)
            except Exception as e:
                 print(f"Error evaluating hand for player {player_id} ({hole} + {self.community_cards}): {e}")
                 showdown_hands[player_id] = {'hand': hole, 'score': -1, 'desc': "Eval Error"}


        # print(f"Best Score: {best_score}, Winners: {winners}") # Debug
        return winners, showdown_hands


    def _distribute_pot(self, winners):
        """ Distributes the pot among winners. Handles simple cases, NO SIDE POTS. """
        if not winners: return 0 # No winners, pot stays? (Shouldn't happen)

        # --- Extremely Simplified Pot Distribution ---
        # Ignores side pots entirely. Splits the total pot among winners.
        # This is incorrect if players were all-in for different amounts.
        # A proper implementation requires tracking contributions and calculating side pots.
        num_winners = len(winners)
        total_pot_to_distribute = sum(self.round_total_bets.values()) # Use total bets this round
        win_amount_per_winner = total_pot_to_distribute / num_winners if num_winners > 0 else 0

        agent_reward_this_round = 0
        # print(f"Distributing Pot: {total_pot_to_distribute}, Winners: {winners}, Amount/Winner: {win_amount_per_winner}") # Debug

        for winner_id in winners:
            if winner_id in self.stacks: # Ensure player exists
                 self.stacks[winner_id] += win_amount_per_winner
                 # print(f"Player {winner_id} stack updated to {self.stacks[winner_id]}") # Debug

        # Calculate agent reward = net change in stack over the round
        if self.agent_id in self._get_playing_players(): # Only calculate if agent is playing
            agent_stack_before = self.initial_stacks_this_round.get(self.agent_id, 0)
            agent_stack_after = self.stacks.get(self.agent_id, 0)
            agent_reward_this_round = agent_stack_after - agent_stack_before
            # print(f"Agent {self.agent_id} Reward this round: {agent_reward_this_round} (Stack {agent_stack_before} -> {agent_stack_after})") # Debug

        self.pot = 0 # Pot is distributed (reset internal tracking pot)
        return agent_reward_this_round


    def step(self, action):
        """ Executes one agent action or processes opponent turns. """

        # *** FIX: Check for round_over at the beginning ***
        if self.round_over:
             self._start_new_round()
             # If starting the new round immediately ends the tournament (e.g., <2 players)
             if self.tournament_over:
                  obs = self._get_obs(self.agent_id)
                  # Return 0 reward as this step didn't involve agent action leading to end
                  return obs, 0.0, True, False, self._get_info(round_over=True, terminated=True)
             # If new round started, proceed to get the current player and potentially act

        if self.tournament_over:
            # Return terminal observation consistent with Gym API
            obs = self._get_obs(self.agent_id)
            # Reward should be 0 if already terminated
            return obs, 0.0, True, False, self._get_info(round_over=self.round_over, terminated=True)


        # --- Main Step Logic ---
        player_id = self.current_player_id
        if player_id is None or self.seat_config.get(player_id) == 'empty':
             # Should not happen if turn logic is correct, but handle defensively
             print(f"Warning: step() called with invalid/empty current_player_id ({player_id}). Attempting recovery.")
             self.current_player_id = self._get_next_active_player_id(self.button_pos) # Find next valid player
             player_id = self.current_player_id
             if player_id is None:
                 print("Error Recovery Failed: No active player found. Ending tournament.")
                 self.tournament_over = True
                 obs = self._get_obs(self.agent_id)
                 return obs, 0.0, True, False, self._get_info(error="current_player_id was None/Empty", terminated=True)


        is_agent_turn = (player_id == self.agent_id)
        # Use the action passed if it's the agent's turn, otherwise -1 signals opponent turn
        action_idx = action if is_agent_turn else -1
        last_action_details = {} # To store action taken in info

        # --- Action Execution Loop (Agent or Opponents) ---
        # This loop continues until control returns to the agent or the round ends
        while True: # Loop until break
            # Check loop start conditions
            if self.round_over or self.tournament_over: break

            player_id = self.current_player_id
            if player_id is None or self.seat_config.get(player_id) == 'empty':
                 # If turn lands on empty seat or becomes None, find next valid player
                 # print(f"Debug: Skipping turn for invalid/empty player {player_id}") # Debug
                 next_player = self._get_next_active_player_id(player_id) # Pass current player_id (even if None)
                 if next_player == self.current_player_id and next_player is not None: # Avoid infinite loop if stuck
                      print(f"Error: Stuck finding next player from {player_id}. Ending round.")
                      self.round_over = True
                      break
                 self.current_player_id = next_player
                 if self.current_player_id is None: # Still couldn't find one
                      # print("Debug: No next player found after skip. Checking betting round.") # Debug
                      if self._check_betting_round_over(): break # Exit loop
                      else: self.round_over = True; break # End round if stuck
                 continue # Restart loop with the new current_player_id

            # --- Get Action ---
            action_to_execute = None
            if player_id == self.agent_id:
                 # Use the action passed into the step function (if it was agent's turn initially)
                 if action_idx != -1:
                     action_to_execute = self._action_to_string.get(action_idx, 'fold')
                 else:
                      # Agent's turn came up during opponent loop, need to return control
                      break # Exit loop, step returns observation for agent
                 # print(f"--- Agent {player_id} turn. Action: {action_to_execute} ---") # Debug
            else:
                 # Get opponent action
                 opp_obs_dict = self._get_obs_dict(player_id)
                 opp_policy = self.opponent_policies.get(player_id)
                 if opp_policy:
                      action_to_execute = opp_policy(opp_obs_dict)
                 else: # Default for missing policy
                      action_to_execute = 'fold'
                      if self.seat_config.get(player_id) != 'empty':
                          print(f"Warning: No policy for opponent {player_id}. Folding.")
                 # print(f"--- Opponent {player_id} turn. Action: {action_to_execute} ---") # Debug

            # --- Execute Action ---
            self._execute_action(player_id, action_to_execute)
            last_action_details = {player_id: action_to_execute} # Store last action

            # --- Check for immediate round end due to folds ---
            if len(self.active_players_in_round) <= 1:
                 self.round_over = True
                 break # Exit action loop

            # --- Determine Next Player ---
            next_player_id = self._get_next_active_player_id(player_id)
            self.current_player_id = next_player_id

            # --- Check if Betting Round is Over ---
            if self._check_betting_round_over():
                # print("Debug: Betting round over detected in action loop.") # Debug
                break # Exit loop to handle stage progression/showdown

            # If the next player is the agent, exit the opponent loop
            if self.current_player_id == self.agent_id and player_id != self.agent_id:
                 break


        # --- Post-Action Processing ---
        agent_round_reward = 0.0
        info = {'last_action': last_action_details} # Start info dict with last action

        # --- Handle Round End or Stage Progression ---
        betting_round_ended_this_step = self._check_betting_round_over()
        round_ended_this_step = self.round_over # Use flag potentially set above

        # --- Check for All-in Runout ---
        # This check happens *after* the betting round is determined to be over
        needs_runout = False
        if betting_round_ended_this_step and not round_ended_this_step:
            active_in_round = list(self.active_players_in_round)
            # Need runout if >1 player left, and all players left with chips are all-in
            players_with_chips = [p for p in active_in_round if self.stacks.get(p, 0) > 0]
            # All-in check: Are all players who are *still in the round* either all-in OR have folded?
            # Simpler: Are there players left who *can* still bet? If not, and >1 player, run it out.
            if len(active_in_round) > 1 and not players_with_chips:
                needs_runout = True

        if needs_runout:
            print("--- All-in Runout Detected ---")
            # Loop through remaining stages and deal cards
            while self.stage != 'river':
                next_stage_map = {'preflop': 'flop', 'flop': 'turn', 'turn': 'river'}
                if self.stage in next_stage_map:
                    self.stage = next_stage_map[self.stage]
                    print(f"Dealing {self.stage}...")
                    self._deal_cards()
                else: break # Should not happen

            self.stage = 'showdown'
            self.round_over = True # Mark round as over
            round_ended_this_step = True # Treat as round end for info population

            # Get winners and distribute pot now
            winners, showdown_hands = self._get_winners()
            agent_round_reward = self._distribute_pot(winners)
            info.update({ 'round_over': True, 'round_reward': agent_round_reward, 'winners': winners, 'showdown_hands': showdown_hands, 'final_pot': sum(self.round_total_bets.values()) })

            # Check tournament end again after runout
            active_players_final = self._get_active_players(include_all_in=False)
            if self.agent_id in self._get_playing_players() and self.stacks.get(self.agent_id, 0) <= 0: self.tournament_over = True
            elif len(active_players_final) <= 1: self.tournament_over = True

        elif round_ended_this_step: # Round ended normally (e.g. folds, or showdown after river betting)
            # print(f"--- Round Ended Normally ---") # Less verbose
            winners, showdown_hands = self._get_winners()
            agent_round_reward = self._distribute_pot(winners) # Distributes pot and calculates agent reward

            # Populate info dictionary for round end
            info.update({
                'round_over': True,
                'round_reward': agent_round_reward,
                'winners': winners,
                'showdown_hands': showdown_hands,
                'final_pot': sum(self.round_total_bets.values())
            })

            # Check for tournament termination AFTER distributing pot
            active_players_final = self._get_active_players(include_all_in=False)
            if self.agent_id in self._get_playing_players() and self.stacks.get(self.agent_id, 0) <= 0:
                print(f"Tournament Over: Agent {self.agent_id} busted.")
                self.tournament_over = True
            elif len(active_players_final) <= 1:
                winner_declared = False
                if active_players_final:
                     final_winner_id = active_players_final[0]
                     if final_winner_id == self.agent_id: print(f"Tournament Over: Agent {self.agent_id+1} wins!")
                     else: print(f"Tournament Over: Player {final_winner_id+1} wins!")
                     winner_declared = True
                if not winner_declared: print(f"Tournament Over: No single winner.")
                self.tournament_over = True

        elif betting_round_ended_this_step: # Betting round ended, but hand continues (and not all-in runout)
             # print(f"--- Betting Round Over ---") # Less verbose
             # Move to next stage (flop, turn, river, showdown)
             self.current_bets = {i: 0 for i in self._get_playing_players()} # Reset bets
             self.last_raiser = None
             self.min_raise = self.big_blind
             self.players_acted_this_round = set() # Reset actors for new street

             # Determine next player to act (usually SB or first active player after button)
             next_actor = self._get_next_active_player_id(self.button_pos)
             self.current_player_id = next_actor

             next_stage_map = {'preflop': 'flop', 'flop': 'turn', 'turn': 'river', 'river': 'showdown'}
             if self.stage in next_stage_map:
                 new_stage = next_stage_map[self.stage]
                 if new_stage == 'showdown':
                      self.stage = 'showdown'
                      self.round_over = True # Mark round as over
                      round_ended_this_step = True # Treat as round end for info

                      # Get winners and distribute pot
                      winners, showdown_hands = self._get_winners()
                      agent_round_reward = self._distribute_pot(winners)
                      info.update({ 'round_over': True, 'round_reward': agent_round_reward, 'winners': winners, 'showdown_hands': showdown_hands, 'final_pot': sum(self.round_total_bets.values()) })
                      # Check tournament end again
                      active_players_final = self._get_active_players(include_all_in=False)
                      if self.agent_id in self._get_playing_players() and self.stacks.get(self.agent_id, 0) <= 0: self.tournament_over = True
                      elif len(active_players_final) <= 1: self.tournament_over = True

                 else: # Advance to flop, turn, or river
                      self.stage = new_stage
                      self._deal_cards()
                      # print(f"--- Advancing to Stage: {self.stage} ---") # Less verbose
             else:
                 print(f"Error: Cannot advance from stage {self.stage}")
                 self.round_over = True # End round if stage is invalid
                 round_ended_this_step = True


        # --- Prepare return values ---
        # *** FIX: Explicitly set current_player_id to None if round ended ***
        if round_ended_this_step:
             self.current_player_id = None

        terminated = self.tournament_over
        truncated = False # Use for other limits if needed

        obs = self._get_obs(self.agent_id)
        final_info = self._get_info() # Get base info like stacks, pot, etc.
        final_info.update(info) # Add round_over, round_reward etc. if round ended

        # The reward returned by step() itself is the per-round reward if round ended, else 0
        step_reward = agent_round_reward if final_info.get('round_over', False) else 0.0

        if self.render_mode == "human":
            self._render_frame()

        # Ensure round_over flag in final_info is accurate
        final_info['round_over'] = round_ended_this_step
        final_info['terminated'] = terminated # Pass terminated status

        return obs, step_reward, terminated, truncated, final_info


    def _get_obs_dict(self, player_id):
         """ Returns the observation as a dictionary (useful for policies). """
         # Return empty dict if player is not playing
         if self.seat_config.get(player_id) == 'empty':
             return {}

         obs_dict = {
             "hand": self.hands.get(player_id, []),
             "community_cards": self.community_cards[:],
             "pot": self.pot,
             "stacks": self.stacks.copy(),
             "current_bets": self.current_bets.copy(),
             "stage": self.stage,
             "legal_actions": self._get_legal_actions(player_id),
             "current_player_id": self.current_player_id,
             "player_id": player_id,
             "button_pos": self.button_pos,
             "small_blind": self.small_blind,
             "big_blind": self.big_blind,
             "num_active_players": len(self._get_active_players(include_all_in=False)), # Count players with chips
             "active_in_hand": list(self.active_players_in_round), # IDs still in the current hand
         }
         return obs_dict

    def _get_obs(self, player_id):
        """
        Gets the observation for the specified player and encodes it using utils.encode_obs.
        Returns zero vector if player_id is invalid or not playing.
        """
        # Return zero vector if agent is not a playing player
        if self.seat_config.get(self.agent_id) == 'empty':
             # print(f"Warning: Agent ID {self.agent_id} is set to empty. Returning zero observation.") # Reduce noise
             return np.zeros(self.observation_space.shape, dtype=np.float32)

        obs_dict = self._get_obs_dict(player_id)
        encoded_state = np.zeros(self.observation_space.shape, dtype=np.float32) # Default
        if not obs_dict: # Handle case where obs_dict is empty (e.g., invalid player)
             return encoded_state

        try:
            # Now relies on the updated utils.encode_obs
            from .utils import encode_obs
            encoded_state = encode_obs(obs_dict) # Pass the dictionary

        except ImportError:
            print("ERROR: Could not import encode_obs from .utils. Update utils.py!")
        except Exception as e:
            print(f"ERROR during observation encoding: {e}. Check utils.encode_obs!")
            print(f"Observation dict causing error: {obs_dict}") # Print dict for debugging

        # Ensure the encoded state fits the defined observation space (safety check)
        if encoded_state.shape != self.observation_space.shape:
             print(f"Warning: Encoded state shape {encoded_state.shape} != observation space {self.observation_space.shape}. Check encoding!")
             # Attempt to pad or truncate (crude fix)
             target_len = self.observation_space.shape[0]
             current_len = encoded_state.shape[0]
             if current_len < target_len:
                  padding = np.zeros(target_len - current_len, dtype=np.float32)
                  encoded_state = np.concatenate([encoded_state, padding])
             elif current_len > target_len:
                  encoded_state = encoded_state[:target_len]

        return encoded_state.astype(np.float32)


    def _get_info(self, **extra_info):
        """ Returns auxiliary information dictionary. """
        info = {
            "stage": self.stage,
            "pot": self.pot,
            "stacks": self.stacks.copy(),
            "community_cards": self.community_cards[:],
            "current_bets": self.current_bets.copy(),
            "button_pos": self.button_pos,
            "active_players": list(self.active_players_in_round),
            # Ensure round_over defaults to False if not explicitly set
            "round_over": self.round_over, # Use current state flag
            "current_player_id": self.current_player_id, # Include current player ID
            "terminated": self.tournament_over # Include termination status
        }
        info.update(extra_info) # Add specific info like round_over=True, round_reward
        return info

    def render(self):
        """ Renders the current environment state. """
        if self.render_mode == "human":
            self._render_frame()
        # Add rgb_array rendering if needed

    def _render_frame(self):
        """ Renders one frame for human viewing (simple text). """
        print("\n" + "="*30)
        print(f"Stage: {self.stage} | Pot: {self.pot:.2f} | Button: Seat {self.button_pos + 1}")
        print(f"Community Cards: {' '.join(self.community_cards)}")
        print("-"*30)
        playing_players = self._get_playing_players()
        for i in range(self.num_players):
            if i not in playing_players:
                 print(f"Seat {i+1}: --- EMPTY ---")
                 continue

            # Try to import render_card for better display if available
            try: from Front_End.card_utils import render_hand # Use render_hand for simpler text output
            except ImportError: render_hand = lambda x: " ".join(x) # Fallback

            hand_str = render_hand(self.hands.get(i, []))
            if not hand_str and i in self.active_players_in_round: hand_str = "? ?" # Show placeholders only if active
            elif not hand_str: hand_str = "" # Don't show placeholders if folded/out

            stack_str = f"{self.stacks.get(i, 0):.0f}" # No decimals for stack
            bet_str = f"{self.current_bets.get(i, 0):.0f}" # No decimals for bet
            status = ""
            if i not in self.active_players_in_round and i in self.hands and self.stacks.get(i,0) > 0: status = " (Folded)"
            elif self.stacks.get(i,0) <= 0 and i in self.active_players_in_round: status = " (All-In)"
            elif self.stacks.get(i,0) <= 0: status = " (Out)" # Use stack check for Out status

            is_button = " (BTN)" if i == self.button_pos else ""
            is_turn = " <= TURN" if i == self.current_player_id else ""
            is_agent = " (AGENT)" if i == self.agent_id else ""
            print(f"Seat {i+1}{is_agent}{is_button}: Stack={stack_str}, Bet={bet_str}, Hand=[{hand_str}]{status}{is_turn}")
        print("="*30)
        if self.current_player_id == self.agent_id:
             print(f"Agent ({self.agent_id+1}) to act. Legal: {self._get_legal_actions(self.agent_id)}")


    def close(self):
        """ Performs any necessary cleanup. """
        print("Closing Poker Environment.")
        # Add cleanup logic here if using external resources


# Optional: Define TrainFullPokerEnv inheriting if needed, or just use Base directly
class TrainFullPokerEnv(BaseFullPokerEnv):
     """
     Environment specifically for training, potentially with slight modifications
     or additions compared to the base evaluation environment.
     Inherits tournament logic from BaseFullPokerEnv.
     """
     def __init__(self, **kwargs):
          super().__init__(**kwargs)

     # Add any training-specific methods or overrides here if needed

     # --- Helpers for Agent/UI ---
     def get_legal_actions_for_agent(self):
         """ Helper specifically for the agent """
         # Ensure agent is actually playing before getting actions
         if self.seat_config.get(self.agent_id) == 'empty':
             return []
         return self._get_legal_actions(self.agent_id)

     def get_player_hand(self, player_id):
          """ Helper to get a specific player's hand """
          # Return empty if seat is empty
          if self.seat_config.get(player_id) == 'empty':
              return []
          return self.hands.get(player_id, [])

     def get_stacks(self, include_all=True):
          # Optionally filter out empty seats if include_all is False
          if include_all:
              return self.stacks.copy()
          else:
              playing_players = self._get_playing_players()
              return {p: s for p, s in self.stacks.items() if p in playing_players}


     def get_pot(self):
          return self.pot

     def get_community_cards(self):
          return self.community_cards[:]

     def get_current_bets(self):
          return self.current_bets.copy()


# Example of how to run tests if the file is executed directly
if __name__ == "__main__":
    print("Running envs.py directly for testing...")

    # Example: Test hand evaluation import
    try:
        test_score, test_desc = evaluate_hand_improved(["AS", "KS", "QS", "JS", "TS"], [])
        print(f"Test Eval (Royal Flush): {test_desc} (Score: {test_score})")
        test_score_2, test_desc_2 = evaluate_hand_improved(["7H", "7D", "7C", "2S", "KH"], [])
        print(f"Test Eval (Three 7s): {test_desc_2} (Score: {test_score_2})")
    except Exception as e:
        print(f"Error during hand evaluation test: {e}")

    # Example: Instantiate and reset the environment
    try:
        print("\nInstantiating environment...")
        test_env = BaseFullPokerEnv()
        print("Resetting environment...")
        obs, info = test_env.reset()
        print("Reset successful. Initial Info:")
        # Pretty print the info dictionary
        import json
        print(json.dumps(info, indent=2, default=str))
        print("\nRendering initial state:")
        test_env.render()
        test_env.close()
    except Exception as e:
        print(f"\nError during environment instantiation/reset test: {e}")
        import traceback
        traceback.print_exc()

