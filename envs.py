# filename: envs.py
"""
Poker Environment for Reinforcement Learning (Tournament Structure)

Implements a Gymnasium environment where:
- An episode represents a full multi-round tournament.
- Rewards are potentially given per round via the info dict.
- Termination occurs when the agent busts or wins the tournament.

MODIFIED: Removed placeholder warning and temporary encoding logic
          from _get_obs method. Now relies fully on utils.encode_obs.
"""

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

# --- Placeholder Hand Evaluation ---
# Replace this with a proper poker hand evaluator
def evaluate_hand(hole_cards, community_cards):
    """ Placeholder function to evaluate hand strength. """
    # Combine hole and community cards
    all_cards = hole_cards + community_cards
    if not all_cards:
        return 0, "No Cards" # No score if no cards

    # Extremely simplified evaluation: Count pairs/ranks
    ranks = [card[:-1] for card in all_cards] # Get ranks '2', 'T', 'A' etc.
    rank_counts = Counter(ranks)
    score = 0
    description = "High Card"

    pairs = 0
    trips = 0
    quads = 0
    max_count = 0
    for rank, count in rank_counts.items():
        if count == 2: pairs += 1; score = max(score, 100); description="One Pair"
        if count == 3: trips += 1; score = max(score, 300); description="Three of a Kind"
        if count == 4: quads += 1; score = max(score, 700); description="Four of a Kind"
        max_count = max(max_count, count)

    if trips == 1 and pairs >= 1: score = max(score, 600); description="Full House"
    elif pairs >= 2: score = max(score, 200); description="Two Pair"

    # Add simple rank value bonus (Ace=14, King=13...)
    rank_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
    try:
        # Add value of highest card in hand only for simplicity
        hole_ranks = [card[:-1] for card in hole_cards]
        if hole_ranks:
             score += max(rank_values.get(r, 0) for r in hole_ranks)
    except Exception:
        pass # Ignore errors in placeholder

    # Return score and a simple description
    return score, description
# --- End Placeholder ---


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

    def __init__(self, num_players=6, agent_id=0, starting_stack=10000, small_blind=50, big_blind=100, render_mode=None):
        super().__init__()

        self.num_players = num_players
        self.agent_id = agent_id # The ID of the RL agent player
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind

        # --- State Attributes ---
        self.deck = []
        self.stacks = {} # Player ID -> Stack size
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
            from utils import NEW_STATE_DIM
            self.observation_space_dim = NEW_STATE_DIM
        except ImportError:
            print("Warning: Could not import NEW_STATE_DIM from utils. Using default 333.")
            self.observation_space_dim = 333 # Fallback, ensure utils.py is updated

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.observation_space_dim,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        # Add rendering elements if needed (e.g., pygame)


    def set_opponent_policy(self, player_id, policy_func):
        """ Sets the policy function for a specific opponent """
        if player_id != self.agent_id:
            self.opponent_policies[player_id] = policy_func

    def _get_active_players(self, include_all_in=True):
        """ Returns a list of player IDs who have chips > 0 """
        active = []
        for i in range(self.num_players):
            if self.stacks.get(i, 0) > 0:
                 active.append(i)
            elif include_all_in and i in self.active_players_in_round and self.stacks.get(i, 0) <= 0:
                 # Include players who are all-in but still in the hand
                 active.append(i)
        return active

    def _get_next_active_player_id(self, start_id):
        """ Finds the next player ID in rotation who is still active in the round. """
        if not self.active_players_in_round: return None # No active players left

        current_id = (start_id + 1) % self.num_players
        checked_count = 0
        while checked_count < self.num_players: # Prevent infinite loop
            if current_id in self.active_players_in_round:
                 # Check if player can still act (not all-in unless it's closing action)
                 # Simplified: Assume active players can always act if round not closed
                 # More robust check needed for closing action with all-ins.
                 if self.stacks.get(current_id, 0) > 0:
                      return current_id
            current_id = (current_id + 1) % self.num_players
            checked_count += 1

        # If only one player left with chips who can act, return their ID
        players_can_act = [p for p in self.active_players_in_round if self.stacks.get(p, 0) > 0]
        if len(players_can_act) == 1:
             return players_can_act[0]
        # If loop completes without finding next player (e.g., all remaining are all-in),
        # it might mean action is closed or something went wrong. Return None or handle appropriately.
        return None # Indicates action might be closed or no one can act

    def _deal_cards(self):
        """ Deals hole cards and community cards based on stage """
        if self.stage == 'preflop':
            self.hands = {}
            active_players = self._get_active_players(include_all_in=False) # Only deal to players with chips
            for _ in range(2): # Deal 2 hole cards
                for player_id in active_players:
                    if player_id not in self.hands: self.hands[player_id] = []
                    if self.deck: self.hands[player_id].append(self.deck.pop())
            # print(f"Dealt Hands: {self.hands}") # Debug
        elif self.stage == 'flop':
            if self.deck: self.deck.pop() # Burn card
            self.community_cards = [self.deck.pop() for _ in range(3) if self.deck]
            # print(f"Dealt Flop: {self.community_cards}") # Debug
        elif self.stage == 'turn' or self.stage == 'river':
            if self.deck: self.deck.pop() # Burn card
            if self.deck: self.community_cards.append(self.deck.pop())
            # print(f"Dealt Turn/River: {self.community_cards}") # Debug

    def _start_new_round(self):
        """ Initializes a new hand/round within the tournament. """
        self.round_over = False
        active_tournament_players = self._get_active_players(include_all_in=False)
        if len(active_tournament_players) <= 1:
            print("Tournament ended between rounds.")
            self.tournament_over = True
            return # Don't start round if tournament is over

        # print(f"\n--- Starting Round ---") # Less verbose
        self.deck = DECK[:]
        random.shuffle(self.deck)
        self.community_cards = []
        self.pot = 0
        self.current_bets = {i: 0 for i in range(self.num_players)}
        self.round_total_bets = {i: 0 for i in range(self.num_players)}
        self.active_players_in_round = set(active_tournament_players) # Players starting the round
        self.last_raiser = None
        self.stage = 'preflop'
        self.initial_stacks_this_round = self.stacks.copy() # Store stacks at round start

        # Determine positions (handle small number of players)
        num_active = len(active_tournament_players)
        # Rotate button only if more than 2 players? Standard rules vary. Assume rotate always.
        self.button_pos = (self.button_pos + 1) % self.num_players
        while self.button_pos not in active_tournament_players: # Find next active player for button
             self.button_pos = (self.button_pos + 1) % self.num_players

        # Find SB and BB among active players
        active_list_sorted = sorted(list(active_tournament_players))
        button_idx_in_active = -1
        for i, p_id in enumerate(active_list_sorted):
            if p_id == self.button_pos:
                button_idx_in_active = i
                break

        if num_active == 2: # Heads up posting logic
             sb_player = active_list_sorted[button_idx_in_active]
             bb_player = active_list_sorted[(button_idx_in_active + 1) % num_active]
             self.current_player_id = sb_player # Button/SB acts first preflop HU
        else: # 3+ players
             sb_player = active_list_sorted[(button_idx_in_active + 1) % num_active]
             bb_player = active_list_sorted[(button_idx_in_active + 2) % num_active]
             next_player_idx = (button_idx_in_active + 3) % num_active
             self.current_player_id = active_list_sorted[next_player_idx] # UTG

        # Post blinds
        sb_amount = min(self.small_blind, self.stacks.get(sb_player, 0))
        bb_amount = min(self.big_blind, self.stacks.get(bb_player, 0))
        self._post_bet(sb_player, sb_amount)
        self._post_bet(bb_player, bb_amount)
        self.last_raiser = bb_player # BB is the initial "raiser"
        self.min_raise = self.big_blind # Initial min raise amount

        # Deal cards
        self._deal_cards()
        # print(f"Button: {self.button_pos}, SB: {sb_player}, BB: {bb_player}, Start Player: {self.current_player_id}") # Debug


    def reset(self, seed=None, options=None):
        """ Resets the environment to start a new tournament. """
        super().reset(seed=seed)
        print("\n===== RESETTING TOURNAMENT =====")

        # Initialize tournament state
        self.stacks = {i: self.starting_stack for i in range(self.num_players)}
        self.button_pos = random.randint(0, self.num_players - 1) # Random initial button
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
             # Should reset return done=True? Gym API typically doesn't.
        else:
             observation = self._get_obs(self.agent_id)
             info = self._get_info()


        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _post_bet(self, player_id, amount):
        """ Handles posting a bet/blind/call """
        if player_id not in self.stacks: return # Safety check
        actual_bet = min(amount, self.stacks.get(player_id, 0)) # Cannot bet more than stack
        self.stacks[player_id] = self.stacks.get(player_id, 0) - actual_bet
        self.current_bets[player_id] = self.current_bets.get(player_id, 0) + actual_bet
        self.round_total_bets[player_id] = self.round_total_bets.get(player_id, 0) + actual_bet
        self.pot += actual_bet
        # print(f"Player {player_id} posts {actual_bet}. Stack: {self.stacks[player_id]}") # Debug

    def _get_legal_actions(self, player_id):
        """ Determines legal actions for the current player. """
        legal = []
        current_player_stack = self.stacks.get(player_id, 0)
        # If player is not active or has no stack, no actions are legal
        if player_id not in self.active_players_in_round or current_player_stack <= 0:
            return []

        max_bet_on_table = max(self.current_bets.values()) if self.current_bets else 0
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
             if 'check' in self._get_legal_actions(player_id): # Re-check current legal actions
                  action_str = 'check'
                  print(f"Warning: Player {player_id} invalid action '{action_str}'. Checking instead.")
             else:
                  action_str = 'fold'
                  print(f"Warning: Player {player_id} invalid action '{action_str}'. Folding instead.")
                  if player_id in self.active_players_in_round: # Ensure removal if folding
                       self.active_players_in_round.remove(player_id)
                  return # Stop processing action if folded


        player_stack = self.stacks.get(player_id, 0)
        max_bet_on_table = max(self.current_bets.values()) if self.current_bets else 0
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
                 is_raise = True
                 self.last_raiser = player_id
                 raise_delta = total_player_bet - max_bet_on_table
                 self.min_raise = max(self.min_raise, raise_delta)
        else: # Handle bets (bet_small, bet_big)
            is_raise = True
            if action_str == 'bet_small':
                # Simplified: Bet half pot (ensure min raise/bet size)
                bet_size = max(self.big_blind, int(self.pot * 0.5))
            elif action_str == 'bet_big':
                # Simplified: Bet full pot (ensure min raise/bet size)
                bet_size = max(self.big_blind, int(self.pot))
            else: # Should not happen if action validation is correct
                 print(f"Error: Unknown bet action '{action_str}'. Folding.")
                 if player_id in self.active_players_in_round: self.active_players_in_round.remove(player_id)
                 return

            # Calculate total bet amount for the player this round
            required_increase = 0
            if max_bet_on_table > player_current_bet: # Facing a bet
                 min_raise_delta = max(self.min_raise, self.big_blind)
                 required_total_bet = max_bet_on_table + min_raise_delta
                 required_increase = required_total_bet - player_current_bet
            else: # Opening bet
                 required_increase = max(self.big_blind, bet_size) # Bet at least BB or chosen size

            # Actual increase is the calculated size, but capped by stack and min requirements
            actual_increase = min(player_stack, required_increase)

            self._post_bet(player_id, actual_increase)
            bet_amount = actual_increase
            self.last_raiser = player_id
            # Update min_raise based on the actual increase relative to the previous max bet
            current_total_bet = player_current_bet + actual_increase
            self.min_raise = current_total_bet - max_bet_on_table # New minimum raise delta


    def _check_betting_round_over(self):
        """ Checks if the current betting round should end. More robust version. """
        active_in_round = list(self.active_players_in_round)
        if len(active_in_round) <= 1:
            return True # Betting ends if only one player left

        # Find players who can still act (have chips and haven't folded)
        players_can_act = [p for p in active_in_round if self.stacks.get(p, 0) > 0]
        if not players_can_act:
             return True # Betting ends if all remaining players are all-in

        # Check if all players who *can* act have matched the highest bet
        max_bet = max(self.current_bets.get(p, 0) for p in active_in_round)
        all_matched = True
        for p_id in players_can_act:
            if self.current_bets.get(p_id, 0) < max_bet:
                all_matched = False
                break

        # Check if the action has closed (returned to the last aggressor who didn't re-raise)
        # Requires tracking who acted last and if the last action was aggressive.
        # Simplified: If all bets are matched, and the current player to act cannot act (e.g., _get_next_active_player_id returns None or loops), end the round.
        # This assumes the main step loop correctly sets current_player_id.
        next_player = self._get_next_active_player_id(self.current_player_id)

        # Condition: All players who can act have the same bet amount, AND
        # the action is on a player who was the last aggressor OR action has gone around once without a raise.
        # This requires more state tracking (e.g., self.player_last_action).
        # Let's use a simpler proxy: if all bets are matched AND the next player calculation fails or points back to the current player inappropriately.
        if all_matched and (next_player is None or next_player == self.current_player_id):
             # This implies action is closed or stalled
             return True

        # A common case: Big blind checks preflop when facing no raise.
        if self.stage == 'preflop' and self.current_player_id == self._get_next_active_player_id(self.button_pos) and max_bet == self.big_blind and self.last_raiser == self._get_next_active_player_id(self._get_next_active_player_id(self.button_pos)):
             # If action is on SB (or equivalent in HU) and they called/folded, and BB checks.
             # Need better position tracking. Assume for now this is handled by the main loop correctly identifying the last action.
             pass # Let the main loop handle turn progression


        return False # Default: betting continues


    def _get_winners(self):
        """ Determines the winner(s) at showdown. """
        if not self.active_players_in_round: return [], {} # No active players

        if len(self.active_players_in_round) == 1:
            winner_id = list(self.active_players_in_round)[0]
            # Hand isn't relevant if only one player left, but return structure consistent
            hand_info = {'hand': self.hands.get(winner_id, []), 'score': 0, 'desc': "Default Win"}
            return [winner_id], {winner_id: hand_info}

        best_score = -1
        winners = []
        showdown_hands = {} # Store hands for info dict

        # print(f"Showdown between: {list(self.active_players_in_round)}") # Debug
        # print(f"Community Cards: {self.community_cards}") # Debug

        for player_id in self.active_players_in_round:
            hole = self.hands.get(player_id, [])
            if not hole:
                 # print(f"Player {player_id} has no hand for showdown?") # Debug
                 continue # Skip players with no hand

            # Use the (placeholder) hand evaluator
            score, desc = evaluate_hand(hole, self.community_cards)
            showdown_hands[player_id] = {'hand': hole, 'score': score, 'desc': desc}
            # print(f"Player {player_id} Showdown: Hand={hole} -> Score={score} ({desc})") # Debug

            if score > best_score:
                best_score = score
                winners = [player_id]
            elif score == best_score:
                winners.append(player_id)

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
        win_amount_per_winner = self.pot / num_winners if num_winners > 0 else 0

        agent_reward_this_round = 0
        # print(f"Distributing Pot: {self.pot}, Winners: {winners}, Amount/Winner: {win_amount_per_winner}") # Debug

        for winner_id in winners:
            if winner_id in self.stacks: # Ensure player exists
                 self.stacks[winner_id] += win_amount_per_winner
                 # print(f"Player {winner_id} stack updated to {self.stacks[winner_id]}") # Debug

        # Calculate agent reward = net change in stack over the round
        agent_stack_before = self.initial_stacks_this_round.get(self.agent_id, 0)
        agent_stack_after = self.stacks.get(self.agent_id, 0)
        agent_reward_this_round = agent_stack_after - agent_stack_before

        # print(f"Agent {self.agent_id} Reward this round: {agent_reward_this_round} (Stack {agent_stack_before} -> {agent_stack_after})") # Debug
        self.pot = 0 # Pot is distributed
        return agent_reward_this_round


    def step(self, action):
        """ Executes one agent action or processes opponent turns. """
        if self.tournament_over:
            # Return terminal observation consistent with Gym API
            obs = self._get_obs(self.agent_id)
            # Reward should be 0 if already terminated
            return obs, 0.0, True, False, self._get_info(round_over=self.round_over)

        # If the previous round ended, start a new one before processing the action
        if self.round_over:
             self._start_new_round()
             # If starting the new round immediately ends the tournament (e.g., <2 players)
             if self.tournament_over:
                  obs = self._get_obs(self.agent_id)
                  return obs, 0.0, True, False, self._get_info(round_over=True)

        # --- Main Step Logic ---
        player_id = self.current_player_id
        if player_id is None:
             # Should not happen if round/tournament logic is correct
             print("Error: current_player_id is None during step. Ending tournament.")
             self.tournament_over = True
             obs = self._get_obs(self.agent_id)
             return obs, 0.0, True, False, self._get_info(error="current_player_id was None")


        is_agent_turn = (player_id == self.agent_id)
        performed_action_player_id = player_id # Track who acted

        # --- Action Execution Loop (Agent or Opponents) ---
        # This loop continues until control returns to the agent or the round ends
        current_turn_processed = False
        while not current_turn_processed:
            if self.round_over or self.tournament_over: break # Exit if state changed mid-loop

            player_id = self.current_player_id
            if player_id is None: # Check again if next player couldn't be found
                 print("Error: current_player_id became None in action loop. Ending round.")
                 # This might indicate all remaining players are all-in
                 # We should force progression to the next stage or showdown
                 # TODO: Add logic to handle all-in scenarios progressing the board
                 self.round_over = True # Assume round ends if no one can act
                 break

            # --- Get Action ---
            action_to_execute = None
            if player_id == self.agent_id:
                 # Use the action passed into the step function
                 action_to_execute = self._action_to_string.get(action, 'fold')
                 current_turn_processed = True # Agent's turn is done for this step call
                 # print(f"--- Agent {player_id} turn. Action: {action_to_execute} ---") # Debug
            else:
                 # Get opponent action
                 opp_obs_dict = self._get_obs_dict(player_id)
                 opp_policy = self.opponent_policies.get(player_id)
                 if opp_policy:
                      action_to_execute = opp_policy(opp_obs_dict)
                 else:
                      print(f"Warning: No policy for opponent {player_id}. Folding.")
                      action_to_execute = 'fold'
                 # print(f"--- Opponent {player_id} turn. Action: {action_to_execute} ---") # Debug

            # --- Execute Action ---
            self._execute_action(player_id, action_to_execute)
            performed_action_player_id = player_id # Track who acted last

            # --- Check for immediate round end ---
            if len(self.active_players_in_round) <= 1:
                 self.round_over = True
                 break # Exit action loop

            # --- Determine Next Player ---
            next_player_id = self._get_next_active_player_id(player_id)
            self.current_player_id = next_player_id

            # --- Check if Betting Round is Over ---
            # Simple check: Did the action return to the last aggressor who didn't re-raise?
            # Or did the action return to the BB who checked? Needs better state tracking.
            # Let's use a placeholder check based on whether the next player can act.
            if self.current_player_id is None: # No one else can act
                 self.round_over = True # Assume this means showdown or pot awarded
                 break
            # TODO: Add more robust betting round end logic here.

            # If the next player is the agent, exit the opponent loop
            if self.current_player_id == self.agent_id and not is_agent_turn:
                 break


        # --- Post-Action Processing ---
        agent_round_reward = 0.0
        info = {}
        round_ended_this_step = self.round_over # Use the flag set during action loop

        # --- Handle Round End ---
        if round_ended_this_step:
            # print(f"--- Round Ended ---") # Less verbose
            winners, showdown_hands = self._get_winners()
            agent_round_reward = self._distribute_pot(winners) # Distributes pot and calculates agent reward

            # Populate info dictionary for round end
            info = {
                'round_over': True,
                'round_reward': agent_round_reward,
                'winners': winners,
                'showdown_hands': showdown_hands,
                'final_pot': sum(self.round_total_bets.values())
            }

            # Check for tournament termination AFTER distributing pot
            active_players_final = self._get_active_players(include_all_in=False)
            if self.stacks.get(self.agent_id, 0) <= 0:
                print(f"Tournament Over: Agent {self.agent_id} busted.")
                self.tournament_over = True
            elif len(active_players_final) <= 1:
                if self.agent_id in active_players_final: print(f"Tournament Over: Agent {self.agent_id} wins!")
                else: print(f"Tournament Over: Agent {self.agent_id} eliminated earlier.")
                self.tournament_over = True

            # If tournament continues, the flag self.round_over=True will trigger
            # _start_new_round() on the *next* step call.

        # --- Handle Betting Round End (If Round Didn't End) ---
        elif self._check_betting_round_over(): # Check if betting round ended
             # print(f"--- Betting Round Over ---") # Less verbose
             # Move to next stage (flop, turn, river, showdown)
             self.current_bets = {i: 0 for i in range(self.num_players)} # Reset bets
             self.last_raiser = None
             self.min_raise = self.big_blind
             # Determine next player to act (usually SB or first active player after button)
             next_actor = self._get_next_active_player_id(self.button_pos)
             self.current_player_id = next_actor

             next_stage_map = {'preflop': 'flop', 'flop': 'turn', 'turn': 'river', 'river': 'showdown'}
             if self.stage in next_stage_map:
                 new_stage = next_stage_map[self.stage]
                 if new_stage == 'showdown':
                      self.stage = 'showdown'
                      self.round_over = True # Trigger round end logic on next cycle if needed
                      # Re-run end-of-round logic here?
                      winners, showdown_hands = self._get_winners()
                      agent_round_reward = self._distribute_pot(winners)
                      info = { 'round_over': True, 'round_reward': agent_round_reward, 'winners': winners, 'showdown_hands': showdown_hands, 'final_pot': sum(self.round_total_bets.values()) }
                      # Check tournament end again
                      active_players_final = self._get_active_players(include_all_in=False)
                      if self.stacks.get(self.agent_id, 0) <= 0: self.tournament_over = True
                      elif len(active_players_final) <= 1: self.tournament_over = True

                 else:
                      self.stage = new_stage
                      self._deal_cards()
                      # print(f"--- Advancing to Stage: {self.stage} ---") # Less verbose
             else:
                 # Should not happen if stage logic is correct
                 print(f"Error: Cannot advance from stage {self.stage}")
                 self.round_over = True # End round if stage is invalid


        # --- Prepare return values ---
        terminated = self.tournament_over
        truncated = False # Use for other limits if needed

        obs = self._get_obs(self.agent_id)
        final_info = self._get_info() # Get base info like stacks, pot, etc.
        final_info.update(info) # Add round_over, round_reward etc. if round ended

        # The reward returned by step() itself is the per-round reward if round ended, else 0
        step_reward = agent_round_reward if final_info.get('round_over', False) else 0.0

        if self.render_mode == "human":
            self._render_frame()

        return obs, step_reward, terminated, truncated, final_info


    def _get_obs_dict(self, player_id):
         """ Returns the observation as a dictionary (useful for policies). """
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
             "num_active_players": len(self._get_active_players(include_all_in=False)),
         }
         return obs_dict

    def _get_obs(self, player_id):
        """
        Gets the observation for the specified player and encodes it using utils.encode_obs.
        """
        obs_dict = self._get_obs_dict(player_id)
        encoded_state = np.zeros(self.observation_space.shape, dtype=np.float32) # Default
        try:
            # Now relies on the updated utils.encode_obs
            from utils import encode_obs
            encoded_state = encode_obs(obs_dict) # Pass the dictionary

        except ImportError:
            print("ERROR: Could not import encode_obs from utils. Update utils.py!")
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
            "round_over": False,
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
        for i in range(self.num_players):
            # Try to import render_card for better display if available
            try: from card_utils import render_card
            except ImportError: render_card = lambda x: x # Fallback

            hand_str = " ".join(render_card(c) for c in self.hands.get(i, []))
            if not hand_str: hand_str = "? ?"
            stack_str = f"{self.stacks.get(i, 0):.2f}"
            bet_str = f"{self.current_bets.get(i, 0):.2f}"
            status = ""
            if i not in self.active_players_in_round and i in self.hands: status = " (Folded)"
            elif self.stacks.get(i,0) <= 0 and i in self.active_players_in_round: status = " (All-In)"
            elif i not in self.active_players_in_round: status = " (Out)"

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
         return self._get_legal_actions(self.agent_id)

     def get_player_hand(self, player_id):
          """ Helper to get a specific player's hand """
          return self.hands.get(player_id, [])

     def get_stacks(self, include_all=True):
          return self.stacks.copy()

     def get_pot(self):
          return self.pot

     def get_community_cards(self):
          return self.community_cards[:]

     def get_current_bets(self):
          return self.current_bets.copy()

