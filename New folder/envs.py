"""
envs.py

This module defines poker environment classes for training and evaluation.
It provides a common base class, BaseFullPokerEnv, which implements the core
poker game logic (dealing, betting, stage progression, hand evaluation, etc.)
and a subclass, TrainFullPokerEnv, that adds additional tracking (e.g., for
all-in events) and modified reward logic for training purposes.

Both classes share common helper functions such as hand evaluation and observation
encoding, ensuring consistency and reducing redundancy across scripts.

MODIFIED:
- Stacks persist between rounds.
- An episode ends only when one player has all the chips.
- Rewards are calculated and returned after each round (showdown or folds).
"""

import random
from collections import deque
import numpy as np

# -----------------------------------------------------------------------------
# Helper Function: Hand Evaluator
# -----------------------------------------------------------------------------
def evaluate_hand(cards):
    """
    Evaluates a poker hand given a list of card strings.
    Each card is represented as 'RS' (e.g., '10H' or 'AS').

    Returns:
        tuple: (hand_rank, tiebreaker) where hand_rank is an integer score
               and tiebreaker is additional information for resolving ties.
    """
    rank_order = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    rank_map = {r: i for i, r in enumerate(rank_order, start=2)}
    parsed = []
    for card in cards:
        if len(card) == 3: # Handle '10' rank
            rank = card[:2]
            suit = card[2]
        else:
            rank = card[0]
            suit = card[1]
        # Ensure rank is valid before accessing rank_map
        if rank in rank_map:
             parsed.append((rank_map[rank], suit))
        else:
            # Handle potential invalid card format gracefully
            print(f"Warning: Invalid card format detected - '{card}'")
            # Option: Skip card, assign default value, or raise error
            # For now, let's skip it to avoid crashing
            continue

    # Basic check if enough cards were parsed
    if len(parsed) < 5:
        # Not enough valid cards to form a 5-card hand
        # Return lowest possible rank or handle as appropriate
        return (0, []) # Example: Return rank 0 (High Card) with empty tiebreaker


    ranks = sorted([r for r, s in parsed], reverse=True) # Sort ranks descending for easier processing
    suits = [s for r, s in parsed]
    rank_counts = {r: ranks.count(r) for r in set(ranks)}
    suit_counts = {s: suits.count(s) for s in set(suits)}

    counts_values = sorted(rank_counts.values(), reverse=True)
    is_flush = any(count >= 5 for count in suit_counts.values())

    # Use a set for faster straight checking and handle Ace-low straight (wheel)
    unique_ranks = sorted(list(set(ranks)), reverse=True)
    is_straight = False
    straight_high_card = 0

    # Check for standard straights
    for i in range(len(unique_ranks) - 4):
        # Check for 5 consecutive ranks
        if all(unique_ranks[i] - j == unique_ranks[i+j] for j in range(1, 5)):
            is_straight = True
            straight_high_card = unique_ranks[i]
            break

    # Check for Ace-low straight (A, 2, 3, 4, 5) - ranks 14, 2, 3, 4, 5
    if not is_straight and set([14, 2, 3, 4, 5]).issubset(set(ranks)):
        is_straight = True
        straight_high_card = 5 # High card is 5 for the wheel

    # Determine hand rank based on combinations
    if is_straight and is_flush:
        # Need to find the highest card of the straight flush
        flush_suit = [s for s, count in suit_counts.items() if count >= 5][0]
        flush_ranks = sorted([r for r, s in parsed if s == flush_suit], reverse=True)
        # Check for straight within the flush ranks
        sf_high_card = 0
        # Standard straight flush check
        for i in range(len(flush_ranks) - 4):
             if all(flush_ranks[i] - j == flush_ranks[i+j] for j in range(1, 5)):
                 sf_high_card = flush_ranks[i]
                 break
        # Ace-low straight flush check
        if sf_high_card == 0 and set([14, 2, 3, 4, 5]).issubset(set(flush_ranks)):
             sf_high_card = 5

        if sf_high_card > 0:
             # Royal flush check (special case of straight flush)
             if sf_high_card == 14 and all(r in flush_ranks for r in [14, 13, 12, 11, 10]):
                 return (10, []) # Rank 10 for Royal Flush
             else:
                 return (9, [sf_high_card]) # Rank 9 for Straight Flush

    if 4 in counts_values:
        quad_rank = [r for r, count in rank_counts.items() if count == 4][0]
        kickers = sorted([r for r in ranks if r != quad_rank], reverse=True)
        return (8, [quad_rank] + kickers[:1]) # Rank 8 for Four of a Kind
    if counts_values == [3, 2] or counts_values == [3, 1, 1]: # Handles 7 cards where pair might not be highest
        three_rank = [r for r, count in rank_counts.items() if count == 3][0]
        pair_ranks = sorted([r for r, count in rank_counts.items() if count >= 2 and r != three_rank], reverse=True)
        if pair_ranks: # Found a pair for the full house
             return (7, [three_rank, pair_ranks[0]]) # Rank 7 for Full House
        else: # Only trips, handle below
             pass
    if is_flush:
         flush_suit = [s for s, count in suit_counts.items() if count >= 5][0]
         flush_ranks = sorted([r for r, s in parsed if s == flush_suit], reverse=True)
         return (6, flush_ranks[:5]) # Rank 6 for Flush
    if is_straight:
         return (5, [straight_high_card]) # Rank 5 for Straight
    if 3 in counts_values:
         three_rank = [r for r, count in rank_counts.items() if count == 3][0]
         kickers = sorted([r for r in ranks if r != three_rank], reverse=True)
         return (4, [three_rank] + kickers[:2]) # Rank 4 for Three of a Kind
    if counts_values.count(2) >= 2: # Two Pair
         pair_ranks = sorted([r for r, count in rank_counts.items() if count == 2], reverse=True)
         kickers = sorted([r for r in ranks if r not in pair_ranks[:2]], reverse=True)
         return (3, pair_ranks[:2] + kickers[:1]) # Rank 3 for Two Pair
    if 2 in counts_values: # One Pair
         pair_rank = [r for r, count in rank_counts.items() if count == 2][0]
         kickers = sorted([r for r in ranks if r != pair_rank], reverse=True)
         return (2, [pair_rank] + kickers[:3]) # Rank 2 for One Pair
    else: # High Card
         return (1, ranks[:5]) # Rank 1 for High Card


# -----------------------------------------------------------------------------
# Base Poker Environment Class
# -----------------------------------------------------------------------------
class BaseFullPokerEnv:
    """
    BaseFullPokerEnv implements the core logic of a full poker game environment.
    It manages the deck, dealing, blinds, betting rounds, stage progression,
    hand evaluation, and opponent actions. Stacks persist between rounds, and
    the game ends when only one player remains with chips. Rewards are given per round.

    Attributes:
        num_players (int): Number of players in the game.
        agent_id (int): The index of the RL agent (often 0).
        action_list (list): Allowed actions.
        full_deck (list): Complete deck of cards (e.g., '2H', 'AD').
        small_blind_amount (int): Small blind bet amount.
        big_blind_amount (int): Big blind bet amount.
        max_steps_per_round (int): Max steps per betting round to prevent infinite loops.
        initial_stack (int): Starting stack size for each player at the beginning of a match.
    """
    def __init__(self, num_players: int = 6, initial_stack: int = 10000,
                 small_blind_amount: int = 50, big_blind_amount: int = 100,
                 max_steps_per_round: int = 100000):
        self.num_players = num_players
        self.agent_id = 0 # Default agent ID
        self.action_list = ['fold', 'call', 'check', 'bet_small', 'bet_big', 'all_in']
        self.suits = ['H', 'D', 'C', 'S']
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.full_deck = [r + s for s in self.suits for r in self.ranks]
        self.small_blind_amount = small_blind_amount
        self.big_blind_amount = big_blind_amount
        self.initial_stack = initial_stack
        self.max_steps_per_round = max_steps_per_round

        self.stacks = {} # Initialized in reset() only once
        self.dealer = -1 # So that first reset assigns dealer 0
        self.opponent_policies = {pid: None for pid in range(self.num_players)} # Placeholder for opponent logic

        # Round specific state (reset each round by _start_new_round)
        self.pot = 0
        self.deck = []
        self.hands = {}
        self.community_cards = []
        self.current_bets = {} # Tracks bets *within* the current round/stage
        self.round_total_bets = {} # Tracks total bets *for the entire round* per player
        self.current_max_bet = 0 # Max bet level in the current betting stage
        self.active_players_in_round = [] # Players still eligible to win the pot this round
        self.players_to_act = deque() # Queue for whose turn it is
        self.stage = None # Stages: preflop, flop, turn, river, showdown, round_end
        self.steps_taken_this_round = 0
        self.last_raiser = None # Tracks the last player who raised in the current betting stage

    def _initialize_stacks(self):
        """Initializes stacks only if they haven't been set."""
        if not self.stacks:
            self.stacks = {pid: self.initial_stack for pid in range(self.num_players)}
            print(f"Initialized stacks: {self.stacks}") # Debugging

    def _start_new_round(self):
        """
        Sets up the environment for a new round, preserving stacks.
        - Determines active players based on stacks > 0.
        - Shuffles the deck and deals cards.
        - Assigns blinds.
        - Initializes betting state for the new round.
        Returns:
            bool: True if the round can start (>=2 players), False otherwise.
        """
        self.steps_taken_this_round = 0
        self.pot = 0
        self.community_cards = []
        self.current_bets = {pid: 0 for pid in range(self.num_players)}
        self.round_total_bets = {pid: 0 for pid in range(self.num_players)}
        self.current_max_bet = 0
        self.last_raiser = None
        self.stage = "preflop"

        # --- Determine players active for this new round ---
        players_with_stacks = [pid for pid in range(self.num_players) if self.stacks.get(pid, 0) > 0]

        if len(players_with_stacks) < 2:
            print("Not enough players with stacks to start a new round.") # Debugging
            return False # Game over condition handled in step/finalize

        self.active_players_in_round = players_with_stacks.copy()
        print(f"Starting round with active players: {self.active_players_in_round}, Stacks: {self.stacks}") # Debugging


        # --- Deck and Hands ---
        self.deck = self.full_deck.copy()
        random.shuffle(self.deck)
        self.hands = {}
        for pid in self.active_players_in_round:
            if len(self.deck) >= 2:
                 self.hands[pid] = [self.deck.pop(), self.deck.pop()]
            else:
                 # Should not happen with standard deck/player counts, but good practice
                 print("Warning: Not enough cards in deck to deal hands.")
                 # Handle error: maybe invalidate round or end game
                 return False # Indicate round cannot start properly

        # --- Blinds ---
        # Find next valid dealer position among active players
        current_dealer_idx = -1
        if self.dealer in self.active_players_in_round:
            current_dealer_idx = self.active_players_in_round.index(self.dealer)

        next_dealer_idx = (current_dealer_idx + 1) % len(self.active_players_in_round)
        self.dealer = self.active_players_in_round[next_dealer_idx]

        # Find SB and BB positions relative to the dealer among active players
        num_active = len(self.active_players_in_round)
        if num_active < 2: return False # Should already be caught, but double check

        sb_idx = (next_dealer_idx + 1) % num_active
        bb_idx = (next_dealer_idx + 2) % num_active

        self.small_blind_pos = self.active_players_in_round[sb_idx]
        self.big_blind_pos = self.active_players_in_round[bb_idx]

        # Handle heads-up case (Dealer is SB, other player is BB)
        if num_active == 2:
            self.small_blind_pos = self.dealer
            self.big_blind_pos = self.active_players_in_round[(next_dealer_idx + 1) % num_active]

        # Post blinds
        sb_amount = min(self.small_blind_amount, self.stacks[self.small_blind_pos])
        self._add_bet(self.small_blind_pos, sb_amount)

        bb_amount = min(self.big_blind_amount, self.stacks[self.big_blind_pos])
        self._add_bet(self.big_blind_pos, bb_amount)

        self.current_max_bet = self.current_bets[self.big_blind_pos] # BB sets the initial bet level
        self.last_raiser = self.big_blind_pos # Initially, the BB is the 'raiser'

        # --- Determine initial player order ---
        start_idx = (bb_idx + 1) % num_active
        self.players_to_act.clear()
        # Create the order based on active players list indices
        ordered_active_players = self.active_players_in_round[start_idx:] + self.active_players_in_round[:start_idx]

        # Filter out players already all-in from posting blinds (they can't act)
        self.players_to_act.extend([p for p in ordered_active_players if self.stacks[p] > 0])

        # If only one player can act (e.g., everyone else folded/all-in pre-flop), the round might end quickly.
        # This is handled naturally by the step loop.

        return True # Round started successfully


    def reset(self):
        """
        Resets the environment for a new MATCH (if stacks empty) or starts the
        first round of a match. Stacks are only initialized once per instance.
        Returns:
            obs (dict): The observation for the agent.
            info (dict): Additional information (e.g., indicates if game cannot start).
        """
        self._initialize_stacks() # Initialize stacks if first time

        if not self._start_new_round():
            # Cannot start the first round (e.g., < 2 players initially)
            return self._get_obs(self.agent_id), {"error": "Cannot start game, not enough players."}

        # Return initial observation
        return self._get_obs(self.agent_id), {}

    def _get_obs(self, player_id: int) -> dict:
        """
        Returns the observation for the given player.
        Handles cases where player might not have a hand (e.g., busted).
        """
        if player_id not in self.stacks or self.stacks[player_id] <= 0:
            # Player is out or observation requested before game start
             return {
                'hand': [], 'community_cards': self.community_cards, 'pot': self.pot,
                'current_bet': self.current_max_bet, 'player_stack': 0,
                'player_current_bet': 0, 'stage': self.stage,
                'legal_actions': [], 'player_round_total_bet': 0,
                'opponent_stacks': {}, 'opponent_bets': {}, 'active_players': []
             }

        # Calculate opponent info safely
        opponent_stacks = {p: self.stacks.get(p, 0) for p in range(self.num_players) if p != player_id}
        opponent_bets = {p: self.current_bets.get(p, 0) for p in range(self.num_players) if p != player_id}


        obs = {
            'hand': self.hands.get(player_id, []), # Use .get for safety
            'community_cards': self.community_cards,
            'pot': self.pot,
            'current_bet': self.current_max_bet, # Amount to call/raise over
            'player_stack': self.stacks.get(player_id, 0),
            'player_current_bet': self.current_bets.get(player_id, 0), # Bet in this stage
            'player_round_total_bet': self.round_total_bets.get(player_id, 0), # Total bet this round
            'stage': self.stage,
            'legal_actions': self._get_legal_actions(player_id),
            'active_players': self.active_players_in_round[:], # Current active players this round
            'opponent_stacks': opponent_stacks,
            'opponent_bets': opponent_bets, # Opponent bets in this stage
             # 'beliefs': self._compute_belief(player_id) # Beliefs can be complex, omitted for brevity
        }
        return obs

    def _get_legal_actions(self, player_id: int) -> list:
        """
        Determines legal actions for the specified player based on current game state.
        """
        # If player is not active this round, or has no stack, or it's not their turn (though checks happen before call)
        if player_id not in self.active_players_in_round or self.stacks[player_id] <= 0:
            return []

        legal = []
        player_bet_this_stage = self.current_bets.get(player_id, 0)
        amount_to_call = self.current_max_bet - player_bet_this_stage
        can_afford_call = self.stacks[player_id] >= amount_to_call
        can_raise = self.stacks[player_id] > amount_to_call # Must have chips *beyond* the call amount to raise

        # Fold is always possible if there's a bet to call
        if amount_to_call > 0:
            legal.append('fold')

        # Call / Check
        if amount_to_call == 0:
            legal.append('check')
        elif can_afford_call:
            legal.append('call')
        else: # Cannot afford call, must go all-in or fold
             legal.append('all_in') # Effectively an all-in call
             if 'fold' not in legal: legal.append('fold') # Still need fold if possible
             return legal # No other betting options possible

        # Betting/Raising Actions (only if player can raise)
        if can_raise:
            # Check if any bet/raise is possible
            # Small Bet (e.g., min raise or % pot/stack) - simplified here
            min_raise_amount = self.big_blind_amount # Simplified minimum raise
            potential_small_bet = self.current_max_bet + min_raise_amount
            cost_of_small_bet = potential_small_bet - player_bet_this_stage
            if self.stacks[player_id] >= cost_of_small_bet:
                 legal.append('bet_small')

            # Big Bet (e.g., larger % pot/stack) - simplified here
            potential_big_bet = self.current_max_bet + 2 * min_raise_amount # Example
            cost_of_big_bet = potential_big_bet - player_bet_this_stage
            if self.stacks[player_id] >= cost_of_big_bet:
                 legal.append('bet_big')

            # All-in is always a betting option if you can raise
            legal.append('all_in')


        # Remove duplicates just in case and ensure 'fold' is first if present
        legal = sorted(list(set(legal)), key=lambda x: (x != 'fold', x))
        return legal

    def _add_bet(self, player: int, amount: int):
        """Helper to add a bet, updating stacks, pot, and bet tracking."""
        actual_amount = min(amount, self.stacks[player]) # Can't bet more than stack
        self.stacks[player] -= actual_amount
        self.current_bets[player] = self.current_bets.get(player, 0) + actual_amount
        self.round_total_bets[player] = self.round_total_bets.get(player, 0) + actual_amount
        self.pot += actual_amount
        # print(f"Player {player} bets {actual_amount}. New stack: {self.stacks[player]}, Pot: {self.pot}, CurrentBet: {self.current_bets[player]}") # Debug

    def step(self, action: str):
        """
        Executes a step in the environment based on the agent's action.
        Processes the agent's action, then simulates opponent actions until:
         - It's the agent's turn again.
         - The betting round ends.
         - The hand ends (showdown or folds).
         - The game ends (one player left).

        Returns:
            (obs, reward, done, info) :
                obs (dict): Observation for the agent for the *next* state.
                reward (float): Reward obtained from the *completed* round/action.
                done (bool): True if the entire game (match) is over.
                info (dict): Additional information (winners, scores, errors).
        """
        current_player_id = self.players_to_act[0] if self.players_to_act else None

        # --- Validate Agent Action ---
        if current_player_id != self.agent_id:
             # This shouldn't happen if called correctly, but handle defensively
             return self._get_obs(self.agent_id), 0, self._is_game_over(), {"error": "Agent action provided when not agent's turn."}

        legal_actions = self._get_legal_actions(self.agent_id)
        if action not in legal_actions:
            # Handle illegal action (e.g., default to fold or check)
            print(f"Warning: Agent chose illegal action '{action}'. Legal: {legal_actions}. Defaulting.") # Debug
            if 'fold' in legal_actions: action = 'fold'
            elif 'check' in legal_actions: action = 'check'
            else: action = legal_actions[0] # Should maybe be all-in if only option?


        # --- Process Agent Action ---
        round_ended, info = self._process_action(self.agent_id, action)
        # info might contain details if the action ended the round (e.g., everyone else folded)


        # --- Simulate Opponent Actions ---
        # Loop while the round is not ended and it's not the agent's turn
        while not round_ended and self.players_to_act and self.players_to_act[0] != self.agent_id:
            opponent_id = self.players_to_act[0]
            opp_action = self._select_opponent_action(opponent_id)
            round_ended, info = self._process_action(opponent_id, opp_action)
            if round_ended: break # Exit loop if opponent action ends the round

            # Check for excessive steps in the round (safety break)
            self.steps_taken_this_round += 1
            if self.steps_taken_this_round > self.max_steps_per_round * self.num_players:
                print("Warning: Max steps per round exceeded. Forcing showdown.")
                round_ended = True
                self.stage = "showdown" # Force showdown
                # Potentially penalize or handle this specific end condition
                info = {"error": "Max steps per round exceeded."}
                break


        # --- Round End Handling ---
        reward = 0 # Default reward for intermediate steps
        done = self._is_game_over() # Check if the game ended after the actions

        if round_ended and not done:
             # Finalize the hand/round, get reward, and potentially start next round
             obs, reward, round_done_flag, info = self._finalize_round(info) # Pass info in case round ended by folds

             # Check game over *after* finalizing the round and distributing pot
             done = self._is_game_over()

             if not done:
                 # Start the next round
                 if not self._start_new_round():
                      # If starting next round fails (e.g. not enough players left after pot distribution)
                      done = True
                      print("Game ended: Cannot start next round.") # Debugging
                      # Provide final observation state?
                      obs = self._get_obs(self.agent_id) # Get obs after failed round start
                 else:
                      # Successfully started new round, get obs for the *new* round
                      obs = self._get_obs(self.agent_id)
                 # Return obs for new round state, reward from completed round, done status, info
                 return obs, reward, done, info
             else:
                 # Game is over after this round was finalized
                 print(f"Game ended. Final stacks: {self.stacks}") # Debugging
                 return obs, reward, done, info # Return final obs, reward, done=True, info

        elif done: # Game ended mid-round (e.g. only one player left active) - Should be caught by _is_game_over earlier?
             print(f"Game ended mid-round? Final stacks: {self.stacks}") # Debugging - this case might need refinement
             # Ensure round is finalized if it hasn't been
             if self.stage != "round_end":
                  obs, reward, _, info = self._finalize_round(info)
             else: # Round already somehow finalized
                  obs = self._get_obs(self.agent_id) # Get current obs
             return obs, reward, True, info

        else:
            # Round continues, it's agent's turn again or betting round finished
            # Get current observation for the agent
            obs = self._get_obs(self.agent_id)
            # Intermediate step reward (optional, can be 0)
            reward = 0 #-0.01 # Small penalty per action?
            return obs, reward, False, info # Return current obs, 0 reward, not done, info


    def _process_action(self, player: int, action: str) -> (bool, dict):
        """
        Processes a player action, updating bets, stacks, pot, and player turn queue.
        Determines if the action ends the betting stage or the round.

        Returns:
            (round_ended, info): Tuple indicating if the round ended and optional info.
        """
        info = {}
        player_bet_this_stage = self.current_bets.get(player, 0)
        amount_to_call = self.current_max_bet - player_bet_this_stage

        if action == 'fold':
            if player in self.active_players_in_round:
                self.active_players_in_round.remove(player)
            # No bet added
            self.players_to_act.popleft() # Player is done for this stage

        elif action == 'check':
            # Only legal if amount_to_call is 0
            if amount_to_call != 0:
                 print(f"Warning: Player {player} checked illegally. Treating as fold.") # Handle illegal check
                 if player in self.active_players_in_round: self.active_players_in_round.remove(player)
            self.players_to_act.popleft() # Player is done for this stage

        elif action == 'call':
            amount = min(amount_to_call, self.stacks[player])
            self._add_bet(player, amount)
            self.players_to_act.popleft() # Player is done for this stage

        elif action == 'all_in':
            amount = self.stacks[player]
            self._add_bet(player, amount)
            if self.current_bets[player] > self.current_max_bet:
                 # All-in constitutes a raise
                 self.current_max_bet = self.current_bets[player]
                 self.last_raiser = player
                 # Resetting who needs to act after a raise
                 self._reset_acting_queue_after_raise(player)
            else:
                 # All-in was just a call
                 self.players_to_act.popleft() # Player is done for this stage

        elif action.startswith('bet'): # Handles bet_small, bet_big
            # Determine raise amount based on action (simplified)
            # TODO: Implement dynamic bet sizing based on pot/stack/etc.
            min_raise = self.big_blind_amount # Simple min raise
            if action == 'bet_small':
                 raise_amount = min_raise
            elif action == 'bet_big':
                 raise_amount = 2 * min_raise # Example sizing
            else: # Fallback/error
                 raise_amount = min_raise

            # Calculate total bet amount for player this stage
            total_bet_this_stage = self.current_max_bet + raise_amount
            # Amount needed *additional* to current bet
            amount_to_add = total_bet_this_stage - player_bet_this_stage
            # Ensure player has enough, cap at their stack (implicitly becomes all-in raise if necessary)
            actual_amount_to_add = min(amount_to_add, self.stacks[player])
            # Make the bet
            self._add_bet(player, actual_amount_to_add)

            # Update game state for the raise
            self.current_max_bet = self.current_bets[player]
            self.last_raiser = player
            self._reset_acting_queue_after_raise(player)


        # --- Check for End of Betting Stage / Round ---
        # Condition 1: Only one player left active in the round -> Round Ends
        if len(self.active_players_in_round) <= 1:
            self.stage = "round_end" # Mark round as ended due to folds
            info['round_end_reason'] = 'folds'
            print(f"Round ended: Player {self.active_players_in_round[0] if self.active_players_in_round else 'N/A'} wins pot {self.pot} by default.") # Debug
            return True, info # Signal round end

        # Condition 2: Betting round complete (everyone called the last raise or checked around)
        # Check if queue is empty OR if the next player to act is the last raiser and they haven't raised again this turn
        # Also need to ensure everyone still in has bet the same amount (or is all-in)
        betting_complete = False
        if not self.players_to_act: # Everyone acted
             betting_complete = True
        elif self.players_to_act:
             # Check if everyone left in the queue has matched the max bet or is all-in
             all_matched = True
             for p_id in self.players_to_act:
                 if self.stacks[p_id] > 0 and self.current_bets.get(p_id, 0) < self.current_max_bet:
                      all_matched = False
                      break
             if all_matched:
                  # Special case: BB option. If it's BB's turn pre-flop, they were the last raiser initially,
                  # and no one re-raised, they get an option to raise.
                  is_preflop_bb_option = (self.stage == "preflop" and
                                          self.players_to_act[0] == self.big_blind_pos and
                                          self.last_raiser == self.big_blind_pos and # No re-raise happened
                                          self.current_bets.get(self.big_blind_pos, 0) == self.big_blind_amount)

                  if not is_preflop_bb_option:
                      betting_complete = True

        if betting_complete:
            # Proceed to next stage or finish round
            self._progress_stage()
            if self.stage == "showdown" or self.stage == "round_end":
                info.setdefault('round_end_reason', 'showdown') # Add reason if not already set
                return True, info # Signal round end (reached showdown)
            else:
                # Betting stage ended, but round continues
                return False, info # Signal stage end, not round end

        # Condition 3: Action continues
        return False, info # Signal action processed, round continues


    def _reset_acting_queue_after_raise(self, raiser_id):
        """Helper to reset the queue of players to act after a raise."""
        self.players_to_act.clear()
        raiser_idx = -1
        try: # Find index of raiser in the current active list
            raiser_idx = self.active_players_in_round.index(raiser_id)
        except ValueError:
            print(f"Error: Raiser {raiser_id} not found in active players {self.active_players_in_round}")
            return # Should not happen

        num_active = len(self.active_players_in_round)
        # Add players starting from the one after the raiser, wrapping around
        for i in range(1, num_active):
            p_idx = (raiser_idx + i) % num_active
            player = self.active_players_in_round[p_idx]
            # Only add players who are not all-in and haven't folded
            if self.stacks.get(player, 0) > 0:
                 self.players_to_act.append(player)

        # print(f"Queue reset after raise by {raiser_id}. New queue: {list(self.players_to_act)}") # Debug


    def _progress_stage(self) -> None:
        """
        Progresses the game stage (preflop -> flop -> turn -> river -> showdown)
        and resets betting state for the new stage.
        """
        # Reset stage-specific betting
        self.current_max_bet = 0
        self.last_raiser = None
        for pid in self.active_players_in_round:
            self.current_bets[pid] = 0 # Reset bets for the new street

        # Deal community cards based on stage
        if self.stage == "preflop":
            self.stage = "flop"
            for _ in range(3):
                if self.deck: self.community_cards.append(self.deck.pop())
        elif self.stage == "flop":
            self.stage = "turn"
            if self.deck: self.community_cards.append(self.deck.pop())
        elif self.stage == "turn":
            self.stage = "river"
            if self.deck: self.community_cards.append(self.deck.pop())
        elif self.stage == "river":
            self.stage = "showdown" # Ready for showdown
            self.players_to_act.clear() # No more actions in showdown stage
            return # Don't reset queue for showdown

        # --- Determine player order for the new stage ---
        self.players_to_act.clear()
        # Action starts with the first active player to the left of the dealer
        if not self.active_players_in_round: return # Should not happen if stage progresses

        dealer_idx = -1
        try:
             dealer_idx = self.active_players_in_round.index(self.dealer)
        except ValueError:
              # Dealer might have folded, find first active player conceptually after dealer
              pos = (self.dealer + 1) % self.num_players
              while pos not in self.active_players_in_round:
                   pos = (pos + 1) % self.num_players
                   if pos == self.dealer: break # Safety break
              # Find index of this player
              try:
                   dealer_idx = self.active_players_in_round.index(pos) -1 # Start from player *after* this one
              except ValueError:
                    dealer_idx = -1 # Fallback: start from first active player


        num_active = len(self.active_players_in_round)
        start_idx = (dealer_idx + 1) % num_active

        # Add players in order, starting from start_idx, wrapping around
        ordered_players = self.active_players_in_round[start_idx:] + self.active_players_in_round[:start_idx]
        self.players_to_act.extend([p for p in ordered_players if self.stacks.get(p, 0) > 0])

        # print(f"Stage progressed to {self.stage}. Community: {self.community_cards}. Acting queue: {list(self.players_to_act)}") # Debug


    # Inside the TrainFullPokerEnv class in envs.py

    def _finalize_round(self, info):
        """
        Overridden finalize round function for training.
        Evaluates hands, distributes pot, calculates potentially modified reward,
        and checks for game end condition OR if agent 0 has busted.

        Returns:
            (obs, reward, done, info)
        """
        winners = []
        scores = {}
        agent_reward = 0 # Initialize reward

        # --- Determine Winner(s) ---
        # Ensure full board is dealt before evaluation if needed
        while len(self.community_cards) < 5 and self.deck:
            self.community_cards.append(self.deck.pop())

        eligible_players = [p for p in self.active_players_in_round if p in self.hands]

        if len(eligible_players) == 1: # Won by default
            winners = eligible_players[:]
            scores[winners[0]] = "Won by default"
        elif len(eligible_players) > 1: # Showdown evaluation
            best_score = (-1, [])
            for pid in eligible_players:
                full_hand_cards = self.hands[pid] + self.community_cards
                scores[pid] = evaluate_hand(full_hand_cards)
                if scores[pid] > best_score:
                    best_score = scores[pid]
                    winners = [pid]
                elif scores[pid] == best_score:
                    winners.append(pid)
        else: # Error case
             print("Warning: Finalize round called with no eligible players.")
             winners = []

        # --- Distribute Pot ---
        win_amount_per_winner = 0 # Initialize
        if winners:
            win_amount_per_winner = self.pot / len(winners)
            for pid in winners:
                self.stacks[pid] = self.stacks.get(pid, 0) + win_amount_per_winner
        else:
             print(f"Warning: No winners determined for pot {self.pot}.")

        # --- Calculate Agent Reward (potentially modified for training) ---
        agent_initial_bet = self.round_total_bets.get(self.agent_id, 0)
        agent_winnings = win_amount_per_winner if self.agent_id in winners else 0

        # Example training modification: Bonus for winning when all-in
        agent_was_all_in = self.all_in_flag_round.get(self.agent_id, False)
        if agent_winnings > 0 and agent_was_all_in:
             # Define a bonus (e.g., flat amount, percentage of pot, etc.)
             all_in_win_bonus = 100 # Example flat bonus
             agent_reward = (agent_winnings - agent_initial_bet) + all_in_win_bonus
             info['training_bonus'] = all_in_win_bonus # Log the bonus
             # print(f"Agent {self.agent_id} got all-in win bonus: {all_in_win_bonus}") # Debug
        else:
             # Standard reward calculation
             agent_reward = agent_winnings - agent_initial_bet


        # --- Prepare Info ---
        info['winners'] = winners
        info['scores'] = scores
        info['agent_reward'] = agent_reward
        info['final_stacks'] = self.stacks.copy() # Stacks *after* pot distribution

        # --- Check Game Over Condition ---
        self.stage = "round_end" # Mark round as officially over

        # MODIFICATION: Check if agent 0 busted OR if only one player remains
        game_over_normally = self._is_game_over()
        agent_busted = self.stacks.get(self.agent_id, 0) <= 0

        # Episode is done if game ends normally OR if the training agent busted
        done = game_over_normally or agent_busted

        if not game_over_normally and agent_busted:
             print(f"Agent {self.agent_id} busted. Ending training episode.") # Log reason

        # --- Get final observation ---
        obs = self._get_obs(self.agent_id)

        return obs, agent_reward, done, info

    def _is_game_over(self):
        """Checks if only one player has chips left."""
        players_with_chips = [pid for pid, stack in self.stacks.items() if stack > 0]
        return len(players_with_chips) <= 1


    def _select_opponent_action(self, player: int) -> str:
        """
        Selects an action for an opponent based on policy or randomness.
        """
        if self.opponent_policies.get(player) is not None:
            # Use a provided policy function/model
            obs = self._get_obs(player)
            try:
                action = self.opponent_policies[player](obs)
                legal = self._get_legal_actions(player)
                if action not in legal:
                     # Policy returned illegal action, fallback
                     print(f"Warning: Opponent {player} policy returned illegal action '{action}'. Legal: {legal}. Defaulting.")
                     if 'fold' in legal: action = 'fold'
                     elif 'check' in legal: action = 'check'
                     else: action = legal[0] # Fallback
                return action
            except Exception as e:
                 print(f"Error executing opponent {player} policy: {e}. Defaulting.")
                 # Fallback to random if policy fails
                 pass # Fall through to random choice

        # Default: Choose a random legal action
        legal = self._get_legal_actions(player)
        if not legal:
            # This might happen if player is forced all-in previously and has no actions
            # Or if logic error occurred. Ensure game progresses.
            print(f"Warning: Opponent {player} has no legal actions, but it's their turn?")
            return 'check' # Default to check if somehow possible, though likely error state

        # Simple random choice - improve this for better opponents!
        # Prioritize check/call/fold over betting for basic random opponent
        preferred_actions = [a for a in ['check', 'call', 'fold'] if a in legal]
        if preferred_actions and random.random() < 0.8: # 80% chance to play passively
            return random.choice(preferred_actions)
        else: # Bet/raise or fallback if no passive options
            return random.choice(legal)


# -----------------------------------------------------------------------------
# Training Environment Subclass (Example - may need adjustments)
# -----------------------------------------------------------------------------
class TrainFullPokerEnv(BaseFullPokerEnv):
    """
    TrainFullPokerEnv extends the base environment.
    Can add specific features for training like modified rewards or observation spaces.
    Inherits the round continuation and persistent stack logic.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add any training-specific attributes here
        self.all_in_flag_round = {pid: False for pid in range(self.num_players)}

    def _start_new_round(self):
        """Reset training-specific flags when a new round starts."""
        self.all_in_flag_round = {pid: False for pid in range(self.num_players)}
        return super()._start_new_round() # Call base class method

    def _process_action(self, player: int, action: str) -> (bool, dict):
        """Track all-ins during the round."""
        if action == 'all_in' or (action == 'call' and self.stacks[player] == self.current_max_bet - self.current_bets.get(player, 0)):
             # Consider call that uses exactly remaining stack as all-in
             self.all_in_flag_round[player] = True
        return super()._process_action(player, action)

    def _finalize_round(self, info):
        """Apply potentially modified reward logic for training."""
        # Get results from base class finalize
        obs, base_reward, done, info = super()._finalize_round(info)

        # --- Training-Specific Reward Modification ---
        # Example: Add a bonus for winning a round after going all-in
        agent_won = self.agent_id in info.get('winners', [])
        agent_was_all_in = self.all_in_flag_round.get(self.agent_id, False)

        modified_reward = base_reward
        if agent_won and agent_was_all_in:
            # Example bonus - adjust magnitude as needed
            all_in_bonus = 100 # Flat bonus? Or % of pot?
            modified_reward += all_in_bonus
            info['training_bonus'] = all_in_bonus # Add bonus info for logging
            # print(f"Agent {self.agent_id} got all-in win bonus: {all_in_bonus}") # Debug

        # You could add other reward shaping here (e.g., penalties for folding good hands)

        info['agent_reward'] = modified_reward # Update info dict with potentially modified reward

        return obs, modified_reward, done, info # Return modified reward