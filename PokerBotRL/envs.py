"""
envs.py

This module defines poker environment classes for training and evaluation.
It provides a common base class, BaseFullPokerEnv, which implements the core
poker game logic (dealing, betting, stage progression, hand evaluation, etc.)
and a subclass, TrainFullPokerEnv, that adds additional tracking (e.g., for
all-in events) and modified reward logic for training purposes.

Both classes share common helper functions such as hand evaluation and observation
encoding, ensuring consistency and reducing redundancy across scripts.
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
        if len(card) == 3:
            rank = card[:2]
            suit = card[2]
        else:
            rank = card[0]
            suit = card[1]
        parsed.append((rank_map[rank], suit))
    ranks = [r for r, s in parsed]
    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    counts = sorted(rank_counts.values(), reverse=True)
    suit_counts = {}
    for r, s in parsed:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    flush = any(count >= 5 for count in suit_counts.values())
    unique_ranks = sorted(set(ranks))
    straight = False
    straight_high = 0
    for i in range(len(unique_ranks) - 4):
        if unique_ranks[i+4] - unique_ranks[i] == 4:
            straight = True
            straight_high = unique_ranks[i+4]
    if flush and straight:
        hand_rank = 9
        tiebreaker = straight_high
    elif 4 in counts:
        hand_rank = 8
        quad = max(r for r, cnt in rank_counts.items() if cnt == 4)
        kicker = max(r for r in ranks if r != quad)
        tiebreaker = (quad, kicker)
    elif 3 in counts and 2 in counts:
        hand_rank = 7
        triple = max(r for r, cnt in rank_counts.items() if cnt == 3)
        pair = max(r for r, cnt in rank_counts.items() if cnt >= 2 and r != triple)
        tiebreaker = (triple, pair)
    elif flush:
        hand_rank = 6
        tiebreaker = sorted(ranks, reverse=True)
    elif straight:
        hand_rank = 5
        tiebreaker = straight_high
    elif 3 in counts:
        hand_rank = 4
        triple = max(r for r, cnt in rank_counts.items() if cnt == 3)
        kickers = sorted([r for r in ranks if r != triple], reverse=True)
        tiebreaker = (triple, kickers)
    elif counts.count(2) >= 2:
        hand_rank = 3
        pairs = sorted([r for r, cnt in rank_counts.items() if cnt == 2], reverse=True)
        remaining = [r for r in ranks if r not in pairs]
        kicker = max(remaining) if remaining else 0
        tiebreaker = (pairs, kicker)
    elif 2 in counts:
        hand_rank = 2
        pair = max(r for r, cnt in rank_counts.items() if cnt == 2)
        kickers = sorted([r for r in ranks if r != pair], reverse=True)
        tiebreaker = (pair, kickers)
    else:
        hand_rank = 1
        tiebreaker = sorted(ranks, reverse=True)
    return (hand_rank, tiebreaker)

# -----------------------------------------------------------------------------
# Base Poker Environment Class
# -----------------------------------------------------------------------------
class BaseFullPokerEnv:
    """
    BaseFullPokerEnv implements the core logic of a full poker game environment.
    It manages the deck, dealing, blinds, betting rounds, stage progression,
    hand evaluation, and opponent actions.

    Attributes:
        num_players (int): Number of players in the game.
        agent_id (int): The index of the RL agent.
        action_list (list): Allowed actions.
        full_deck (list): Complete deck of cards (e.g., '2H', 'AD').
        small_blind_amount (int): Small blind bet amount.
        big_blind_amount (int): Big blind bet amount.
        max_steps (int): Maximum allowed steps per round.
    """
    def __init__(self, num_players: int = 6, small_blind_amount: int = 50,
                 big_blind_amount: int = 100, max_steps: int = 100):
        self.num_players = num_players
        self.agent_id = 0
        self.action_list = ['fold', 'call', 'check', 'bet_small', 'bet_big', 'all_in']
        self.suits = ['H', 'D', 'C', 'S']
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.full_deck = [r + s for s in self.suits for r in self.ranks]
        self.small_blind_amount = small_blind_amount
        self.big_blind_amount = big_blind_amount

        self.dealer = -1  # so that first reset assigns dealer 0
        self.pot = 0
        self.stacks = {}
        self.hands = {}
        self.community_cards = []
        self.current_bets = {}
        self.current_max_bet = 0
        self.active_players = []
        self.players_to_act = []
        self.stage = None  # stages: preflop, flop, turn, river, showdown
        self.round_over = False
        self.opponent_policies = {pid: None for pid in range(1, self.num_players)}
        self.max_steps = max_steps
        self.steps_taken = 0

    def reset(self):
        """
        Resets the environment for a new round:
         - Shuffles the deck and deals cards to all players.
         - Assigns blinds.
         - Initializes betting and game stage.
        Returns:
            obs (dict): The observation for the agent.
        """
        self.steps_taken = 0
        self.stacks = {pid: 10000 for pid in range(self.num_players)}
        self.deck = self.full_deck.copy()
        random.shuffle(self.deck)
        self.community_cards = []
        self.pot = 0
        self.hands = {}
        for pid in range(self.num_players):
            self.hands[pid] = [self.deck.pop(), self.deck.pop()]

        self.dealer = (self.dealer + 1) % self.num_players
        self.small_blind = (self.dealer + 1) % self.num_players
        self.big_blind = (self.dealer + 2) % self.num_players

        self.current_bets = {pid: 0 for pid in range(self.num_players)}
        self.active_players = [pid for pid in range(self.num_players) if self.stacks[pid] > 0]

        # Post blinds.
        sb = self.small_blind
        bb = self.big_blind
        sb_bet = min(self.small_blind_amount, self.stacks[sb])
        self.stacks[sb] -= sb_bet
        self.current_bets[sb] = sb_bet
        self.pot += sb_bet

        bb_bet = min(self.big_blind_amount, self.stacks[bb])
        self.stacks[bb] -= bb_bet
        self.current_bets[bb] = bb_bet
        self.pot += bb_bet
        self.current_max_bet = bb_bet

        self.stage = "preflop"
        self.round_over = False

        # Determine order of actions starting from player after big blind.
        start = (bb + 1) % self.num_players
        self.players_to_act = []
        p = start
        while True:
            if p in self.active_players:
                self.players_to_act.append(p)
            if p == bb:
                break
            p = (p + 1) % self.num_players
        self.current_player = self.players_to_act[0] if self.players_to_act else None
        return self._get_obs(self.agent_id)

    def _compute_belief(self, player_id: int) -> dict:
        """
        Computes a simple belief state for a player: all cards not yet seen.
        """
        known = set(self.hands[player_id]) | set(self.community_cards)
        belief = {}
        for opp in range(self.num_players):
            if opp != player_id:
                belief[opp] = [card for card in self.full_deck if card not in known]
        return belief

    def _get_obs(self, player_id: int) -> dict:
        """
        Returns the observation for the given player.
        """
        obs = {
            'hand': self.hands[player_id],
            'community_cards': self.community_cards,
            'pot': self.pot,
            'current_bet': self.current_max_bet,
            'player_stack': self.stacks[player_id],
            'player_current_bet': self.current_bets[player_id],
            'stage': self.stage,
            'legal_actions': self._get_legal_actions(player_id),
            'beliefs': self._compute_belief(player_id)
        }
        return obs

    def _get_legal_actions(self, player_id: int) -> list:
        """
        Determines legal actions for the specified player.
        """
        if player_id not in self.active_players:
            return []
        legal = []
        player_bet = self.current_bets[player_id]
        if player_bet < self.current_max_bet:
            legal.extend(['fold', 'call', 'bet_small', 'bet_big', 'all_in'])
        else:
            legal.extend(['check', 'bet_small', 'bet_big', 'all_in', 'fold'])
        return legal

    def step(self, action: str):
        """
        Executes a step in the environment. The agent's action is processed,
        then opponent actions are taken until the agent's turn is reached again.
        The game stage is progressed as needed.

        Returns:
            (obs, reward, done, info)
        """
        self.steps_taken += 1
        if self.steps_taken >= self.max_steps:
            self.stage = "showdown"

        if self.current_player == self.agent_id:
            self._process_action(self.agent_id, action)

        self._update_players_to_act(self.current_player)

        while self.current_player is not None and self.current_player != self.agent_id:
            self.steps_taken += 1
            if self.steps_taken >= self.max_steps:
                self.stage = "showdown"
                break
            opp_action = self._select_opponent_action(self.current_player)
            self._process_action(self.current_player, opp_action)
            self._update_players_to_act(self.current_player)
            if not self.players_to_act:
                self._progress_stage()
                if self.steps_taken >= self.max_steps or self.stage == "showdown":
                    self.stage = "showdown"
                    break

        if self.current_player is None and self.stage != "showdown":
            self._progress_stage()

        if self.stage == "showdown":
            return self._finalize_hand()

        return self._get_obs(self.agent_id), 0, False, {}

    def _process_action(self, player: int, action: str) -> None:
        """
        Processes a player action, updating bets, stacks, and the pot accordingly.
        """
        if self.stacks[player] <= 0:
            if player in self.active_players:
                self.active_players.remove(player)
            return

        if action == 'fold':
            if player in self.active_players:
                self.active_players.remove(player)
            self.current_bets[player] = 0
        elif action in ['call', 'check']:
            call_amount = self.current_max_bet - self.current_bets[player]
            amount = min(call_amount, self.stacks[player])
            self.stacks[player] -= amount
            self.current_bets[player] += amount
            self.pot += amount
        elif action == 'bet_small':
            raise_amount = 50
            new_bet = self.current_max_bet + raise_amount
            amount_to_call = new_bet - self.current_bets[player]
            amount = min(amount_to_call, self.stacks[player])
            self.stacks[player] -= amount
            self.current_bets[player] += amount
            self.pot += amount
            if new_bet > self.current_max_bet:
                self.current_max_bet = self.current_bets[player]
                self.players_to_act = [p for p in self.active_players if p != player]
        elif action == 'bet_big':
            raise_amount = 100
            new_bet = self.current_max_bet + raise_amount
            amount_to_call = new_bet - self.current_bets[player]
            amount = min(amount_to_call, self.stacks[player])
            self.stacks[player] -= amount
            self.current_bets[player] += amount
            self.pot += amount
            if new_bet > self.current_max_bet:
                self.current_max_bet = self.current_bets[player]
                self.players_to_act = [p for p in self.active_players if p != player]
        elif action == 'all_in':
            amount = self.stacks[player]
            self.stacks[player] = 0
            self.current_bets[player] += amount
            self.pot += amount
            if self.current_bets[player] > self.current_max_bet:
                self.current_max_bet = self.current_bets[player]
                self.players_to_act = [p for p in self.active_players if p != player]
        if player in self.players_to_act:
            self.players_to_act.remove(player)

    def _update_players_to_act(self, current: int) -> None:
        """
        Updates the current player based on the remaining players to act.
        """
        if self.players_to_act:
            self.current_player = self.players_to_act[0]
        else:
            self.current_player = None

    def _progress_stage(self) -> None:
        """
        Progresses the game stage (from preflop to flop to turn to river to showdown)
        and resets per-round bets.
        """
        for pid in self.active_players:
            self.current_bets[pid] = 0
        self.current_max_bet = 0
        self.players_to_act = []

        if self.stage == "preflop":
            # Deal the flop (3 community cards)
            self.community_cards.extend([self.deck.pop() for _ in range(3)])
            self.stage = "flop"
        elif self.stage == "flop":
            self.community_cards.append(self.deck.pop())
            self.stage = "turn"
        elif self.stage == "turn":
            self.community_cards.append(self.deck.pop())
            self.stage = "river"
        elif self.stage == "river":
            self.stage = "showdown"

        start = (self.dealer + 1) % self.num_players
        order = []
        p = start
        while len(order) < len(self.active_players):
            if p in self.active_players:
                order.append(p)
            p = (p + 1) % self.num_players
        self.players_to_act = order
        self.current_player = self.players_to_act[0] if self.players_to_act else None

    def _finalize_hand(self):
        """
        Evaluates each active player's hand, determines the winners, updates stacks,
        and computes the agent's reward.
        """
        scores = {}
        for pid in self.active_players:
            full_hand = self.hands[pid] + self.community_cards
            scores[pid] = evaluate_hand(full_hand)
        winners = []
        best_score = None
        for pid, score in scores.items():
            if best_score is None or score > best_score:
                best_score = score
                winners = [pid]
            elif score == best_score:
                winners.append(pid)
        win_amount = self.pot / len(winners) if winners else 0
        for pid in winners:
            self.stacks[pid] += win_amount
        agent_reward = win_amount if self.agent_id in winners else -self.current_bets[self.agent_id]
        self.round_over = True
        obs = self._get_obs(self.agent_id)
        info = {'winners': winners, 'scores': scores}
        return obs, agent_reward, True, info

    def _select_opponent_action(self, player: int) -> str:
        """
        Selects an action for an opponent.
        If a fixed opponent policy is provided, it is used; otherwise a random legal action is chosen.
        """
        if self.opponent_policies[player] is not None:
            obs = self._get_obs(player)
            action = self.opponent_policies[player](obs)
            return action
        else:
            legal = self._get_legal_actions(player)
            return random.choice(legal) if legal else 'fold'

# -----------------------------------------------------------------------------
# Training Environment Subclass
# -----------------------------------------------------------------------------
class TrainFullPokerEnv(BaseFullPokerEnv):
    """
    TrainFullPokerEnv extends the base environment with additional features for training:
      - Tracking of all-in events via all_in_flag.
      - Modified reward computation: if the agent wins while having gone all in,
        a flat reward is provided.
      - Ensuring the full board is dealt prior to hand evaluation.
    """
    def __init__(self, *args, **kwargs):
        super(TrainFullPokerEnv, self).__init__(*args, **kwargs)
        self.all_in_flag = {pid: False for pid in range(self.num_players)}

    def reset(self):
        obs = super(TrainFullPokerEnv, self).reset()
        self.all_in_flag = {pid: False for pid in range(self.num_players)}
        return obs

    def _process_action(self, player: int, action: str) -> None:
        """
        Processes actions similarly to the base environment, but tracks
        all-in events by setting the corresponding flag.
        """
        if action == 'all_in':
            self.all_in_flag[player] = True
        super(TrainFullPokerEnv, self)._process_action(player, action)

    def _finalize_hand(self):
        """
        Overridden finalize hand function that ensures the full board is dealt
        and applies modified reward logic for training.
        """
        # Ensure all five community cards are dealt.
        while len(self.community_cards) < 5:
            self.community_cards.append(self.deck.pop())
        self.stage = "showdown"
        scores = {}
        for pid in self.active_players:
            full_hand = self.hands[pid] + self.community_cards
            scores[pid] = evaluate_hand(full_hand)
        winners = []
        best_score = None
        for pid, score in scores.items():
            if best_score is None or score > best_score:
                best_score = score
                winners = [pid]
            elif score == best_score:
                winners.append(pid)
        win_amount = self.pot / len(winners) if winners else 0
        for pid in winners:
            self.stacks[pid] += win_amount
        if self.agent_id in winners and self.all_in_flag.get(self.agent_id, False):
            agent_reward = 1000
        else:
            agent_reward = win_amount if self.agent_id in winners else -self.current_bets[self.agent_id]
        self.round_over = True
        obs = self._get_obs(self.agent_id)
        info = {'winners': winners, 'scores': scores}
        return obs, agent_reward, True, info
