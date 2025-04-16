# filename: PokerBotRL/Back_End/constants.py
"""
Shared constants for the Poker RL project.
Moved here from utils.py to avoid circular dependencies.
"""

# --- Game Constants ---
NUM_PLAYERS = 6
STARTING_STACK = 10000
SUITS = ['H', 'D', 'C', 'S']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
DECK = [r + s for s in SUITS for r in RANKS]
CARD_TO_INDEX = {card: i for i, card in enumerate(DECK)}
INDEX_TO_CARD = {i: card for card, i in CARD_TO_INDEX.items()}
STAGES = ['preflop', 'flop', 'turn', 'river'] # Order matters for one-hot
STAGE_TO_INDEX = {stage: i for i, stage in enumerate(STAGES)}

# --- State Dimension ---
# 52 (Hole Cards) + 260 (Community Cards) + 1 (Pot) + 6 (Stacks) + 6 (Bets)
# + 4 (Stage) + 1 (Btn Pos) + 1 (Player Pos) + 2 (Blinds) = 333
NEW_STATE_DIM = 333