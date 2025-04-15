# filename: utils.py
"""
utils.py

This module provides common utilities for the poker RL project.

MODIFIED (Tournament State Encoding):
- Redefined STATE_DIM to 333 to accommodate richer tournament state.
- Updated encode_obs and encode_obs_eval to process the dictionary observation
  provided by the new envs.py (_get_obs_dict).
- Encodes hole cards, community cards, pot, stacks, bets, stage, button/player position, blinds.
- Uses normalization for continuous values.
- NOTE: This requires updating STATE_DIM constants elsewhere and modifying
  the input layer size in models.py. Requires retraining.
"""

import math
import random
from collections import deque
import numpy as np

# --- Constants ---
NUM_PLAYERS = 6 # Should match environment setting
STARTING_STACK = 10000 # Should match environment setting
SUITS = ['H', 'D', 'C', 'S']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
DECK = [r + s for s in SUITS for r in RANKS]
CARD_TO_INDEX = {card: i for i, card in enumerate(DECK)}
INDEX_TO_CARD = {i: card for card, i in CARD_TO_INDEX.items()}
STAGES = ['preflop', 'flop', 'turn', 'river'] # Order matters for one-hot
STAGE_TO_INDEX = {stage: i for i, stage in enumerate(STAGES)}

# --- NEW STATE DIMENSION ---
# 52 (Hole Cards) + 260 (Community Cards) + 1 (Pot) + 6 (Stacks) + 6 (Bets)
# + 4 (Stage) + 1 (Btn Pos) + 1 (Player Pos) + 2 (Blinds) = 333
NEW_STATE_DIM = 333

# Global flag for decision logging.
PRINT_DECISIONS = True


def log_decision(message: str) -> None:
    """
    Logs decision messages if PRINT_DECISIONS is enabled.
    """
    if PRINT_DECISIONS:
        print(message)


def _normalize(value, max_value):
    """ Helper to normalize and clip value between 0 and 1. """
    if max_value is None or max_value == 0:
        return 0.0 # Avoid division by zero
    return max(0.0, min(float(value) / float(max_value), 1.0))


def encode_obs(obs_dict: dict) -> np.ndarray:
    """
    Encodes the observation dictionary (from tournament env) into a state vector.

    Args:
        obs_dict (dict): The observation dictionary from env._get_obs_dict.

    Returns:
        np.ndarray: The encoded state vector (shape=(NEW_STATE_DIM,), dtype=np.float32).
    """
    state = np.zeros(NEW_STATE_DIM, dtype=np.float32)
    current_idx = 0

    # 1. Hole Cards (52 dims)
    hand = obs_dict.get('hand', [])
    for card in hand:
        if card in CARD_TO_INDEX:
            state[CARD_TO_INDEX[card]] = 1.0
    current_idx += 52

    # 2. Community Cards (5 * 52 = 260 dims)
    community = obs_dict.get('community_cards', [])
    for i in range(5):
        if i < len(community):
            card = community[i]
            if card in CARD_TO_INDEX:
                state[current_idx + CARD_TO_INDEX[card]] = 1.0
        current_idx += 52 # Advance index even if card is missing (padding)

    # 3. Pot Size (1 dim, normalized)
    max_pot_estimate = NUM_PLAYERS * STARTING_STACK # Crude max pot estimate
    state[current_idx] = _normalize(obs_dict.get('pot', 0), max_pot_estimate)
    current_idx += 1

    # 4. Stacks (NUM_PLAYERS dims, normalized)
    stacks = obs_dict.get('stacks', {})
    for i in range(NUM_PLAYERS):
        state[current_idx + i] = _normalize(stacks.get(i, 0), STARTING_STACK)
    current_idx += NUM_PLAYERS

    # 5. Current Bets (NUM_PLAYERS dims, normalized)
    # Normalize bets relative to starting stack? Or pot? Use starting stack for now.
    current_bets = obs_dict.get('current_bets', {})
    for i in range(NUM_PLAYERS):
        state[current_idx + i] = _normalize(current_bets.get(i, 0), STARTING_STACK)
    current_idx += NUM_PLAYERS

    # 6. Stage (4 dims, one-hot)
    stage = obs_dict.get('stage', 'preflop').lower()
    if stage in STAGE_TO_INDEX:
        state[current_idx + STAGE_TO_INDEX[stage]] = 1.0
    # If stage is 'showdown' or unknown, leave as zeros
    current_idx += len(STAGES) # Advance by 4

    # 7. Button Position (1 dim, normalized)
    button_pos = obs_dict.get('button_pos', 0)
    state[current_idx] = _normalize(button_pos, NUM_PLAYERS)
    current_idx += 1

    # 8. Player Position relative to Button (1 dim, normalized)
    player_id = obs_dict.get('player_id', 0)
    # Calculate relative position (0=Button, 1=SB, 2=BB, ...) - handle wrap around
    # Position relative to button: (player_id - button_pos + num_players) % num_players
    relative_pos = (player_id - button_pos + NUM_PLAYERS) % NUM_PLAYERS
    state[current_idx] = _normalize(relative_pos, NUM_PLAYERS)
    current_idx += 1

    # 9. Blinds (2 dims, normalized)
    small_blind = obs_dict.get('small_blind', 0)
    big_blind = obs_dict.get('big_blind', 0)
    state[current_idx] = _normalize(small_blind, STARTING_STACK)
    state[current_idx + 1] = _normalize(big_blind, STARTING_STACK)
    current_idx += 2

    # --- Final Check ---
    if current_idx != NEW_STATE_DIM:
        print(f"FATAL ERROR in encode_obs: Final index {current_idx} != NEW_STATE_DIM {NEW_STATE_DIM}")
        # Handle error, maybe return zero vector or raise exception
        return np.zeros(NEW_STATE_DIM, dtype=np.float32)

    return state


# Make encode_obs_eval identical for now, as obs_dict doesn't contain explicit beliefs
def encode_obs_eval(obs_dict: dict) -> np.ndarray:
    """
    Encodes the observation dictionary for evaluation.
    Currently identical to encode_obs.
    """
    return encode_obs(obs_dict)


# --- Epsilon Decay (Unchanged) ---
def epsilon_by_frame(frame_idx: int, epsilon_start: float = 1.0, epsilon_final: float = 0.1, epsilon_decay: float = 200000) -> float: # Increased decay significantly for longer tournament episodes
    """
    Computes the epsilon value for a given frame index using exponential decay.

    Args:
        frame_idx (int): The current frame index (agent steps).
        epsilon_start (float): Starting epsilon value.
        epsilon_final (float): Final epsilon value.
        epsilon_decay (float): Decay rate. Adjusted for potentially longer episodes.

    Returns:
        float: The epsilon value for the current frame.
    """
    if epsilon_decay <= 0: epsilon_decay = 1.0
    epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1.0 * frame_idx / epsilon_decay)
    return epsilon


# --- Replay Buffer (Unchanged - Handles arbitrary state/next_state numpy arrays) ---
class ReplayBuffer:
    """
    ReplayBuffer stores experiences for experience replay during training.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        # Use standard deque from collections
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        """
        Saves an experience tuple.
        Expects state and next_state to be NumPy arrays.
        Expects done to be a boolean or float/int convertible to boolean.

        Args:
            state: The current state (NumPy array).
            action: The action taken (integer index).
            reward: The reward received (float).
            next_state: The next state (NumPy array).
            done: Whether the episode (tournament) terminated or truncated (boolean/numeric).
        """
        # Basic type/shape checks can be added if needed, but rely on caller for now
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        # Store done as float (0.0 or 1.0) for consistency in calculations
        done_float = float(done)

        self.buffer.append((state, action, reward, next_state, done_float))

    def sample(self, batch_size: int):
        """
        Samples a batch of experiences.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones)
                 Returns empty arrays if buffer size is less than batch_size.
        """
        # Ensure batch_size is not larger than the current buffer size
        actual_batch_size = min(batch_size, len(self.buffer))
        if actual_batch_size <= 0:
            # Return empty arrays with correct dimensions if buffer is empty or batch_size is 0
             # Need state shape - assume it's known or get from first element if buffer not empty
             state_shape = (NEW_STATE_DIM,) # Use the new dimension
             # if len(self.buffer) > 0: state_shape = self.buffer[0][0].shape # Get shape from actual data if possible
             return (
                 np.array([], dtype=np.float32).reshape(0, *state_shape),
                 np.array([], dtype=np.int64),
                 np.array([], dtype=np.float32),
                 np.array([], dtype=np.float32).reshape(0, *state_shape),
                 np.array([], dtype=np.float32)
             )

        batch = random.sample(self.buffer, actual_batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert tuples of arrays/values into single NumPy arrays
        try:
            states_np = np.array(states, dtype=np.float32)
            actions_np = np.array(actions, dtype=np.int64) # Actions are usually long integers for indexing
            rewards_np = np.array(rewards, dtype=np.float32)
            next_states_np = np.array(next_states, dtype=np.float32)
            dones_np = np.array(dones, dtype=np.float32) # Use float for calculations (e.g., 1-dones)
        except ValueError as e:
             print(f"Error converting batch to NumPy arrays: {e}")
             # Handle potential shape mismatches if states were not consistent
             # Fallback to returning empty arrays
             state_shape = (NEW_STATE_DIM,)
             return (
                 np.array([], dtype=np.float32).reshape(0, *state_shape),
                 np.array([], dtype=np.int64),
                 np.array([], dtype=np.float32),
                 np.array([], dtype=np.float32).reshape(0, *state_shape),
                 np.array([], dtype=np.float32)
             )


        return (
            states_np,
            actions_np,
            rewards_np,
            next_states_np,
            dones_np
        )

    def __len__(self) -> int:
        """ Returns the current number of experiences in the buffer. """
        return len(self.buffer)

