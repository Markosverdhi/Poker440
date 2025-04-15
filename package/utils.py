"""
utils.py

This module provides common utilities for the poker RL project. These include:
  - Logging helpers.
  - Observation encoding functions for training and evaluation.
  - An adaptive epsilon decay computation for exploration.
  - A ReplayBuffer class for experience replay.

MODIFIED (Gymnasium Adaptation):
  - Added clipping to the 'pot_value_scalar' in both encode_obs and
    encode_obs_eval to ensure the output stays within the [0, 1] bounds
    defined in the Gymnasium environment's observation_space.
  - Ensured output dtype is consistently np.float32 and shape is (313,).
"""

import math
import random
from collections import deque
import numpy as np
# Removed torch import as it's not used in this file

# Global flag for decision logging.
PRINT_DECISIONS = True


def log_decision(message: str) -> None:
    """
    Logs decision messages if PRINT_DECISIONS is enabled.

    Args:
        message (str): The message to log.
    """
    if PRINT_DECISIONS:
        print(message)


def encode_obs(obs: dict, use_half_encoding: bool = False) -> np.ndarray:
    """
    Encodes the observation into a state vector for the RL agent.

    The encoding consists of:
      - A one-hot encoding of the agent's hand.
      - A normalized pot value computed as log(pot+1) divided by log(player_stack+1),
        CLIPPED to the range [0, 1].
      - One-hot encoded belief states for opponent cards (assumed structure).

    Args:
        obs (dict): The observation dictionary with keys 'hand', 'pot', 'player_stack', and potentially 'beliefs'.
        use_half_encoding (bool): If True, use half-poker encoding (not recommended with current space).

    Returns:
        np.ndarray: The concatenated state vector (shape=(313,), dtype=np.float32).
    """
    # For Gymnasium compatibility, ensure full encoding is used if space is fixed at 313
    if use_half_encoding:
        print("Warning: use_half_encoding=True passed to encode_obs, but observation space expects full encoding (313 dims). Forcing full encoding.")
        # cards = [r + s for s in ['H', 'S'] for r in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']]
        # dim = 26
    # Always use full encoding for consistency with the defined observation space
    cards = [r + s for s in ['H', 'D', 'C', 'S'] for r in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']]
    dim = 52
    num_opponents = 5 # Assuming 6 players total, 5 opponents

    card_to_index = {card: i for i, card in enumerate(cards)}

    # Encode the agent's hand.
    hand_encoding = np.zeros(dim, dtype=np.float32)
    for card in obs.get('hand', []):
        if card in card_to_index:
            hand_encoding[card_to_index[card]] = 1.0

    # Encode the pot value as a normalized ratio, clipped to [0, 1].
    player_stack = obs.get('player_stack', 1.0) # Default to 1 to avoid log(1)=0 denominator issues if stack is 0
    if player_stack <= 0: player_stack = 1.0 # Ensure stack is positive for log
    pot = obs.get('pot', 0)
    # Protect against division by zero when player_stack is 0 or 1 (log(1 or 2))
    denominator = math.log(player_stack + 1.0)
    if denominator <= 1e-6: # Use a small threshold instead of exact zero
        denominator = 1.0 # Avoid division by very small number or zero
    pot_value_scalar = math.log(pot + 1.0) / denominator
    # Clip the value to ensure it stays within the [0, 1] bounds of the observation space
    pot_value_scalar = max(0.0, min(pot_value_scalar, 1.0))
    pot_value = np.array([pot_value_scalar], dtype=np.float32)

    # Encode opponent beliefs (assuming a fixed structure for the observation space).
    # If 'beliefs' are not provided or structure varies, this needs adjustment.
    belief_encoding_list = []
    beliefs = obs.get('beliefs', {}) # Use .get for safety
    # Ensure encoding matches the expected space dimension (num_opponents * dim)
    opponent_ids = sorted(beliefs.keys()) # Get opponent IDs present in beliefs
    encoded_opponents = 0
    for opp_id in opponent_ids:
         if encoded_opponents >= num_opponents: break # Don't exceed expected dimensions
         opp_vector = np.zeros(dim, dtype=np.float32)
         opp_cards = beliefs.get(opp_id, []) # Get cards safely
         for card in opp_cards:
             if card in card_to_index:
                 opp_vector[card_to_index[card]] = 1.0
         belief_encoding_list.append(opp_vector)
         encoded_opponents += 1

    # Pad with zeros if fewer opponents provided than expected
    while encoded_opponents < num_opponents:
         belief_encoding_list.append(np.zeros(dim, dtype=np.float32))
         encoded_opponents += 1

    belief_encoding = np.concatenate(belief_encoding_list) if belief_encoding_list else np.zeros(num_opponents * dim, dtype=np.float32)

    # Concatenate final state vector
    state = np.concatenate([hand_encoding, pot_value, belief_encoding])

    # Final check for shape consistency
    expected_shape = (dim + 1 + num_opponents * dim,)
    if state.shape != expected_shape:
        print(f"Warning: Encoded state shape {state.shape} does not match expected {expected_shape}. Check encoding logic.")
        # Attempt to pad or truncate? For now, just warn. Needs careful handling.
        # Example padding (if too short):
        if state.shape[0] < expected_shape[0]:
             padding = np.zeros(expected_shape[0] - state.shape[0], dtype=np.float32)
             state = np.concatenate([state, padding])
        # Example truncation (if too long):
        elif state.shape[0] > expected_shape[0]:
             state = state[:expected_shape[0]]


    return state.astype(np.float32) # Ensure final dtype


def encode_obs_eval(obs: dict, use_half_encoding: bool = False) -> np.ndarray:
    """
    Encodes the observation into a state vector for evaluation.
    Ensures consistency with the 313-dimensional observation space.

    Args:
        obs (dict): The observation dictionary.
        use_half_encoding (bool): This flag is ignored; evaluation always uses full encoding.

    Returns:
        np.ndarray: The 313-dimensional state vector (dtype=np.float32).
    """
    # Always use full deck for evaluation consistency with the observation space
    deck = [r + s for s in ['H', 'D', 'C', 'S'] for r in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']]
    dim = 52
    num_opponents = 5 # Assuming 6 players total, 5 opponents
    card_to_index = {card: i for i, card in enumerate(deck)}

    hand_encoding = np.zeros(dim, dtype=np.float32)
    for card in obs.get('hand', []):
        if card in card_to_index:
            hand_encoding[card_to_index[card]] = 1.0

    # Encode the pot value as a normalized ratio, clipped to [0, 1].
    player_stack = obs.get('player_stack', 1.0) # Default to 1 to avoid log(1)=0 denominator issues if stack is 0
    if player_stack <= 0: player_stack = 1.0 # Ensure stack is positive for log
    pot = obs.get('pot', 0)
    # Protect against division by zero when player_stack is 0 or 1 (log(1 or 2))
    denominator = math.log(player_stack + 1.0)
    if denominator <= 1e-6: # Use a small threshold instead of exact zero
        denominator = 1.0 # Avoid division by very small number or zero
    pot_value_scalar = math.log(pot + 1.0) / denominator
    # Clip the value to ensure it stays within the [0, 1] bounds of the observation space
    pot_value_scalar = max(0.0, min(pot_value_scalar, 1.0))
    pot_value = np.array([pot_value_scalar], dtype=np.float32)

    # Encode opponent beliefs (assuming fixed structure for evaluation).
    belief_encoding_list = []
    beliefs = obs.get('beliefs', {}) # Use .get for safety
    # Ensure encoding matches the expected space dimension (num_opponents * dim)
    # Loop through expected opponent indices (1 to 5 if agent is 0)
    agent_id = 0 # Assuming agent is 0 for eval context, adjust if needed
    encoded_opponents = 0
    for opp_id in range(num_opponents + 1): # Check all possible IDs
         if opp_id == agent_id: continue # Skip self
         if encoded_opponents >= num_opponents: break # Stop if we have enough

         opp_vector = np.zeros(dim, dtype=np.float32)
         # Check if beliefs for this specific opponent ID are provided
         if opp_id in beliefs:
             opp_cards = beliefs.get(opp_id, []) # Get cards safely
             for card in opp_cards:
                 if card in card_to_index:
                     opp_vector[card_to_index[card]] = 1.0
         belief_encoding_list.append(opp_vector)
         encoded_opponents += 1

    # Pad with zeros if fewer opponents provided than expected
    while encoded_opponents < num_opponents:
         belief_encoding_list.append(np.zeros(dim, dtype=np.float32))
         encoded_opponents += 1

    belief_encoding = np.concatenate(belief_encoding_list) if belief_encoding_list else np.zeros(num_opponents * dim, dtype=np.float32)

    # Concatenate final state vector
    state = np.concatenate([hand_encoding, pot_value, belief_encoding])

    # Final check for shape consistency
    expected_shape = (dim + 1 + num_opponents * dim,)
    if state.shape != expected_shape:
        print(f"Warning: Encoded eval state shape {state.shape} does not match expected {expected_shape}. Check encoding logic.")
        # Attempt to pad or truncate? For now, just warn. Needs careful handling.
        if state.shape[0] < expected_shape[0]:
             padding = np.zeros(expected_shape[0] - state.shape[0], dtype=np.float32)
             state = np.concatenate([state, padding])
        elif state.shape[0] > expected_shape[0]:
             state = state[:expected_shape[0]]

    return state.astype(np.float32) # Ensure final dtype


# --- Epsilon Decay (Unchanged) ---
def epsilon_by_frame(frame_idx: int, epsilon_start: float = 1.0, epsilon_final: float = 0.1, epsilon_decay: float = 5000) -> float:
    """
    Computes the epsilon value for a given frame index using exponential decay.

    Args:
        frame_idx (int): The current frame index.
        epsilon_start (float): Starting epsilon value.
        epsilon_final (float): Final epsilon value.
        epsilon_decay (float): Decay rate (increased from 500 to 2000 for prolonged exploration).

    Returns:
        float: The epsilon value for the current frame.
    """
    # Ensure decay rate is positive to avoid math domain errors or unexpected behavior
    if epsilon_decay <= 0:
        epsilon_decay = 1.0 # Use a default small positive value if invalid input
    # Calculate epsilon using exponential decay formula
    epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1.0 * frame_idx / epsilon_decay)
    return epsilon


# --- Replay Buffer (Unchanged) ---
class ReplayBuffer:
    """
    ReplayBuffer stores experiences for experience replay during training.

    Attributes:
        capacity (int): Maximum number of experiences to store.
        buffer (deque): Internal buffer storing the experiences.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
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
            done: Whether the episode terminated or truncated (boolean/numeric).
        """
        # Ensure states are NumPy arrays (or convertible) - basic check
        # More robust checks could be added if needed
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        # Ensure done is stored consistently, e.g., as float for DQN calculations
        done_float = float(done)

        self.buffer.append((state, action, reward, next_state, done_float))

    def sample(self, batch_size: int):
        """
        Samples a batch of experiences.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones)
        """
        # Ensure batch_size is not larger than the current buffer size
        actual_batch_size = min(batch_size, len(self.buffer))
        if actual_batch_size <= 0:
            return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))

        batch = random.sample(self.buffer, actual_batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert tuples of arrays/values into single NumPy arrays
        states_np = np.array(states, dtype=np.float32)
        actions_np = np.array(actions, dtype=np.int64) # Actions are usually long integers for indexing
        rewards_np = np.array(rewards, dtype=np.float32)
        next_states_np = np.array(next_states, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.float32) # Use float for calculations (e.g., 1-dones)

        return (
            states_np,
            actions_np,
            rewards_np,
            next_states_np,
            dones_np
        )

    def __len__(self) -> int:
        """
        Returns:
            int: The current number of experiences in the buffer.
        """
        return len(self.buffer)
