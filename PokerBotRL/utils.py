"""
utils.py

This module provides common utilities for the poker RL project. These include:
  - Logging helpers.
  - Observation encoding functions for training and evaluation.
  - Epsilon decay computation for exploration.
  - A ReplayBuffer class for experience replay.

These utilities are designed to be imported by other modules (e.g., training and simulation scripts).
"""

import math
import random
from collections import deque
import numpy as np
import torch

# Global flag for decision logging.
PRINT_DECISIONS = False


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
      - A log-transformed pot value.
      - One-hot encoded belief states for opponent cards.
    
    Args:
        obs (dict): The observation dictionary with keys 'hand', 'pot', and 'beliefs'.
        use_half_encoding (bool): If True, use half-poker encoding (26 dims per card group); otherwise, use full encoding (52 dims).
    
    Returns:
        np.ndarray: The concatenated state vector.
    """
    if use_half_encoding:
        # Use only two suits.
        cards = [r + s for s in ['H', 'S'] for r in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']]
        dim = 26
    else:
        # Use full 4-suit encoding.
        cards = [r + s for s in ['H', 'D', 'C', 'S'] for r in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']]
        dim = 52

    card_to_index = {card: i for i, card in enumerate(cards)}

    # Encode the agent's hand.
    hand_encoding = np.zeros(dim, dtype=np.float32)
    for card in obs.get('hand', []):
        if card in card_to_index:
            hand_encoding[card_to_index[card]] = 1.0

    # Encode the pot value (log-transformed).
    pot_value = np.array([math.log(obs.get('pot', 0) + 1.0)], dtype=np.float32)

    # Encode opponent beliefs.
    belief_encoding_list = []
    beliefs = obs.get('beliefs', {})
    for opp in sorted(beliefs.keys()):
        opp_vector = np.zeros(dim, dtype=np.float32)
        for card in beliefs[opp]:
            if card in card_to_index:
                opp_vector[card_to_index[card]] = 1.0
        belief_encoding_list.append(opp_vector)
    belief_encoding = np.concatenate(belief_encoding_list) if belief_encoding_list else np.array([], dtype=np.float32)

    state = np.concatenate([hand_encoding, pot_value, belief_encoding])
    return state


def encode_obs_eval(obs: dict, use_half_encoding: bool = False) -> np.ndarray:
    """
    Encodes the observation into a state vector for evaluation.
    
    For evaluation, the encoding is fixed to:
      - 52 dims for the agent's hand.
      - 1 dim for the log-transformed pot.
      - 5 groups of 52 dims each for opponent beliefs (assuming a 6-player game).
    
    Args:
        obs (dict): The observation dictionary.
        use_half_encoding (bool): This flag is maintained for compatibility, but evaluation always uses full encoding.
    
    Returns:
        np.ndarray: The 313-dimensional state vector.
    """
    # Always use full deck for evaluation.
    deck = [r + s for s in ['H', 'D', 'C', 'S'] for r in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']]
    card_to_index = {card: i for i, card in enumerate(deck)}

    hand_encoding = np.zeros(52, dtype=np.float32)
    for card in obs.get('hand', []):
        if card in card_to_index:
            hand_encoding[card_to_index[card]] = 1.0

    pot_value = np.array([math.log(obs.get('pot', 0) + 1.0)], dtype=np.float32)

    belief_encoding_list = []
    beliefs = obs.get('beliefs', {})
    for opp in range(1, 6):  # Assuming opponents with IDs 1 through 5.
        opp_vector = np.zeros(52, dtype=np.float32)
        if opp in beliefs:
            for card in beliefs[opp]:
                if card in card_to_index:
                    opp_vector[card_to_index[card]] = 1.0
        belief_encoding_list.append(opp_vector)
    belief_encoding = np.concatenate(belief_encoding_list)

    state = np.concatenate([hand_encoding, pot_value, belief_encoding])
    return state


def epsilon_by_frame(frame_idx: int, epsilon_start: float = 1.0, epsilon_final: float = 0.1, epsilon_decay: float = 500) -> float:
    """
    Computes the epsilon value for a given frame index using exponential decay.
    
    Args:
        frame_idx (int): The current frame index.
        epsilon_start (float): Starting epsilon value.
        epsilon_final (float): Final epsilon value.
        epsilon_decay (float): Decay rate.
    
    Returns:
        float: The epsilon value for the current frame.
    """
    return epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1.0 * frame_idx / epsilon_decay)


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
        
        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
            done: Whether the episode terminated.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Samples a batch of experiences.
        
        Args:
            batch_size (int): The number of experiences to sample.
        
        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self) -> int:
        """
        Returns:
            int: The current number of experiences in the buffer.
        """
        return len(self.buffer)
