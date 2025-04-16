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

MODIFIED (Refactoring):
- Added `load_agent_model` function to centralize model loading for inference.
- Added `get_opponent_policy` function to centralize opponent policy creation.

MODIFIED (Circular Import Fix):
- Moved the `from models import BestPokerModel` import inside the functions
  that actually use it (`load_agent_model`, `get_opponent_policy`) to
  resolve a circular dependency between utils.py and models.py.
- **FIXED**: Changed the local imports inside functions to relative: `from .models import BestPokerModel`.

MODIFIED (Constants Centralization):
- Removed local constant definitions (NUM_PLAYERS, NEW_STATE_DIM, etc.).
- Imported constants from Back_End.constants.
"""

import math
import random
from collections import deque
import numpy as np
import torch
import os

# Import constants using relative path
from .constants import (
    NUM_PLAYERS, STARTING_STACK, SUITS, RANKS, DECK, CARD_TO_INDEX,
    INDEX_TO_CARD, STAGES, STAGE_TO_INDEX, NEW_STATE_DIM
)

# Global flag for decision logging.
PRINT_DECISIONS = False


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
    # Ensure value is treated as float for division
    try:
        norm_val = float(value) / float(max_value)
    except (ValueError, TypeError):
        norm_val = 0.0 # Handle cases where value might not be numeric
    return max(0.0, min(norm_val, 1.0))


def encode_obs(obs_dict: dict) -> np.ndarray:
    """
    Encodes the observation dictionary (from tournament env) into a state vector.

    Args:
        obs_dict (dict): The observation dictionary from env._get_obs_dict.

    Returns:
        np.ndarray: The encoded state vector (shape=(NEW_STATE_DIM,), dtype=np.float32).
                    Returns a zero vector if input is not a valid dict.
    """
    # Handle potential invalid input
    if not isinstance(obs_dict, dict) or not obs_dict:
         # print("Warning: encode_obs received invalid obs_dict. Returning zero vector.") # Reduce noise
         return np.zeros(NEW_STATE_DIM, dtype=np.float32)

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
    # Normalize blinds relative to starting stack for consistency
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

# --- Model Loading Utility ---
def load_agent_model(path: str, num_actions: int, device: torch.device = None):
    """
    Loads a BestPokerModel from a checkpoint file for inference.

    Args:
        path (str): Path to the checkpoint file (.pt).
        num_actions (int): Number of actions the model should output.
        device (torch.device, optional): Device to load the model onto ('cpu' or 'cuda').
                                         Defaults to 'cpu' if None.

    Returns:
        BestPokerModel or None: The loaded model in evaluation mode, or None if loading fails.
    """
    # *** Use RELATIVE import here to break circular dependency ***
    try:
        from .models import BestPokerModel # <<< FIX: Changed to relative import
    except ImportError:
        # This error message might now indicate a problem finding models.py relative to utils.py
        print("ERROR in load_agent_model (utils.py): Could not import BestPokerModel from .models.")
        return None

    if device is None:
        device = torch.device('cpu')

    if not path or not os.path.exists(path):
        print(f"Error: Checkpoint file not found: {path}")
        return None
    try:
        # Instantiate the model (ensure models.py defines BestPokerModel correctly)
        model = BestPokerModel(num_actions=num_actions).to(device)

        # Load the checkpoint data
        checkpoint_data = torch.load(path, map_location=device)

        # Extract the state dictionary (handle different saving formats)
        state_dict = None
        if isinstance(checkpoint_data, dict):
            # Common keys used for saving state dictionaries
            state_dict = checkpoint_data.get('agent_state_dict', checkpoint_data.get('state_dict', checkpoint_data))
        else:
            # Assume the checkpoint file *is* the state dictionary
            state_dict = checkpoint_data

        if not state_dict or not isinstance(state_dict, dict):
            raise TypeError("Could not find a valid state_dict in the checkpoint file.")

        # Clean keys (remove 'module.' prefix if saved using DataParallel)
        cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Load the state dict into the model
        # strict=False allows loading even if there are minor mismatches (e.g., missing keys)
        # However, size mismatches (e.g., due to changed NEW_STATE_DIM) will still cause errors here.
        model.load_state_dict(cleaned_state_dict, strict=False)
        model.eval() # Set the model to evaluation mode

        print(f"Successfully loaded model from {path} onto {device}.")
        return model
    except Exception as e:
        print(f"Error: Failed to load checkpoint from {path}: {e}")
        return None


# --- Opponent Policy Creation Utility ---
def get_opponent_policy(opponent_type, agent_model, action_list, num_actions, device=None):
    """
    Creates a policy function for an opponent based on the specified type.
    Moved from main_ui.py / simulate.py for centralization.

    Args:
        opponent_type (str): Type of opponent ('model', 'random', 'variable').
        agent_model (torch.nn.Module or None): The loaded neural network model (for 'model' type).
                                             Should be already loaded and on the correct device.
        action_list (list): List of possible action strings (e.g., ['fold', 'call', ...]).
        num_actions (int): Total number of possible discrete actions.
        device (torch.device, optional): Device the model is on. Defaults to 'cpu'.

    Returns:
        function: A policy function that takes an observation dictionary and returns an action string.
    """
    # *** Use RELATIVE import here to break circular dependency ***
    # Although not strictly necessary here if agent_model is passed in,
    # it's cleaner to keep it consistent with load_agent_model if models.py
    # might be needed for type hinting or other reasons later.
    # If agent_model is guaranteed to be a loaded model, this import isn't needed.
    # Let's keep it for now for robustness.
    try:
        from .models import BestPokerModel # <<< FIX: Changed to relative import
    except ImportError:
        print("ERROR in get_opponent_policy (utils.py): Could not import BestPokerModel from .models.")
        # Fallback to random policy if model import fails
        if opponent_type != "random":
             print("Falling back to random policy due to model import error.")
             return get_opponent_policy("random", None, action_list, num_actions, device)
        # If already random, proceed without model logic

    if device is None:
        device = torch.device('cpu')

    action_index_to_str = {i: s for i, s in enumerate(action_list)}

    if opponent_type == "model":
        if not agent_model:
            print("Warning: 'model' opponent selected but no model provided. Using 'random'.")
            # Recursively call to get the random policy
            return get_opponent_policy("random", None, action_list, num_actions, device)

        # Ensure model is in eval mode (caller should ideally ensure this, but double-check)
        agent_model.eval()

        def policy_fn(obs_dict):
            """ Policy function for a model-based opponent. """
            if not isinstance(obs_dict, dict) or not obs_dict: return 'fold' # Safety check
            legal_actions = obs_dict.get('legal_actions', [])
            if not legal_actions: return 'fold' # No legal actions possible

            try:
                # Encode the observation dictionary into a state vector
                # Use encode_obs_eval for consistency during inference
                state = encode_obs_eval(obs_dict)
            except Exception as e:
                print(f"Error encoding opponent observation: {e}. Folding.")
                return 'fold'

            # Convert state to tensor and get model prediction
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            # Model should already be on the correct device
            with torch.no_grad():
                q_values = agent_model(state_tensor)

            # Process Q-values to select the best legal action
            q_values_np = q_values.squeeze().cpu().numpy()
            sorted_indices = np.argsort(q_values_np)[::-1] # Indices sorted by Q-value descending

            # Iterate through sorted actions and pick the first legal one
            for action_idx in sorted_indices:
                if 0 <= action_idx < num_actions: # Check index validity
                    action_str = action_index_to_str.get(action_idx)
                    if action_str and action_str in legal_actions:
                        return action_str # Return the best legal action

            # Fallback logic if model's preferred actions aren't legal
            if 'check' in legal_actions: return 'check'
            if 'call' in legal_actions: return 'call'
            if 'fold' in legal_actions: return 'fold'
            # Last resort: choose randomly among legal actions if any exist
            return random.choice(legal_actions) if legal_actions else 'fold'

        return policy_fn

    elif opponent_type == "random":
        def policy_fn(obs_dict):
            """ Policy function for a random opponent. """
            if not isinstance(obs_dict, dict) or not obs_dict: return 'fold'
            legal = obs_dict.get("legal_actions", [])
            return random.choice(legal) if legal else "fold"
        return policy_fn

    elif opponent_type == "variable":
        # Create both model and random policies
        model_policy = get_opponent_policy("model", agent_model, action_list, num_actions, device)
        random_policy = get_opponent_policy("random", None, action_list, num_actions, device)
        def policy_fn(obs_dict):
            """ Policy function that randomly chooses between model and random behavior. """
            # 50% chance to use the model policy, 50% chance for random
            return model_policy(obs_dict) if random.random() < 0.5 else random_policy(obs_dict)
        return policy_fn

    else:
        # Default to random if the type is unknown
        print(f"Warning: Unknown opponent type '{opponent_type}'. Using 'random'.")
        return get_opponent_policy("random", None, action_list, num_actions, device)

