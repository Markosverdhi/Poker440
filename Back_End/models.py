# filename: models.py
"""
models.py

This module defines the core neural network architectures and model utilities
for the poker RL agent. It includes:
  - NoisyLinear: A linear layer with learnable noise, used for exploration.
  - ResidualBlock: A residual block to improve gradient flow.
  - BestPokerModel: The dueling DQN architecture for the poker agent.
  - convert_half_to_full_state_dict: Utility function to remap checkpoints
    from half-poker (reduced input) to full-poker dimensions (Likely obsolete now).

MODIFIED:
  - Imported NEW_STATE_DIM from utils.py.
  - Updated the input dimension of the first layer (fc1) in BestPokerModel
    to use NEW_STATE_DIM instead of the previous hardcoded or passed input_dim.
    This is CRITICAL for compatibility with the new tournament state encoding.
  - Requires retraining the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import the new state dimension definition

from .constants import NEW_STATE_DIM

# ---------------------------------------------------------------------------
# NoisyLinear Layer (Unchanged)
# ---------------------------------------------------------------------------
class NoisyLinear(nn.Module):
    """
    NoisyLinear implements a linear transformation with added learnable noise.
    This layer is useful for exploration in reinforcement learning.
    """
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017) -> None:
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters for mean and standard deviation of weights and biases.
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """Initialize weight and bias parameters."""
        mu_range = 1 / (self.in_features ** 0.5)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / (self.in_features ** 0.5))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / (self.out_features ** 0.5)) # Corrected denominator based on original code

    def reset_noise(self) -> None:
        """Reset the noise for weights and biases."""
        # Factorised Gaussian noise
        epsilon_in = torch.randn(self.in_features)
        epsilon_out = torch.randn(self.out_features)
        epsilon_in = epsilon_in.sign() * epsilon_in.abs().sqrt()
        epsilon_out = epsilon_out.sign() * epsilon_out.abs().sqrt()
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the noisy linear layer.

        During training, the output is computed with added noise; during evaluation,
        only the mean parameters are used.
        """
        if self.training:
            # Use the stored noise buffers during training
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Use only the mean parameters during evaluation
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

# ---------------------------------------------------------------------------
# ResidualBlock (Unchanged)
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """
    ResidualBlock implements a basic residual connection with two linear layers.
    It is used to improve gradient flow and model capacity.
    """
    def __init__(self, dim: int) -> None:
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection and layer normalization.
        """
        residual = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out = out + residual
        out = self.layer_norm(out)
        return F.relu(out)

# ---------------------------------------------------------------------------
# BestPokerModel (Dueling DQN Architecture) - MODIFIED INPUT DIM
# ---------------------------------------------------------------------------
class BestPokerModel(nn.Module):
    """
    BestPokerModel defines the RL agent architecture used in training.

    The network comprises:
      - Two NoisyLinear layers,
      - Three ResidualBlocks,
      - Dueling streams: one for state value and one for action advantages.

    MODIFIED: The input dimension for fc1 now uses NEW_STATE_DIM from utils.py
              to match the updated tournament state encoding.
    """
    # Removed input_dim from __init__ args as it's now fixed by NEW_STATE_DIM
    def __init__(self, num_actions: int) -> None:
        super(BestPokerModel, self).__init__()

        # Use the imported NEW_STATE_DIM for the first layer's input
        self.input_dim = NEW_STATE_DIM
        self.num_actions = num_actions
        hidden_dim = 256 # Hidden dimension size

        # --- Network Layers ---
        # First layer now takes the new state dimension
        self.fc1 = NoisyLinear(self.input_dim, hidden_dim)
        self.fc2 = NoisyLinear(hidden_dim, hidden_dim)
        self.res_block1 = ResidualBlock(hidden_dim)
        self.res_block2 = ResidualBlock(hidden_dim)
        # Optional extra residual block from original code
        self.res_block3 = ResidualBlock(hidden_dim)

        # --- Dueling Architecture Streams ---
        # Value stream
        self.value_fc = NoisyLinear(hidden_dim, hidden_dim // 2) # Example: 128
        self.value_out = NoisyLinear(hidden_dim // 2, 1)

        # Advantage stream
        self.advantage_fc = NoisyLinear(hidden_dim, hidden_dim // 2) # Example: 128
        self.advantage_out = NoisyLinear(hidden_dim // 2, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that computes Q-values using a dueling architecture.
        """
        # Initial layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x) # Use the third block

        # Value stream
        value = F.relu(self.value_fc(x))
        value = self.value_out(value) # Shape: (batch_size, 1)

        # Advantage stream
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage_out(advantage) # Shape: (batch_size, num_actions)

        # Combine value and advantage streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values

    def reset_noise(self) -> None:
        """
        Reset noise parameters for all noisy layers to ensure fresh exploration.
        """
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.value_fc.reset_noise()
        self.value_out.reset_noise()
        self.advantage_fc.reset_noise()
        self.advantage_out.reset_noise()

# ---------------------------------------------------------------------------
# Checkpoint Conversion Utility (Likely Obsolete)
# ---------------------------------------------------------------------------
def convert_half_to_full_state_dict(old_state_dict: dict) -> dict:
    """
    Convert a state dict from a half-poker model (reduced input dimensions, 157) to
    a full-poker model state dict (original full input dimensions, 313).

    NOTE: This function is likely obsolete now that the state representation
    has changed significantly to NEW_STATE_DIM (e.g., 333). It is kept here
    for potential backward compatibility reference but should not be used
    with models trained on the new state dimension.

    Expected dimensions (OLD):
      - Half-poker fc1.weight: (out_features, 157)
      - Full-poker fc1.weight: (out_features, 313)

    Returns:
        new_state_dict (dict): A state dict compatible with the *original* full-poker model (313 dims).
    """
    print("Warning: convert_half_to_full_state_dict is likely obsolete due to new state representation.")
    new_state_dict = {}
    OLD_HALF_DIM = 157
    OLD_FULL_DIM = 313

    # Check if the input state dict matches the expected half-poker dimension
    if "fc1.weight" not in old_state_dict or old_state_dict["fc1.weight"].shape[1] != OLD_HALF_DIM:
        print(f"Error: Input state dict fc1.weight shape is not the expected half-poker dim ({OLD_HALF_DIM}). Cannot convert.")
        return old_state_dict # Return original dict if conversion cannot proceed

    # Remap the first layer's weights.
    old_w = old_state_dict["fc1.weight"]  # shape: (out_features, 157)
    new_w = torch.zeros((old_w.shape[0], OLD_FULL_DIM), dtype=old_w.dtype)

    # Map agent's hand: first 13 dimensions for hearts, next 13 (half-poker) map to spades (full indices 39-51).
    new_w[:, 0:13] = old_w[:, 0:13]    # Hearts
    new_w[:, 39:52] = old_w[:, 13:26]   # Spades

    # Map pot value from half index 26 to full index 52.
    new_w[:, 52] = old_w[:, 26]

    # Map opponent belief encodings: there are 5 groups in the full model.
    for k in range(5): # 5 opponents
        start_half = 27 + 26 * k # Start index in half-poker encoding (H, S only)
        start_full = 53 + 52 * k # Start index in full-poker encoding (H, D, C, S)
        # Map hearts.
        new_w[:, start_full : start_full + 13] = old_w[:, start_half : start_half + 13]
        # Map spades.
        new_w[:, start_full + 39 : start_full + 52] = old_w[:, start_half + 13 : start_half + 26]

    new_state_dict["fc1.weight"] = new_w
    # Handle bias and other parameters
    if "fc1.bias" in old_state_dict:
        new_state_dict["fc1.bias"] = old_state_dict["fc1.bias"]
    # Handle potential noisy layer parameters if they exist in old dict
    if "fc1.weight_mu" in old_state_dict: # Check one example param
         # Assume structure matches, remap weight_mu like weight
         old_w_mu = old_state_dict["fc1.weight_mu"]
         new_w_mu = torch.zeros((old_w_mu.shape[0], OLD_FULL_DIM), dtype=old_w_mu.dtype)
         new_w_mu[:, 0:13] = old_w_mu[:, 0:13]; new_w_mu[:, 39:52] = old_w_mu[:, 13:26]
         new_w_mu[:, 52] = old_w_mu[:, 26]
         for k in range(5):
              start_half = 27 + 26 * k; start_full = 53 + 52 * k
              new_w_mu[:, start_full : start_full + 13] = old_w_mu[:, start_half : start_half + 13]
              new_w_mu[:, start_full + 39 : start_full + 52] = old_w_mu[:, start_half + 13 : start_half + 26]
         new_state_dict["fc1.weight_mu"] = new_w_mu
         # Copy other noisy params assuming they don't need reshaping (bias, sigma)
         for p in ["weight_sigma", "bias_mu", "bias_sigma"]:
              key = f"fc1.{p}";
              if key in old_state_dict: new_state_dict[key] = old_state_dict[key]


    # Copy remaining parameters unmodified.
    for key in old_state_dict:
        if not key.startswith("fc1."): # Copy params from other layers
            new_state_dict[key] = old_state_dict[key]

    return new_state_dict

