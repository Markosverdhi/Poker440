"""
models.py

This module defines the core neural network architectures and model utilities
for the poker RL agent. It includes:
  - NoisyLinear: A linear layer with learnable noise, used for exploration.
  - ResidualBlock: A residual block to improve gradient flow.
  - BestPokerModel: The dueling DQN architecture for the poker agent.
  - convert_half_to_full_state_dict: Utility function to remap checkpoints
    from half-poker (reduced input) to full-poker dimensions.

These components are intended to be imported and used by training and evaluation scripts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.bias_sigma.data.fill_(self.sigma_init / (self.out_features ** 0.5))

    def reset_noise(self) -> None:
        """Reset the noise for weights and biases."""
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the noisy linear layer.
        
        During training, the output is computed with added noise; during evaluation,
        only the mean parameters are used.
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


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


class BestPokerModel(nn.Module):
    """
    BestPokerModel defines the RL agent architecture used in training.
    
    The network comprises:
      - Two NoisyLinear layers,
      - Two ResidualBlocks,
      - Dueling streams: one for state value and one for action advantages.
    
    The forward pass outputs Q-values for each discrete action.
    """
    def __init__(self, input_dim: int, num_actions: int) -> None:
        super(BestPokerModel, self).__init__()
        self.fc1 = NoisyLinear(input_dim, 256)
        self.fc2 = NoisyLinear(256, 256)
        self.res_block1 = ResidualBlock(256)
        self.res_block2 = ResidualBlock(256)

        # Dueling architecture streams.
        self.value_fc = NoisyLinear(256, 128)
        self.value_out = NoisyLinear(128, 1)
        self.advantage_fc = NoisyLinear(256, 128)
        self.advantage_out = NoisyLinear(128, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that computes Q-values using a dueling architecture.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.res_block1(x)
        x = self.res_block2(x)
        value = F.relu(self.value_fc(x))
        value = self.value_out(value)
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage_out(advantage)
        # Combine value and advantage streams.
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


def convert_half_to_full_state_dict(old_state_dict: dict) -> dict:
    """
    Convert a state dict from a half-poker model (reduced input dimensions) to
    a full-poker model state dict (full input dimensions).
    
    The conversion re-maps the weights of the first fully connected layer.
    
    Expected dimensions:
      - Half-poker fc1.weight: (out_features, 157)
      - Full-poker fc1.weight: (out_features, 313)
    
    Returns:
        new_state_dict (dict): A state dict compatible with the full-poker model.
    """
    new_state_dict = {}
    # Remap the first layer's weights.
    old_w = old_state_dict["fc1.weight"]  # shape: (out_features, 157)
    new_w = torch.zeros((old_w.shape[0], 313), dtype=old_w.dtype)

    # Map agent's hand: first 13 dimensions for hearts, next 13 (half-poker) map to spades (full indices 39-51).
    new_w[:, 0:13] = old_w[:, 0:13]
    new_w[:, 39:52] = old_w[:, 13:26]

    # Map pot value from half index 26 to full index 52.
    new_w[:, 52] = old_w[:, 26]

    # Map opponent belief encodings: there are 5 groups in the full model.
    for k in range(5):
        start_half = 27 + 26 * k
        start_full = 53 + 52 * k
        # Map hearts.
        new_w[:, start_full : start_full + 13] = old_w[:, start_half : start_half + 13]
        # Map spades.
        new_w[:, start_full + 39 : start_full + 52] = old_w[:, start_half + 13 : start_half + 26]

    new_state_dict["fc1.weight"] = new_w
    new_state_dict["fc1.bias"] = old_state_dict["fc1.bias"]

    # Copy remaining parameters unmodified.
    for key in old_state_dict:
        if key not in ["fc1.weight", "fc1.bias"]:
            new_state_dict[key] = old_state_dict[key]

    return new_state_dict
