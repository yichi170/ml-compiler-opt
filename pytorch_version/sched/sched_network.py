# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Actor network for Instruction Scheduling in PyTorch."""

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F

@gin.configurable
class SchedNetwork(nn.Module):
    """Creates the actor network for instruction scheduling policy training."""

    def __init__(self, observation_spec, num_actions, fc_layer_params=(200, 100)):
        """Creates an instance of `SchedNetwork`.

        Args:
            observation_spec: A dictionary of TensorSpecs defining the observation space.
            num_actions: The number of possible actions.
            fc_layer_params: Optional list of fully_connected parameters, where each
                item is the number of units in the layer.
        """
        super(SchedNetwork, self).__init__()

        self._observation_spec = observation_spec
        self._is_scalar = {k: (len(spec.shape) == 0) for k, spec in observation_spec.items()}

        total_input_dim = 0
        for spec in observation_spec.values():
            total_input_dim += torch.prod(torch.tensor(spec.shape)).item()

        self.encoder = nn.Sequential(
            nn.Linear(int(total_input_dim), fc_layer_params[0]),
            nn.ReLU(),
            nn.Linear(fc_layer_params[0], fc_layer_params[1]),
            nn.ReLU(),
        )

        self.projection_network = nn.Linear(fc_layer_params[1], num_actions)

    def forward(self, observations, mask=None):
        """Forward pass of the network.

        Args:
            observations: A dictionary of input observations.
            mask: An optional mask to apply to the output logits.

        Returns:
            The output logits from the network.
        """
        # Concatenate all observation features
        # Ensure consistent order by sorting keys
        sorted_keys = sorted(self._observation_spec.keys())

        # Handle batch dimension for scalar features
        # If a feature is scalar (shape=()), it will be (batch_size,) after from_numpy
        # We need to unsqueeze it to (batch_size, 1) to concatenate correctly
        processed_observations = []
        for key in sorted_keys:
            tensor = observations[key]
            if self._is_scalar[key]: # It's a scalar
                processed_observations.append(tensor.unsqueeze(-1))
            else:
                processed_observations.append(tensor)

        concatenated_observations = torch.cat(processed_observations, dim=-1)

        # Pass the concatenated observations through the encoder
        state = self.encoder(concatenated_observations)

        # Get the logits from the projection network
        logits = self.projection_network(state)

        # Apply the mask if provided
        if mask is not None:
            logits = logits.masked_fill(mask, -float('inf'))

        return logits

if __name__ == '__main__':
    # Example usage:
    # The input shape and number of actions are based on the original code's
    # context for instruction scheduling.
    # Input shape: (batch_size, sequence_length, feature_dim)
    # Number of actions: 128 (representing the scheduling window)
    input_shape = (1, 1, 10)  # Example input shape
    num_actions = 128

    # Create the network
    network = SchedNetwork(input_shape=input_shape, num_actions=num_actions)

    # Create a dummy input tensor
    dummy_input = torch.randn(*input_shape)

    # Create a dummy mask
    dummy_mask = torch.zeros(1, 1, num_actions).bool()
    dummy_mask[:, :, 5] = 1  # Mask action 5

    # Forward pass
    logits = network(dummy_input, mask=dummy_mask)

    print("Input shape:", dummy_input.shape)
    print("Output logits shape:", logits.shape)
    print("Output logits:", logits)
