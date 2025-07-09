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
"""Actor network for Register Allocation in PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RegAllocNetwork(nn.Module):
    """Creates the actor network for register allocation policy training."""

    def __init__(self, input_shape, num_actions, fc_layer_params=(200, 100), dropout_layer_params=(0.0, 0.0)):
        """Creates an instance of `RegAllocNetwork`.

        Args:
            input_shape: A tuple representing the input shape.
            num_actions: The number of possible actions.
            fc_layer_params: Optional list of fully_connected parameters, where each
                item is the number of units in the layer.
            dropout_layer_params: Optional list of dropout layer parameters.
        """
        super(RegAllocNetwork, self).__init__()

        layers = []
        for i, units in enumerate(fc_layer_params):
            if i == 0:
                layers.append(nn.Linear(input_shape[-1], units))
            else:
                layers.append(nn.Linear(fc_layer_params[i-1], units))
            layers.append(nn.ReLU())
            if dropout_layer_params and i < len(dropout_layer_params) and dropout_layer_params[i] > 0.0:
                layers.append(nn.Dropout(p=dropout_layer_params[i]))
        
        self.encoder = nn.Sequential(*layers)

        self.projection_network = nn.Linear(fc_layer_params[-1], num_actions)

    def forward(self, observations, mask=None):
        """Forward pass of the network.

        Args:
            observations: The input observations.
            mask: An optional mask to apply to the output logits.

        Returns:
            The output logits from the network.
        """
        # Pass the observations through the encoder
        state = self.encoder(observations)

        # Get the logits from the projection network
        logits = self.projection_network(state)

        # Apply the mask if provided
        if mask is not None:
            logits = logits.masked_fill(mask, -float('inf'))

        return logits

if __name__ == '__main__':
    # Example usage:
    # The input shape and number of actions are based on the original code's
    # context for register allocation.
    # Input shape: (batch_size, sequence_length, feature_dim)
    # Number of actions: 33 (representing registers)
    input_shape = (1, 1, 10)  # Example input shape
    num_actions = 33

    # Create the network
    network = RegAllocNetwork(input_shape=input_shape, num_actions=num_actions)

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
