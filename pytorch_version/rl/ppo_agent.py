import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Import for ONNX and TensorFlow conversion
import onnx
import onnx_tf
import tensorflow as tf

@gin.configurable
class Encoder(nn.Module):
    def __init__(self, observation_spec, fc_layer_params=(200, 100)):
        super(Encoder, self).__init__()
        self._observation_spec = observation_spec
        total_input_dim = 0
        for spec in observation_spec.values():
            total_input_dim += torch.prod(torch.tensor(spec.shape)).item()

        self.network = nn.Sequential(
            nn.Linear(int(total_input_dim), fc_layer_params[0]),
            nn.ReLU(),
            nn.Linear(fc_layer_params[0], fc_layer_params[1]),
            nn.ReLU(),
        )

    def forward(self, observations):
        sorted_keys = sorted(self._observation_spec.keys())
        processed_observations = []
        for key in sorted_keys:
            tensor = observations[key]
            if len(self._observation_spec[key].shape) == 0: # It's a scalar
                processed_observations.append(tensor.unsqueeze(-1))
            else:
                processed_observations.append(tensor)

        concatenated_observations = torch.cat(processed_observations, dim=-1)
        return self.network(concatenated_observations)

@gin.configurable
class ActorNetwork(nn.Module):
    def __init__(self, encoder_output_dim, num_actions):
        super(ActorNetwork, self).__init__()
        self.actor_head = nn.Linear(encoder_output_dim, num_actions)

    def forward(self, state):
        action_logits = self.actor_head(state)
        return action_logits

@gin.configurable
class CriticNetwork(nn.Module):
    def __init__(self, encoder_output_dim):
        super(CriticNetwork, self).__init__()
        self.critic_head = nn.Linear(encoder_output_dim, 1)

    def forward(self, state):
        value = self.critic_head(state)
        return value

@gin.configurable
class PPOAgent(nn.Module):
    def __init__(self, observation_spec, num_actions, actor_fc_layer_params=(200, 100), critic_fc_layer_params=(200, 100)):
        super(PPOAgent, self).__init__()
        self._observation_spec = observation_spec # Store for export

        # Shared Encoder
        self.encoder = Encoder(observation_spec, actor_fc_layer_params) # Using actor_fc_layer_params for encoder
        encoder_output_dim = actor_fc_layer_params[-1]

        self.actor = ActorNetwork(encoder_output_dim, num_actions)
        self.critic = CriticNetwork(encoder_output_dim)

    def forward(self, observations):
        state = self.encoder(observations)
        action_logits = self.actor(state)
        value = self.critic(state)
        return action_logits, value

    def act(self, observations):
        state = self.encoder(observations)
        action_logits = self.actor(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate_actions(self, observations, actions):
        state = self.encoder(observations)
        action_logits = self.actor(state)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.critic(state)
        return log_probs, value, entropy

    def export_to_tf_savedmodel(self, export_path):
        # Create a dummy input for ONNX export
        dummy_input_dict = {}
        for key, spec in self._observation_spec.items():
            # Assuming batch size of 1 for export
            shape = (1,) + tuple(spec.shape) if len(spec.shape) > 0 else (1, 1) # Handle scalar features
            dummy_input_dict[key] = torch.randn(shape, dtype=torch.float32)

        # Define a wrapper for the actor network to handle dictionary input for ONNX export
        class ActorWrapper(nn.Module):
            def __init__(self, encoder_net, actor_head_net, observation_spec):
                super().__init__()
                self.encoder_net = encoder_net
                self.actor_head_net = actor_head_net
                self._observation_spec = observation_spec # For sorted keys

            def forward(self, **kwargs):
                sorted_keys = sorted(self._observation_spec.keys())
                observations = {key: kwargs[key] for key in sorted_keys}
                
                state = self.encoder_net(observations)
                action_logits = self.actor_head_net(state)
                return action_logits

        actor_wrapper = ActorWrapper(self.encoder, self.actor, self._observation_spec)
        actor_wrapper.eval() # Set to eval mode for export

        # Export to ONNX
        onnx_path = export_path + ".onnx"
        input_names = sorted(self._observation_spec.keys())
        output_names = ["action_logits"]

        # Prepare dummy inputs as a tuple of tensors in sorted order
        dummy_inputs_tuple = tuple(dummy_input_dict[key] for key in input_names)

        torch.onnx.export(
            actor_wrapper,
            dummy_inputs_tuple,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=11, # Or a compatible opset version
            do_constant_folding=True,
        )

        # Load ONNX model and convert to TensorFlow SavedModel
        onnx_model = onnx.load(onnx_path)
        tf_rep = onnx_tf.backend.prepare(onnx_model)
        tf_rep.export_graph(export_path)

        print(f"Successfully exported PyTorch ActorNetwork to TensorFlow SavedModel at {export_path}")
