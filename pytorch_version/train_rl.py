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
"""Train behavioral cloning policy in PyTorch."""

import argparse
import os
import time
import gin
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Temporarily suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from pytorch_version import specs

from pytorch_version.rl.ppo_agent import PPOAgent

# Assuming SchedNetwork and RegAllocNetwork are still needed for their specs
# from pytorch_version.sched.sched_network import SchedNetwork
# from pytorch_version.regalloc.regalloc_network import RegAllocNetwork

class TFRecordDataset(Dataset):
    """A PyTorch Dataset for reading TFRecord files for RL training."""

    def __init__(self, data_path, time_step_spec, action_spec, batch_size, sequence_length):
        self.data_path = data_path
        self.time_step_spec = time_step_spec
        self.action_spec = action_spec
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        
        self._create_iterator()

    def _create_iterator(self):
        """Creates a TensorFlow dataset iterator."""
        parser_fn = self._create_parser_fn()
        
        dataset = (
            tf.data.TFRecordDataset(self.data_path)
            .filter(lambda string: tf.strings.length(string) > 0)
            .map(parser_fn)
            .unbatch()
            .batch(self.sequence_length, drop_remainder=True)
            .batch(self.batch_size, drop_remainder=True)
        )
        self.iterator = iter(dataset)

    def _create_parser_fn(self):
        """Creates a parser function for the TFRecord data."""
        def _parser_fn(serialized_proto):
            context_features = {}
            sequence_features = {
                tensor_spec.name: tf.io.FixedLenSequenceFeature(
                    shape=tensor_spec.shape, dtype=tensor_spec.dtype
                )
                for tensor_spec in self.time_step_spec['observation'].values()
            }
            sequence_features[self.action_spec.name] = tf.io.FixedLenSequenceFeature(
                shape=self.action_spec.shape, dtype=self.action_spec.dtype
            )
            # Add reward parsing
            sequence_features['reward'] = tf.io.FixedLenSequenceFeature(
                shape=self.time_step_spec['reward'].shape, dtype=self.time_step_spec['reward'].dtype
            )

            _, parsed_sequence = tf.io.parse_single_sequence_example(
                serialized_proto,
                context_features=context_features,
                sequence_features=sequence_features,
            )
            
            observation = parsed_sequence
            action = parsed_sequence.pop(self.action_spec.name)
            reward = parsed_sequence.pop('reward') # Extract reward
            
            return {'observation': observation, 'action': action, 'reward': reward}

        return _parser_fn

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.iterator)
            # Convert TensorFlow tensors to NumPy arrays
            obs_numpy = {k: v.numpy() for k, v in data['observation'].items()}
            action_numpy = data['action'].numpy()
            reward_numpy = data['reward'].numpy() # Convert reward to NumPy
            return obs_numpy, action_numpy, reward_numpy
        except StopIteration:
            # Restart the iterator for the next epoch
            self._create_iterator()
            raise StopIteration

    def __len__(self):
        return 0self):
        return 0

@gin.configurable
def train(
    problem_name,
    data_path,
    model_dir,
    num_epochs,
    batch_size,
    sequence_length,
    learning_rate,
    log_interval,
    save_interval,
    # PPO specific hyperparameters
    ppo_epochs=4, # Number of epochs to train on the collected data
    clip_epsilon=0.2, # PPO clip ratio
    gae_lambda=0.95, # GAE lambda parameter
    value_coeff=0.5, # Coefficient for value loss
    entropy_coeff=0.01, # Coefficient for entropy loss
    gamma=0.99, # Discount factor
):
    """Main training loop."""
    if problem_name == 'regalloc':
        time_step_spec, action_spec = specs.get_regalloc_signature_spec()
        num_actions = 33
    elif problem_name == 'sched':
        time_step_spec, action_spec = specs.get_sched_signature_spec()
        num_actions = 256
    else:
        raise ValueError(f"Unknown problem: {problem_name}")

    dataset = TFRecordDataset(
        data_path,
        time_step_spec,
        action_spec,
        batch_size,
        sequence_length,
    )
    
    agent = PPOAgent(
        observation_spec=time_step_spec['observation'],
        num_actions=num_actions,
    )
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)

    print(f"Starting training for {problem_name} on {device}...")

    for epoch in range(num_epochs):
        start_time = time.time()
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0
        total_loss = 0
        num_batches = 0

        for observations, actions, rewards in dataset:
            # Convert all observation features to PyTorch tensors and move to device
            processed_observations = {}
            for key, value in observations.items():
                tensor = torch.from_numpy(value).float().to(device)
                if len(tensor.shape) > 2: # If it has sequence_length dimension (e.g., (B, S, F))
                    processed_observations[key] = tensor.view(-1, *tensor.shape[2:])
                else: # Scalar features will be (B, S) after TFRecordDataset
                    processed_observations[key] = tensor.view(-1)

            actions = torch.from_numpy(actions).long().to(device).view(-1)
            rewards = torch.from_numpy(rewards).float().to(device).view(-1)

            # Calculate returns (GAE is not implemented here for simplicity, using simple returns)
            # For full PPO, GAE would be calculated over trajectories.
            # Here, we assume rewards are already aligned with actions for each step.
            returns = torch.zeros_like(rewards)
            R = 0
            for t in reversed(range(rewards.shape[0])):
                R = rewards[t] + gamma * R
                returns[t] = R
            
            # Normalize returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Get current policy's log_probs and values
            log_probs, values, entropy = agent.evaluate_actions(processed_observations, actions)

            # Calculate advantage
            advantages = returns - values.detach()

            # PPO Loss
            # For simplicity, we're not using an 'old_policy' here, which is crucial for PPO.
            # A full PPO implementation would involve collecting data with an old policy,
            # then updating the current policy for several epochs, and then making the current
            # policy the old policy for the next data collection phase.
            # This current setup is more like A2C with a clipping mechanism.
            
            # Policy Loss (Actor Loss)
            ratio = torch.exp(log_probs - log_probs.detach()) # log_probs.detach() acts as old_log_probs
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value Loss (Critic Loss)
            critic_loss = F.mse_loss(values.squeeze(-1), returns)

            # Total Loss
            loss = actor_loss + value_coeff * critic_loss - entropy_coeff * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy_loss += entropy.mean().item()
            total_loss += loss.item()
            num_batches += 1

            if num_batches % log_interval == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {num_batches}, "
                      f"Total Loss: {loss.item():.4f}, Actor Loss: {actor_loss.item():.4f}, "
                      f"Critic Loss: {critic_loss.item():.4f}, Entropy: {entropy.mean().item():.4f}")

        end_time = time.time()
        epoch_duration = end_time - start_time
        avg_total_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_actor_loss = total_actor_loss / num_batches if num_batches > 0 else 0
        avg_critic_loss = total_critic_loss / num_batches if num_batches > 0 else 0
        avg_entropy_loss = total_entropy_loss / num_batches if num_batches > 0 else 0

        print(f"Epoch {epoch+1} completed in {epoch_duration:.2f}s. "
              f"Average Total Loss: {avg_total_loss:.4f}, Average Actor Loss: {avg_actor_loss:.4f}, "
              f"Average Critic Loss: {avg_critic_loss:.4f}, Average Entropy: {avg_entropy_loss:.4f}")

        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(model_dir, f"ppo_agent_epoch_{epoch+1}.pt")
            torch.save(agent.state_dict(), checkpoint_path)
            print(f"Saved PPO agent checkpoint to {checkpoint_path}")

    print("Training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a behavioral cloning model.')
    parser.add_argument('--problem', type=str, required=True, help='The problem to train for (e.g., regalloc, sched).')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the TFRecord training data.')
    parser.add_argument('--model_dir', type=str, default='pytorch_models', help='Directory to save the trained models.')
    parser.add_argument('--gin_files', nargs='+', help='List of paths to gin configuration files.')
    parser.add_argument('--gin_bindings', nargs='+', help='Gin bindings to override the values set in the config files.')
    args = parser.parse_args()

    gin.parse_config_files_and_bindings(args.gin_files, args.gin_bindings)

    os.makedirs(args.model_dir, exist_ok=True)
    train(problem_name=args.problem, data_path=args.data_path, model_dir=args.model_dir)
