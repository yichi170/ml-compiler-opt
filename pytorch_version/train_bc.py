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

from pytorch_version.regalloc_network import RegAllocNetwork
from pytorch_version.sched.sched_network import SchedNetwork

class TFRecordDataset(Dataset):
    """A PyTorch Dataset for reading TFRecord files."""

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

            _, parsed_sequence = tf.io.parse_single_sequence_example(
                serialized_proto,
                context_features=context_features,
                sequence_features=sequence_features,
            )
            
            observation = parsed_sequence
            action = parsed_sequence.pop(self.action_spec.name)
            
            return {'observation': observation, 'action': action}

        return _parser_fn

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.iterator)
            # Convert TensorFlow tensors to NumPy arrays
            obs_numpy = {k: v.numpy() for k, v in data['observation'].items()}
            action_numpy = data['action'].numpy()
            return obs_numpy, action_numpy
        except StopIteration:
            # Restart the iterator for the next epoch
            self._create_iterator()
            raise StopIteration

    def __len__(self):
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
):
    """Main training loop."""
    if problem_name == 'regalloc':
        time_step_spec, action_spec = specs.get_regalloc_signature_spec()
        model_class = RegAllocNetwork
        num_actions = 33
        input_feature_name = 'node_features'
    elif problem_name == 'sched':
        time_step_spec, action_spec = specs.get_sched_signature_spec()
        model_class = SchedNetwork
        num_actions = 256
        input_feature_name = 'pos'
    else:
        raise ValueError(f"Unknown problem: {problem_name}")

    dataset = TFRecordDataset(
        data_path,
        time_step_spec,
        action_spec,
        batch_size,
        sequence_length,
    )
    
    input_shape = time_step_spec['observation'][input_feature_name].shape

    model = model_class(
        input_shape=input_shape,
        num_actions=num_actions,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Starting training for {problem_name} on {device}...")

    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0
        num_batches = 0

        for observations, actions in dataset:
            inputs = torch.from_numpy(observations[input_feature_name]).float().to(device)
            if inputs.dim() == 4 and inputs.shape[2] == 1:
                inputs = inputs.squeeze(2)
            if inputs.dim() == 3 and inputs.shape[2] == 1:
                inputs = inputs.squeeze(2)

            targets = torch.from_numpy(actions).long().to(device)
            if targets.dim() == 3 and targets.shape[2] == 1:
                 targets = targets.squeeze(2)
            
            logits = model(inputs)
            
            logits_reshaped = logits.view(-1, num_actions)
            targets_reshaped = targets.view(-1)

            optimizer.zero_grad()
            loss = loss_fn(logits_reshaped, targets_reshaped)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % log_interval == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {num_batches}, Loss: {loss.item():.4f}")

        end_time = time.time()
        epoch_duration = end_time - start_time
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1} completed in {epoch_duration:.2f}s. Average loss: {avg_loss:.4f}")

        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(model_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")

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
