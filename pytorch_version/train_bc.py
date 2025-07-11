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
import torch
import torch.nn as nn

import pytorch_version.data as torchdata
from pytorch_version import specs
from pytorch_version.sched.sched_network import SchedNetwork

@gin.configurable
def train(
    problem_name,
    data_path,
    model_dir,
    num_epochs=10,
    batch_size=64,
    sequence_length=1,
    learning_rate=0.001,
    log_interval=10,
    save_interval=1,
):
    """Main training loop."""
    if problem_name == 'sched':
        time_step_spec, action_spec = specs.get_sched_signature_spec()
        model_class = SchedNetwork
        num_actions = 256
    else:
        raise ValueError(f"Unknown problem: {problem_name}")

    dataset_cache_path = os.path.join(model_dir, f"{problem_name}_dataset.pt")

    if os.path.exists(dataset_cache_path):
        print(f"Loading dataset from cache: {dataset_cache_path}")
        observations, actions = torchdata.load_dataset(dataset_cache_path)
    else:
        print(f"Parsing TFRecords from: {data_path}")
        observations, actions = torchdata.parse_tfrecord_file(
            data_path, time_step_spec, action_spec, sequence_length, batch_size
        )
        torchdata.save_dataset(observations, actions, dataset_cache_path)
        print(f"Saved parsed dataset to {dataset_cache_path}")
    torch_dataset = torchdata.TorchDataset(observations, actions)
    dataloader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)

    model = model_class(
        observation_spec=time_step_spec['observation'],
        num_actions=num_actions,
        fc_layer_params=(256, 128),
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

        for observations, actions in dataloader:
            processed_observations = {}
            for key, tensor in observations.items():
                tensor = tensor.to(device)

                # Reshape from (batch_size, sequence_length, ...) to (batch_size * sequence_length, ...)
                # For scalar features, this will be (batch_size, sequence_length) -> (batch_size * sequence_length)
                # For (256,) features, this will be (batch_size, sequence_length, 256) -> (batch_size * sequence_length, 256)
                if len(tensor.shape) > 2: # If it has sequence_length dimension (e.g., (B, S, F))
                    processed_observations[key] = tensor.view(-1, *tensor.shape[2:])
                else: # Scalar features will be (B, S) after TFRecordDataset
                    processed_observations[key] = tensor.view(-1)

            # Reshape targets from (batch_size, sequence_length) to (batch_size * sequence_length)
            targets = actions.to(device).view(-1)

            logits = model(processed_observations)
            
            # Reshape logits and targets for CrossEntropyLoss
            # logits will be (batch_size * sequence_length, num_actions)
            # targets will be (batch_size * sequence_length)
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
    parser.add_argument('--model_dir', type=str, default='models_and_dataset', help='Directory to save the trained models.')
    parser.add_argument('--gin_files', nargs='+', help='List of paths to gin configuration files.')
    parser.add_argument('--gin_bindings', nargs='+', help='Gin bindings to override the values set in the config files.')
    args = parser.parse_args()

    gin.parse_config_files_and_bindings(args.gin_files, args.gin_bindings)

    os.makedirs(args.model_dir, exist_ok=True)
    train(problem_name=args.problem, data_path=args.data_path, model_dir=args.model_dir)
