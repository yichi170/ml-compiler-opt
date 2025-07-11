import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset

def parse_tfrecord_file(data_path, time_step_spec, action_spec, sequence_length, batch_size):
    """Reads and parses TFRecord data into PyTorch-ready NumPy arrays."""
    def _parser_fn(serialized_proto):
        context_features = {}
        sequence_features = {
            tensor_spec.name: tf.io.FixedLenSequenceFeature(
                shape=tensor_spec.shape, dtype=tensor_spec.dtype
            )
            for tensor_spec in time_step_spec['observation'].values()
        }
        sequence_features[action_spec.name] = tf.io.FixedLenSequenceFeature(
            shape=action_spec.shape, dtype=action_spec.dtype
        )

        _, parsed_sequence = tf.io.parse_single_sequence_example(
            serialized_proto,
            context_features=context_features,
            sequence_features=sequence_features,
        )

        observation = parsed_sequence
        action = parsed_sequence.pop(action_spec.name)
        return {'observation': observation, 'action': action}

    dataset = (
        tf.data.TFRecordDataset(data_path)
        .filter(lambda string: tf.strings.length(string) > 0)
        .map(_parser_fn)
        .unbatch()
        .batch(sequence_length, drop_remainder=True)
        .batch(batch_size, drop_remainder=True)
    )

    observations_list = []
    actions_list = []

    for item in dataset:
        obs = {k: v.numpy() for k, v in item['observation'].items()}
        actions = item['action'].numpy()
        observations_list.append(obs)
        actions_list.append(actions)

    # Merge observations across batches
    combined_obs = {}
    for key in observations_list[0]:
        combined_obs[key] = np.concatenate([obs[key] for obs in observations_list], axis=0)

    combined_actions = np.concatenate(actions_list, axis=0)
    return combined_obs, combined_actions


def save_dataset(observations, actions, save_path):
    """Save parsed dataset as a .pt file."""
    torch.save({'observations': observations, 'actions': actions}, save_path)

def load_dataset(save_path):
    """Load a previously saved PyTorch dataset from a .pt file."""
    data = torch.load(save_path, weights_only=False)
    return data['observations'], data['actions']


class TorchDataset(Dataset):
    def __init__(self, observations: dict, actions: np.ndarray):
        self.observations = {
            k: torch.tensor(v, dtype=torch.float32) for k, v in observations.items()
        }
        self.actions = torch.tensor(actions, dtype=torch.long)
        self.length = self.actions.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        obs = {k: v[idx] for k, v in self.observations.items()}
        return obs, self.actions[idx]

#class TFRecordDataset(Dataset):
#    """A PyTorch Dataset for reading TFRecord files."""
#
#    def __init__(self, data_path, time_step_spec, action_spec, batch_size, sequence_length):
#        self.data_path = data_path
#        self.time_step_spec = time_step_spec
#        self.action_spec = action_spec
#        self.batch_size = batch_size
#        self.sequence_length = sequence_length
#        
#        self._create_iterator()
#
#    def _create_iterator(self):
#        """Creates a TensorFlow dataset iterator."""
#        parser_fn = self._create_parser_fn()
#        
#        dataset = (
#            tf.data.TFRecordDataset(self.data_path)
#            .filter(lambda string: tf.strings.length(string) > 0)
#            .map(parser_fn)
#            .unbatch()
#            .batch(self.sequence_length, drop_remainder=True)
#            .batch(self.batch_size, drop_remainder=True)
#        )
#        self.iterator = iter(dataset)
#
#    def _create_parser_fn(self):
#        """Creates a parser function for the TFRecord data."""
#        def _parser_fn(serialized_proto):
#            context_features = {}
#            sequence_features = {
#                tensor_spec.name: tf.io.FixedLenSequenceFeature(
#                    shape=tensor_spec.shape, dtype=tensor_spec.dtype
#                )
#                for tensor_spec in self.time_step_spec['observation'].values()
#            }
#            sequence_features[self.action_spec.name] = tf.io.FixedLenSequenceFeature(
#                shape=self.action_spec.shape, dtype=self.action_spec.dtype
#            )
#
#            _, parsed_sequence = tf.io.parse_single_sequence_example(
#                serialized_proto,
#                context_features=context_features,
#                sequence_features=sequence_features,
#            )
#            
#            observation = parsed_sequence
#            action = parsed_sequence.pop(self.action_spec.name)
#            
#            return {'observation': observation, 'action': action}
#
#        return _parser_fn
#
#    def __iter__(self):
#        return self
#
#    def __next__(self):
#        try:
#            data = next(self.iterator)
#            # Convert TensorFlow tensors to NumPy arrays
#            obs_numpy = {k: v.numpy() for k, v in data['observation'].items()}
#            action_numpy = data['action'].numpy()
#            return obs_numpy, action_numpy
#        except StopIteration:
#            # Restart the iterator for the next epoch
#            self._create_iterator()
#            raise StopIteration
#
#    def __len__(self):
#        return 0
#
