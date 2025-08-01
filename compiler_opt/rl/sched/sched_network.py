# Copyright 2020 Google LLC
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
"""Actor network for Instruction Scheduling."""

from typing import Any
from collections.abc import Sequence, Callable

import gin
import tensorflow as tf
from tf_agents.networks import categorical_projection_network
from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.typing import types
from tf_agents.utils import nest_utils
from tf_agents.networks.utils import BatchSquash


class SchedEncodingNetwork(encoding_network.EncodingNetwork):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # remove the first layer (Flatten) in postprocessing_layers cause this will
    # flatten the B x T x 256 x dim to B x T x (256 x dim).
    self._postprocessing_layers = self._postprocessing_layers[1:]

  def call(self,
           observation: types.NestedTensor,
           step_type=None,
           network_state=(),
           training: bool = False):
    # The following code is a modified version of the base class's call method.
    # It is modified to apply the mask to the processed features before they
    # are combined.
    del step_type # unused.

    if self._batch_squash:
      outer_rank = nest_utils.get_outer_rank(observation,
                                             self.input_tensor_spec)
      batch_squash = BatchSquash(outer_rank)
      observation = tf.nest.map_structure(batch_squash.flatten, observation)

    # Reconstruct the dictionary of layers to ensure correct mapping.
    layers_dict = tf.nest.pack_sequence_as(self._preprocessing_nest,
                                           self._flat_preprocessing_layers)

    # Process each observation with its corresponding layer.
    processed_dict = {}
    if layers_dict is not None:
      for key, tensor in observation.items():
        processed_dict[key] = layers_dict[key](tensor, training=training)
    else:
      processed_dict = observation

    # Apply the mask to the processed features.
    mask = processed_dict['mask']
    masked_dict = {}
    for name, tensor in processed_dict.items():
      if name == 'mask':
        masked_dict[name] = tensor
      else:
        masked_dict[name] = tf.multiply(tensor, mask)
    processed = masked_dict

    # Combine the processed features.
    if self._preprocessing_combiner is not None:
      # Flatten the dict in alphabetical order for the combiner.
      tensors_to_combine = [
          processed[key] for key in sorted(processed.keys())
      ]
      processed = self._preprocessing_combiner(tensors_to_combine)

    # Apply post-processing layers.
    state = processed
    for layer in self._postprocessing_layers:
      state = layer(state, training=training)

    # Un-squash.
    if self._batch_squash:
      state = tf.nest.map_structure(batch_squash.unflatten, state)

    return state, network_state


class SchedProbProjectionNetwork(
    categorical_projection_network.CategoricalProjectionNetwork):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # shape after projection_layer: B x T x 256 x 1; then gets re-shaped to
    # B x T x 256.
    self._projection_layer = tf.keras.layers.Dense(
        1,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=kwargs['logits_init_output_factor']),
        bias_initializer=tf.keras.initializers.Zeros(),
        name='logits')

@gin.configurable
class SchedRNDEncodingNetwork(SchedEncodingNetwork):

  def __init__(self, **kwargs):
    pooling_layer = tf.keras.layers.GlobalMaxPool1D(data_format='channels_last')
    super().__init__(**kwargs)
    # add a pooling layer at the end to to convert B x T x 256 x dim to
    # B x T x dim.
    self._postprocessing_layers.append(pooling_layer)


@gin.configurable
class SchedNetwork(network.DistributionNetwork):
  """Creates the actor network for instruction scheduling policy training."""

  def __init__(
      self,
      input_tensor_spec: types.NestedTensorSpec,
      output_tensor_spec: types.NestedTensorSpec,
      preprocessing_layers: types.NestedLayer | None = None,
      preprocessing_combiner: tf.keras.layers.Layer | None = None,
      conv_layer_params: Sequence[Any] | None = None,
      fc_layer_params: Sequence[int] | None = (200, 100),
      dropout_layer_params: Sequence[float] | None = None,
      activation_fn: Callable[[types.Tensor],
                              types.Tensor] = tf.keras.activations.relu,
      kernel_initializer: tf.keras.initializers.Initializer | None = None,
      batch_squash: bool = True,
      dtype: tf.DType = tf.float32,
      name: str = 'SchedNetwork'):
    """Creates an instance of `SchedNetwork`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input.
      output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
        the output.
      preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
        representing preprocessing for the different observations.
        All of these layers must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them. Good options include
        `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
        This layer must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer.
      dropout_layer_params: Optional list of dropout layer parameters, each item
        is the fraction of input units to drop or a dictionary of parameters
        according to the keras.Dropout documentation. The additional parameter
        `permanent`, if set to True, allows to apply dropout at inference for
        approximated Bayesian inference. The dropout layers are interleaved with
        the fully connected layers; there is a dropout layer after each fully
        connected layer, except if the entry in the list is None. This list must
        have the same length of fc_layer_params, or be None.
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      kernel_initializer: Initializer to use for the kernels of the conv and
        dense layers. If none is provided a default glorot_uniform.
      batch_squash: If True the outer_ranks of the observation are squashed into
        the batch dimension. This allow encoding networks to be used with
        observations with shape [BxTx...].
      dtype: The dtype to use by the convolution and fully connected layers.
      name: A string representing name of the network.

    Raises:
      ValueError: If `input_tensor_spec` contains more than one observation.
    """

    if not kernel_initializer:
      kernel_initializer = tf.compat.v1.keras.initializers.glorot_uniform()

    # input: B x T x obs_spec
    # output: B x T x 256 x dim
    encoder = SchedEncodingNetwork(
        input_tensor_spec=input_tensor_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
        dropout_layer_params=dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        batch_squash=batch_squash,
        dtype=dtype)

    projection_network = SchedProbProjectionNetwork(
        sample_spec=output_tensor_spec, logits_init_output_factor=0.1)
    output_spec = projection_network.output_spec

    super().__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        output_spec=output_spec,
        name=name)

    self._encoder = encoder
    self._projection_network = projection_network
    self._output_tensor_spec = output_tensor_spec

  @property
  def output_tensor_spec(self):
    return self._output_tensor_spec

  def call(self,
           observations: types.NestedTensor,
           step_type: types.NestedTensor,
           network_state=(),
           training: bool = False,
           mask=None):
    _ = mask
    state, network_state = self._encoder(
        observations,
        step_type=step_type,
        network_state=network_state,
        training=training)
    tf.debugging.check_numerics(state, message="[SchedNetwork] output contains NaN or Inf!")
    outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)

    distribution, _ = self._projection_network(
        state, outer_rank, training=training, mask=None)

    tf.debugging.check_numerics(distribution.logits,
                                message="[SchedNetwork] distribution contains NaN or Inf!")

    return distribution, network_state
