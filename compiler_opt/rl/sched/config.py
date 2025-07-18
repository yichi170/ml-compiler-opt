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
"""Instruction scheduling training config."""

import gin
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from compiler_opt.rl import feature_ops


def get_num_candidates():
  return 256


# pylint: disable=g-complex-comprehension
@gin.configurable()
def get_sched_signature_spec():
  """Returns (time_step_spec, action_spec) for LLVM register allocation."""
  num_candidates = get_num_candidates()

  observation_spec = {
      key: tf.TensorSpec(dtype=tf.int64, shape=(num_candidates), name=key)
      for key in ('mask', 'is_top', 'is_bot')
  }

  observation_spec.update({
      key: tf.TensorSpec(dtype=tf.int64, shape=(num_candidates), name=key)
      for key in ('pos', 'excess', 'current_max', 'critical_max',
                  'su_latency', 'su_height', 'su_depth',
                  'su_succs_left', 'su_preds_left', 'su_succs', 'su_preds')
  })

  observation_spec.update({
      key: tf.TensorSpec(dtype=tf.int64, shape=(), name=key)
      for key in ('sgpr_critical_limit', 'vgpr_critical_limit',
                  'sgpr_excess_limit', 'vgpr_excess_limit')
  })

  reward_spec = tf.TensorSpec(dtype=tf.float32, shape=(), name='reward')
  time_step_spec = time_step.time_step_spec(observation_spec, reward_spec)

  action_spec = tensor_spec.BoundedTensorSpec(
      dtype=tf.int64,
      shape=(),
      name='index_to_sched',
      minimum=0,
      maximum=num_candidates - 1)

  return time_step_spec, action_spec


@gin.configurable
def get_observation_processing_layer_creator(quantile_file_dir=None,
                                             with_sqrt=True,
                                             with_z_score_normalization=True,
                                             eps=1e-6):
  """Wrapper for observation_processing_layer."""
  quantile_map = feature_ops.build_quantile_map(quantile_file_dir)

  def cast_to_float_and_apply(fn):
    def wrapped(x):
        x = tf.cast(x, tf.float32)
        return fn(x)
    return wrapped

  def debug_layer(x):
    tf.print("Debug obs:", x, "shape:", tf.shape(x), "dtype:", x.dtype)
    return x

  def observation_processing_layer(obs_spec):
    """Creates the layer to process observation given obs_spec."""
#    if obs_spec.name in ('mask'):
#      return tf.keras.layers.Lambda(feature_ops.discard_fn)

    if obs_spec.name in ('mask', 'is_top', 'is_bot'):
      return tf.keras.layers.Lambda(feature_ops.identity_fn)

    normalize_fn = log_normalize_fn = None
    if obs_spec.name not in get_nonnormalized_features():
      quantile = quantile_map[obs_spec.name]

      first_non_zero = 0
      for x in quantile:
        if x > 0:
          first_non_zero = x
          break

      if first_non_zero == 0:
        first_non_zero = eps

      normalize_fn = feature_ops.get_normalize_fn(quantile, with_sqrt,
                                                  with_z_score_normalization,
                                                  eps)
      log_normalize_fn = feature_ops.get_normalize_fn(
          quantile,
          with_sqrt,
          with_z_score_normalization,
          eps,
          preprocessing_fn=lambda x: tf.math.log(x + first_non_zero))

    if obs_spec.name in ['pos', 'excess', 'current_max', 'critical_max',
                         'su_succs_left', 'su_preds_left', 'su_succs', 'su_preds']:
      fn = cast_to_float_and_apply(normalize_fn)
      return tf.keras.layers.Lambda(fn)

    if obs_spec.name in ('su_latency', 'su_height', 'su_depth'):
      fn = cast_to_float_and_apply(log_normalize_fn)
      return tf.keras.layers.Lambda(fn)

    if obs_spec.name in get_scalar_features():

      def gpr_limit_processing_fn(obs):
        obs = tf.cast(tf.expand_dims(obs, -1), tf.float32)
        obs = tf.tile(obs, [1, get_num_candidates()])
        obs = normalize_fn(obs)
        return obs
      return tf.keras.layers.Lambda(gpr_limit_processing_fn)

    # Make sure all features have a preprocessing function.
    raise KeyError('Missing preprocessing function for some feature.')

  return observation_processing_layer

def get_scalar_features():
  return [
    "sgpr_critical_limit", "vgpr_critical_limit",
    "sgpr_excess_limit", "vgpr_excess_limit",
  ]

def get_nonnormalized_features():
  return [
      'mask', 'is_top', 'is_bot', 'reward'
  ]
