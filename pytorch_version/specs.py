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
"""Data specifications for PyTorch models."""

import tensorflow as tf

def get_regalloc_signature_spec():
    """Returns (time_step_spec, action_spec) for LLVM register allocation."""
    num_registers = 33

    observation_spec = {
        key: tf.TensorSpec(dtype=tf.int64, shape=(num_registers,), name=key)
        for key in ('mask', 'is_hint', 'is_local', 'is_free')
    }
    observation_spec.update({
        key:
            tf.TensorSpec(
                dtype=tf.int64,
                shape=(num_registers,),
                name=key) for key in ('max_stage', 'min_stage')
    })
    observation_spec.update({
        key: tf.TensorSpec(dtype=tf.float32, shape=(num_registers,), name=key)
        for key in ('weighed_reads_by_max', 'weighed_writes_by_max',
                    'weighed_read_writes_by_max', 'weighed_indvars_by_max',
                    'hint_weights_by_max', 'start_bb_freq_by_max',
                    'end_bb_freq_by_max', 'hottest_bb_freq_by_max',
                    'liverange_size', 'use_def_density', 'nr_defs_and_uses',
                    'nr_broken_hints', 'nr_urgent', 'nr_rematerializable')
    })
    observation_spec['progress'] = tf.TensorSpec(
        dtype=tf.float32, shape=(), name='progress')

    reward_spec = tf.TensorSpec(dtype=tf.float32, shape=(), name='reward')
    
    # A minimalist version of TimeStepSpec
    time_step_spec = {
        'observation': observation_spec,
        'reward': reward_spec
    }

    action_spec = tf.TensorSpec(
        dtype=tf.int64,
        shape=(),
        name='index_to_evict')

    return time_step_spec, action_spec

def get_sched_signature_spec():
    """Returns (time_step_spec, action_spec) for LLVM instruction scheduling."""
    num_candidates = 256

    observation_spec = {
        key: tf.TensorSpec(dtype=tf.int64, shape=(num_candidates,), name=key)
        for key in ('mask', 'is_top', 'is_bot', 'pos',
                    'excess', 'current_max', 'critical_max',
                    'su_latency', 'su_height', 'su_depth',
                    'su_succs_left', 'su_preds_left', 'su_succs', 'su_preds')
    }

    observation_spec.update({
        key: tf.TensorSpec(dtype=tf.int64, shape=(), name=key)
        for key in ('sgpr_critical_limit', 'vgpr_critical_limit',
                    'sgpr_excess_limit', 'vgpr_excess_limit')
    })

    reward_spec = tf.TensorSpec(dtype=tf.float32, shape=(), name='reward')
    
    time_step_spec = {
        'observation': observation_spec,
        'reward': reward_spec
    }

    action_spec = tf.TensorSpec(
        dtype=tf.int64,
        shape=(),
        name='index_to_sched')

    return time_step_spec, action_spec
