import functools
import os
import time

from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf

from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts

from compiler_opt.distributed.local.local_worker_manager import LocalWorkerPoolManager
from compiler_opt.rl import corpus
from compiler_opt.rl import local_data_collector
from compiler_opt.rl import registry
from compiler_opt.rl import compilation_runner

_ROOT_DIR = flags.DEFINE_string(
    'root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.')
_DATA_PATH = flags.DEFINE_string(
    'data_path',
    None,
    'Path to directory containing the corpus.',
    required=True)
_POLICY_PATH = flags.DEFINE_string(
    'policy_path', None, 'Path to the exported TensorFlow SavedModel policy.', required=True)
_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers', None,
    'Number of parallel data collection workers. `None` for max available')
_GIN_FILES = flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.')
_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')

@gin.configurable
def collect_data(
    root_dir: str,
    data_path: str,
    policy_path: str,
    num_workers: int,
    worker_manager_class: type[LocalWorkerPoolManager] = LocalWorkerPoolManager,
    num_modules=100,
    moving_average_decay_rate=1,
    output_tfrecord_path=None,
    sequence_length=1, # Add sequence_length parameter
):
    """Collects data using an exported TensorFlow SavedModel policy."""
    problem_config = registry.get_configuration()

    logging.info('Loading module specs from corpus at %s.', data_path)
    cps = corpus.Corpus(
        data_path=data_path,
        additional_flags=problem_config.flags_to_add(),
        delete_flags=problem_config.flags_to_delete(),
        replace_flags=problem_config.flags_to_replace())
    logging.info('Done loading module specs from corpus.')

    # Load the TensorFlow SavedModel policy
    loaded_policy = tf.saved_model.load(policy_path)
    
    class SavedModelPolicyWrapper:
        def __init__(self, policy_model):
            self._policy_model = policy_model

        def action(self, time_step: ts.TimeStep, policy_state=()):
            # time_step.observation is a dict of tensors
            obs_dict = {k: tf.convert_to_tensor(v, dtype=tf.float32) for k, v in time_step.observation.items()}
            
            # Call the serving_default signature of the SavedModel
            # This assumes the signature takes kwargs matching observation keys
            # and returns a dict with 'action_logits' key.
            action_logits = self._policy_model(**obs_dict)['action_logits']
            
            # Sample action from logits (assuming categorical distribution)
            action_distribution = tf.compat.v2.distributions.Categorical(logits=action_logits)
            action = action_distribution.sample()
            
            # Return a PolicyStep object
            return policy_step.PolicyStep(action=action, state=policy_state, info=())

    policy_wrapper = SavedModelPolicyWrapper(loaded_policy)

    with worker_manager_class(
        worker_class=problem_config.get_runner_type(),
        count=num_workers,
        worker_kwargs=dict(
            moving_average_decay_rate=moving_average_decay_rate)) as worker_pool:

        data_collector = local_data_collector.LocalDataCollector(
            cps=cps,
            num_modules=num_modules,
            worker_pool=worker_pool,
            parser=None, # No parser needed here, as we are collecting raw data
            reward_stat_map=None, # Not managing reward stats in this collector
            best_trajectory_repo=None, # Not managing best trajectories here
        )

        logging.info("Starting data collection...")
        dataset_iter, monitor_dict = data_collector.collect_data(
            policy=policy_wrapper,
            model_id=0, # Dummy model_id for collection
        )
        logging.info("Data collection finished.")

        # Batch the collected steps into trajectories
        batched_dataset_iter = dataset_iter.batch(sequence_length, drop_remainder=True)

        # Save the collected data to TFRecord files
        if output_tfrecord_path:
            # The batched_dataset_iter yields (time_step, policy_step) tuples where each element
            # is now a batch of sequence_length steps.
            def to_sequence_example(time_step, policy_step):
                feature_lists = tf.train.FeatureLists()

                # Observations
                for k, v in time_step.observation.items():
                    # v is now (sequence_length, feature_dim) or (sequence_length,)
                    if v.dtype == tf.int64:
                        feature_list = tf.train.FeatureList(feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=tf.reshape(x, [-1]))) for x in tf.unstack(v)])
                    elif v.dtype == tf.float32:
                        feature_list = tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(x, [-1]))) for x in tf.unstack(v)])
                    else:
                        raise ValueError(f"Unsupported dtype for observation {k}: {v.dtype}")
                    feature_lists.feature_list[k].CopyFrom(feature_list)
                
                # Actions
                # policy_step.action is now (sequence_length,)
                action_feature_list = tf.train.FeatureList(feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=[x])) for x in tf.unstack(policy_step.action)])
                feature_lists.feature_list[problem_config.get_signature_spec()[1].name].CopyFrom(action_feature_list)

                # Rewards
                # time_step.reward is now (sequence_length,)
                reward_feature_list = tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=[x])) for x in tf.unstack(time_step.reward)])
                feature_lists.feature_list['reward'].CopyFrom(reward_feature_list)

                return tf.train.SequenceExample(feature_lists=feature_lists)

            # Map the dataset to SequenceExamples and write to TFRecord
            writer = tf.data.experimental.TFRecordWriter(output_tfrecord_path)
            writer.write(batched_dataset_iter.map(to_sequence_example))
            logging.info(f"Collected data saved to {output_tfrecord_path}")

def main(_):
    gin.parse_config_files_and_bindings(
        _GIN_FILES.value, bindings=_GIN_BINDINGS.value, skip_unknown=False)
    logging.info(gin.config_str())

    collect_data(
        root_dir=_ROOT_DIR.value,
        data_path=_DATA_PATH.value,
        policy_path=_POLICY_PATH.value,
        num_workers=_NUM_WORKERS.value,
        output_tfrecord_path=os.path.join(_ROOT_DIR.value, "collected_data.tfrecord")
    )

if __name__ == '__main__':
    multiprocessing.handle_main(functools.partial(app.run, main))

