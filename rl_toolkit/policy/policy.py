from abc import ABC, abstractmethod

import tensorflow as tf


class Policy(ABC):
    """
    Base class for policies
    =================

    Attributes:
        env: the instance of environment object
        log_wandb (bool): log into WanDB cloud
    """

    def __init__(
        self,
        # ---
        env,
        # ---
        log_wandb: bool = False,
    ):
        self._env = env
        self._log_wandb = log_wandb

        # check obseration's ranges
        if tf.all(tf.isfinite(self._env.observation_space.low)) and tf.all(
            tf.isfinite(self._env.observation_space.high)
        ):
            self._normalize = self._normalize_fn

            print("Observation will be normalized !\n")
        else:
            self._normalize = lambda a: a

            print("Observation cannot be normalized !\n")

        # actual training step
        self._train_step = tf.Variable(
            0,
            trainable=False,
            dtype=tf.int32,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            shape=(),
        )

        self._stop_agents = tf.Variable(
            False,
            trainable=False,
            dtype=tf.bool,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            shape=(),
        )

        # prepare variable container
        self._variables_container = {
            "train_step": self._train_step,
            "stop_agents": self._stop_agents,
            "policy_variables": self._actor.model.variables,
        }

        # variables signature for variable container table
        self._variable_container_signature = tf.nest.map_structure(
            lambda variable: tf.TensorSpec(variable.shape, dtype=variable.dtype),
            self._variables_container,
        )
        self._dtypes = tf.nest.map_structure(
            lambda spec: spec.dtype, self._variable_container_signature
        )

    def _normalize_fn(self, obs):
        # Min-max method
        return obs / self._env.observation_space.high

    @tf.function
    def _get_action(self, state, deterministic):
        a, _ = self._actor.predict(
            tf.expand_dims(state, axis=0),
            with_logprob=False,
            deterministic=deterministic,
        )
        return tf.squeeze(a, axis=0)  # remove batch_size dim

    @tf.function
    def _update_variables(self):
        sample = self.tf_client.sample("variables", data_dtypes=[self._dtypes])
        for variable, value in zip(
            tf.nest.flatten(self._variables_container), tf.nest.flatten(sample.data[0])
        ):
            variable.assign(value)

    def _push_variables(self):
        self.tf_client.insert(
            data=tf.nest.flatten(self._variables_container),
            tables=tf.constant(["variables"]),
            priorities=tf.constant([1.0], dtype=tf.float64),
        )

    @abstractmethod
    def run(self):
        ...