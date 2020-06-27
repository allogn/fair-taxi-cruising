import tensorflow as tf
import numpy as np

from stable_baselines.common.policies import ActorCriticPolicy, register_policy, mlp_extractor
from stable_baselines.common.tf_layers import linear
from stable_baselines import A2C

#! embed fake masks from updated policies from new stable baselines
#TODO what is that reuse option everywhere?
class TaxiPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
                 act_fun=tf.tanh, **kwargs):
        super(TaxiPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=1) # scale=1 because of mlp

        self._kwargs_check("mlp", kwargs)

        layers = [128, 64, 32]
        net_arch = [dict(vf=layers, pi=layers)]

        with tf.variable_scope("model", reuse=reuse):
            pi_latent, vf_latent = mlp_extractor(tf.layers.flatten(self.processed_obs), net_arch, act_fun)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False, action_mask=None):
        feed_dict = {self.obs_ph: obs}
        if action_mask is not None and len(action_mask) != 0:
            feed_dict[self.action_mask_ph] = np.array(action_mask, dtype=np.float32)
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   feed_dict)
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   feed_dict)
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None, action_mask=None):
        feed_dict = {self.obs_ph: obs}
        if action_mask is not None and len(action_mask) != 0:
            feed_dict[self.action_mask_ph] = np.array(action_mask, dtype=np.float32)
        return self.sess.run(self.policy_proba, feed_dict)

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})