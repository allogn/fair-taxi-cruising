import networkx as nx
import numpy as np
import tensorflow as tf
import random, os, sys
import logging
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from alg_utility import *
from copy import deepcopy

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Estimator:
    """ build value network
    """
    def __init__(self,
                 sess,
                 world,
                 n_intervals,
                 seed,
                 entropy_coef,
                 scope="estimator",
                 summary_dir=None,
                 wc=0,
                 include_income=False):
        self.sess = sess
        self.n_valid_grid = len(world)
        self.wc = wc
        self.world = world
        self.seed(seed)
        self.include_income = include_income
        self.entropy_coef = entropy_coef

        self.T = n_intervals
        self.action_dim = int(np.max([d for n, d in world.degree()])) + 1 # last one always means staying
        self.state_dim = len(world) * 3 + self.T
        if include_income:
            self.state_dim += len(world)

        self.scope = scope

        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):

            # Build the value function graph
            # with tf.variable_scope("value"):
            value_loss = self._build_value_model()

            with tf.variable_scope("policy"):
                actor_loss, entropy = self._build_mlp_policy()

            self.loss = actor_loss + .5 * value_loss - 10 * entropy


            # self.loss_gradients = tf.gradients(self.value_loss, tf.trainable_variables(scope=scope))
                                           # tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("value_loss", self.value_loss),
            tf.summary.scalar("value_output", tf.reduce_mean(self.value_output)),
            # tf.summary.scalar("gradient_norm_policy", tf.reduce_sum([tf.norm(item) for item in self.loss_gradients]))
        ])

        self.policy_summaries = tf.summary.merge([
            tf.summary.scalar("policy_loss", self.policy_loss),
            tf.summary.scalar("adv", tf.reduce_mean(self.tfadv)),
            tf.summary.scalar("entropy", self.entropy),
            # tf.summary.scalar("gradient_norm_policy", tf.reduce_sum([tf.norm(item) for item in self.loss_gradients]))
        ])

        self.summary_writer = tf.summary.FileWriter(summary_dir)


        self.neighbors_list = []
        for node_id in range(len(self.world)):
            assert self.world.has_node(node_id), "Nodes in the network assumed to be sequentially enumerated"
            neighbor_ids = list(self.world.neighbors(node_id))
            neighbor_ids.append(node_id)
            self.neighbors_list.append(neighbor_ids)

        # compute valid action mask.
        self.valid_action_mask = np.ones((self.n_valid_grid, self.action_dim))
        self.valid_neighbor_node_id = np.zeros((self.n_valid_grid, self.action_dim))
        for grid_id in self.world.nodes():
            for neighbor_idx, neighbor_node_index in enumerate(self.world.neighbors(grid_id)):
                self.valid_neighbor_node_id[grid_id, neighbor_idx] = neighbor_node_index
            self.valid_neighbor_node_id[grid_id, -1] = grid_id
            for i in range(self.world.degree[grid_id], self.action_dim-1):
                self.valid_action_mask[grid_id, i] = 0

    def _build_value_model(self):

        self.state = X = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name="X")

        # The TD target value
        self.y_pl = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="y")

        self.loss_lr = tf.placeholder(tf.float32, None, "learning_rate")

        # 3 layers feed forward network.
        l1 = fc(X, "l1", 128, act=tf.nn.relu)
        l2 = fc(l1, "l2", 64, act=tf.nn.relu)
        l3 = fc(l2, "l3", 32, act=tf.nn.relu)
        # l1 = tf.layers.dense(X, 1024, tf.nn.sigmoid, trainable=trainable)
        # l2 = tf.layers.dense(l1, 512, tf.nn.sigmoid, trainable=trainable)
        # l3 = tf.layers.dense(l2, 32, tf.nn.sigmoid, trainable=trainable)
        self.value_output = fc(l3, "value_output", 1, act=tf.nn.relu)

        # self.losses = tf.square(self.y_pl - self.value_output)
        self.value_loss = tf.reduce_mean(tf.squared_difference(self.y_pl, self.value_output))

        self.value_train_op = tf.train.AdamOptimizer(self.loss_lr).minimize(self.value_loss)

        return self.value_loss

    def _build_mlp_policy(self):

        self.policy_state = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name="P")
        self.ACTION = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32, name="action")
        self.tfadv = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='advantage')
        self.neighbor_mask = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32, name="neighbormask")
        # this mask filter invalid actions and those action smaller than current grid value.

        l1 = fc(self.policy_state, "l1", 128, act=tf.nn.relu)
        l2 = fc(l1, "l2", 64, act=tf.nn.relu)
        l3 = fc(l2, "l3", 32, act=tf.nn.relu)

        self.logits = logits = fc(l3, "logits", self.action_dim, act=tf.nn.relu) + 1  # avoid valid_logits are all zeros
        self.valid_logits = logits * self.neighbor_mask

        self.softmaxprob = tf.nn.softmax(tf.log(self.valid_logits + 1e-8))
        self.logsoftmaxprob = tf.nn.log_softmax(self.softmaxprob)

        self.neglogprob = - self.logsoftmaxprob * self.ACTION
        self.actor_loss = tf.reduce_mean(tf.reduce_sum(self.neglogprob * self.tfadv, axis=1))
        self.entropy = - tf.reduce_mean(self.softmaxprob * self.logsoftmaxprob)

        self.policy_loss = self.actor_loss - self.entropy_coef * self.entropy # 0.01 is default

        self.policy_train_op = tf.train.AdamOptimizer(self.loss_lr).minimize(self.policy_loss)
        return self.actor_loss, self.entropy

    def predict(self, s):
        value_output = self.sess.run(self.value_output, {self.state: s})

        return value_output

    def seed(self, seed):
        self.random = np.random.RandomState(seed)

    def action(self, s, context, epsilon):
        """ Compute current action for all grids give states

        :param s: 504 x stat_dim,
        :return:
        """
        value_output = self.sess.run(self.value_output, {self.state: s}).flatten()
        action_tuple = []
        valid_prob = []

        # for training policy gradient.
        action_choosen_mat = []
        policy_state = []
        curr_state_value = []
        next_state_ids = []
        curr_neighbor_mask_policy = []

        if self.include_income:
            grid_ids = np.argmax(s[:, -2*self.n_valid_grid:-self.n_valid_grid], axis=1)
        else:
            grid_ids = np.argmax(s[:, -self.n_valid_grid:], axis=1) # one-hot encoding of grid_id, returns grid ids

        # compute neighbor mask according to centralized value
        curr_neighbor_mask = deepcopy(self.valid_action_mask)
        for idx in grid_ids:
            valid_qvalues = value_output[self.neighbors_list[idx]]  # value of current and its nearby grids
            temp_qvalue = np.zeros(self.action_dim)
            temp_qvalue[curr_neighbor_mask[idx] > 0] = valid_qvalues
            # temp_qvalue[temp_qvalue < temp_qvalue[-1]] = 0
            curr_neighbor_mask[idx][np.where(temp_qvalue < temp_qvalue[-1])] = 0
            if (curr_neighbor_mask[idx] == 0).all():
                curr_neighbor_mask[idx] = self.valid_action_mask[idx]

        # compute policy probability.
        action_probs = self.sess.run(self.softmaxprob, {self.policy_state: s,
                                                        self.neighbor_mask: curr_neighbor_mask})

        # sample action.
        assert len(set(grid_ids)) == len(grid_ids)
        for grid_valid_idx in grid_ids:
            action_prob = action_probs[grid_valid_idx]

            # cast invalid action to zero, avlid numerical issue.
            action_prob[self.valid_action_mask[grid_valid_idx] == 0] = 0
            valid_prob.append(action_prob)   # action probability for state value function
            if int(context[grid_valid_idx]) == 0:
                continue

            # context has preassigned drivers
            curr_action_indices_temp = self.random.choice(self.action_dim, int(context[grid_valid_idx]), # from where and how many
                                                        p=action_prob/np.sum(action_prob))
            # print(int(context[grid_valid_idx]), " context for ", grid_valid_idx, len(curr_action_indices_temp))
            # num of drivers dispatched to nearby locations [2,3,2,3,1,3,3]
            # for numerically stable, avoid sum of action_prob > 1 with small value

            curr_action_indices = [0] * self.action_dim
            for kk in curr_action_indices_temp:
                curr_action_indices[kk] += 1

            start_node_id = grid_valid_idx
            for curr_action_idx, num_driver in enumerate(curr_action_indices):
                if num_driver > 0:
                    assert self.valid_action_mask[grid_valid_idx, curr_action_idx] == 1
                    a = [self.valid_neighbor_node_id[grid_valid_idx, i] for i in range(self.action_dim) if self.valid_action_mask[grid_valid_idx, i] == 1]
                    assert len(set(a)) == len(a)

                    end_node_id = int(self.valid_neighbor_node_id[grid_valid_idx, curr_action_idx])
                    if end_node_id != start_node_id:
                        action_tuple.append((start_node_id, end_node_id, num_driver))

                    # book keeping for training
                    temp_a = np.zeros(self.action_dim)
                    temp_a[curr_action_idx] = 1
                    action_choosen_mat.append(temp_a)
                    policy_state.append(s[grid_valid_idx])
                    curr_state_value.append(value_output[grid_valid_idx])
                    next_state_ids.append(self.valid_neighbor_node_id[grid_valid_idx, curr_action_idx])
                    curr_neighbor_mask_policy.append(curr_neighbor_mask[grid_valid_idx])

        return action_tuple, np.stack(valid_prob), \
               np.stack(policy_state) if len(policy_state) > 0 else np.array([]), \
               np.stack(action_choosen_mat) if len(action_choosen_mat) > 0 else np.array([]), curr_state_value, \
               np.stack(curr_neighbor_mask_policy) if len(curr_neighbor_mask_policy) > 0 else np.array([]), next_state_ids

    def compute_advantage(self, curr_state_value, next_state_ids, next_state, node_reward, gamma):
        """for policy network"""
        advantage = []
        node_reward = node_reward.flatten()
        qvalue_next = self.sess.run(self.value_output, {self.state: next_state}).flatten()

        for idx, next_state_id in enumerate(next_state_ids):
            next_state_id = int(next_state_id)
            temp_adv = node_reward[next_state_id] - self.wc + gamma * qvalue_next[next_state_id] - curr_state_value[idx]
            advantage.append(temp_adv)
        return advantage

    def compute_targets(self, valid_prob, next_state, node_reward, gamma):
        targets = []
        node_reward = node_reward.flatten()
        qvalue_next = self.sess.run(self.value_output, {self.state: next_state}).flatten()

        for idx in np.arange(self.n_valid_grid):
            grid_prob = valid_prob[idx][self.valid_action_mask[idx]>0]
            neighbor_grid_ids = self.neighbors_list[idx]

            nw = node_reward[neighbor_grid_ids]
            qn = qvalue_next[neighbor_grid_ids]

            curr_grid_target = np.sum(grid_prob * (nw - self.wc + gamma * qn))
            targets.append(curr_grid_target)

        return np.array(targets).reshape([-1, 1])

    def initialization(self, s, y, learning_rate):
        sess = self.sess
        feed_dict = {self.state: s, self.y_pl: y, self.loss_lr: learning_rate}
        _, value_loss = sess.run([self.value_train_op, self.value_loss], feed_dict)
        return value_loss

    def update_policy(self, policy_state, advantage, action_choosen_mat, curr_neighbor_mask, learning_rate, global_step):
        sess = self.sess
        feed_dict = {self.policy_state: policy_state,
                     self.tfadv: advantage,
                     self.ACTION: action_choosen_mat,
                     self.neighbor_mask: curr_neighbor_mask,
                     self.loss_lr: learning_rate}
        summaries, _, loss = sess.run([self.policy_summaries, self.policy_train_op, self.policy_loss], feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
            self.summary_writer.flush()
        return loss

    def update_value(self, s, y, learning_rate, global_step):
        """
        Updates the estimator towards the given targets.

        Args:
          s: State input of shape [batch_size, state_dim]
          a: Chosen actions of shape [batch_size, action_dim], 0, 1 mask
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        sess = self.sess
        feed_dict = {self.state: s, self.y_pl: y, self.loss_lr: learning_rate}
        summaries, _, loss = sess.run([self.summaries, self.value_train_op, self.value_loss], feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
            self.summary_writer.flush()
        return loss




class stateProcessor:
    """
        Process a raw global state into the states of grids.
    """

    def __init__(self,
                 action_dim,
                 n_valid_grids,
                 n_periods,
                 include_income):
        self.n_valid_grids = n_valid_grids
        self.T = n_periods
        self.include_income = include_income
        self.action_dim = action_dim
        self.extend_state = True

    @staticmethod
    def utility_conver_states(curr_state):
        return np.array(curr_state, dtype=float).flatten()

    @staticmethod
    def utility_normalize_states(curr_s, world_size):
        max_driver_num = np.max(curr_s[:world_size])
        max_order_num = np.max(curr_s[world_size:])
        if max_order_num == 0:
            max_order_num = 1
        if max_driver_num == 0:
            max_driver_num = 1
        curr_s_new = np.zeros_like(curr_s)
        curr_s_new[:world_size] = curr_s[:world_size] / max_driver_num
        curr_s_new[world_size:] = curr_s[world_size:] / max_order_num
        return curr_s_new

    def utility_conver_reward(self, reward_node):
        return np.array(reward_node[0]) # ignore neighbor reward

    def reward_wrapper(self, info, curr_s):
        """ reformat reward from env to the input of model.
        :param info: [node_reward(including neighbors), neighbor_reward]
        :param curr_s:  processed by utility_conver_states, same time step as info.
        :return:
        """

        info_reward = info[0] # [node_reward, neighbor_reward]
        valid_nodes_reward = self.utility_conver_reward(info_reward)
        divide = curr_s[:self.n_valid_grids]
        divide[divide == 0] = 1
        valid_nodes_reward = valid_nodes_reward/divide  # averaged rewards for drivers arriving this grid
        return valid_nodes_reward

    @staticmethod
    def compute_context(info):
        return info.flatten()

    def to_grid_states(self, curr_s, curr_city_time, income_mat = None):
        """ extend global state to all agents' state.

        :param curr_s:
        :param curr_city_time: curr_s time step
        :return:
        """
        T = self.T

        # curr_s = self.utility_conver_states(curr_state)
        time_one_hot = np.zeros((T))
        time_one_hot[curr_city_time % T] = 1
        onehot_grid_id = np.eye(self.n_valid_grids)

        if self.include_income:
            dim = self.n_valid_grids * 4 + T
        else:
            dim = self.n_valid_grids * 3 + T

        s_grid = np.zeros((self.n_valid_grids, dim)) # column of [driver_locs cust_locs time grid_id]
        s_grid[:, :self.n_valid_grids * 2] = np.stack([curr_s] * self.n_valid_grids)
        s_grid[:, self.n_valid_grids * 2:self.n_valid_grids * 2 + T] = np.stack([time_one_hot] * self.n_valid_grids)

        if self.include_income:
            assert income_mat is not None
            s_grid[:, -2*self.n_valid_grids:-self.n_valid_grids] = onehot_grid_id
            s_grid[:, -self.n_valid_grids:] = income_mat
        else:
            s_grid[:, -self.n_valid_grids:] = onehot_grid_id

        return np.array(s_grid)

    def to_grid_rewards(self, node_reward):
        """
        :param node_reward: curr_city_time + 1 's reward
        :return:
        """
        return np.array(node_reward).reshape([-1, 1])

    def to_action_mat(self, action_neighbor_idx):
        action_mat = np.zeros((len(action_neighbor_idx), self.action_dim))
        action_mat[np.arange(action_mat.shape[0]), action_neighbor_idx] = 1
        return action_mat


class policyReplayMemory:
    def __init__(self, memory_size, batch_size):
        self.states = []
        # self.next_states = []
        self.neighbor_mask = []
        self.actions = []
        self.rewards = []  # advantages

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0

    def add(self, s, a, r, mask):
        if self.curr_lens == 0:
            self.states = s
            self.actions = a
            self.rewards = r
            self.neighbor_mask = mask
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, s),axis=0)
            self.neighbor_mask = np.concatenate((self.neighbor_mask, mask), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.curr_lens = self.states.shape[0]
        else:
            new_sample_lens = s.shape[0]
            index = random.randint(0, self.curr_lens - new_sample_lens)

            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r
            self.neighbor_mask[index:(index + new_sample_lens)] = mask

    def sample(self):
        assert self.curr_lens > 0
        if self.curr_lens <= self.batch_size:
            logging.warning("Batch size is too large: no memory sampling performed. Batch size={}".format(self.curr_lens))
            return [self.states, self.actions, np.array(self.rewards), self.neighbor_mask]
        indices = random.sample(range(0, self.curr_lens), self.batch_size)
        batch_s = self.states[indices]
        batch_a = self.actions[indices]
        batch_r = self.rewards[indices]
        batch_mask = self.neighbor_mask[indices]
        return [batch_s, batch_a, batch_r, batch_mask]

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.neighbor_mask = []
        self.curr_lens = 0


class ReplayMemory:
    """ collect the experience and sample a batch for training networks.
        without time ordering
    """
    def __init__(self, memory_size, batch_size):
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0  # current memory lens

    def add(self, s, a, r, next_s):
        if self.curr_lens == 0:
            self.states = s
            self.actions = a
            self.rewards = r
            self.next_states = next_s
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, s),axis=0)
            self.next_states = np.concatenate((self.next_states, next_s), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.curr_lens = self.states.shape[0]
        else:
            new_sample_lens = s.shape[0]
            index = random.randint(0, self.curr_lens - new_sample_lens)

            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r
            self.next_states[index:(index + new_sample_lens)] = next_s

    def sample(self):
        if self.curr_lens <= self.batch_size:
            return [self.states, self.actions, self.rewards, self.next_states]
        indices = random.sample(range(0, self.curr_lens), self.batch_size)
        batch_s = self.states[indices]
        batch_a = self.actions[indices]
        batch_r = self.rewards[indices]
        batch_mask = self.next_states[indices]
        return [batch_s, batch_a, batch_r, batch_mask]

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.curr_lens = 0



class ModelParametersCopier():
    """
    Copy model parameters of one estimator to another.
    """

    def __init__(self, estimator1, estimator2):
        """
        Defines copy-work operation graph.
        Args:
          estimator1: Estimator to copy the paramters from
          estimator2: Estimator to copy the parameters to
        """
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        self.update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            self.update_ops.append(op)

    def make(self, sess):
        """
        Makes copy.
        Args:
            sess: Tensorflow session instance
        """
        sess.run(self.update_ops)
