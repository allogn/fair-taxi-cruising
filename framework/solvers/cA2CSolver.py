import os, sys
import tensorflow as tf
import pickle as pkl
import time
import uuid
from tqdm import tqdm
import logging

import gym

from framework.solvers.cA2C.cA2C import *
from framework.solvers.TestingSolver import TestingSolver
from framework.Generator import Generator
from framework.ParameterManager import ParameterManager

class cA2CSolver(TestingSolver):

    def __init__(self, **params):
        t1 = time.time()

        if params['include_income_to_observation'] == 0:
            params['normalize_rewards'] = 1
        else:
            params['normalize_rewards'] = 0

        super().__init__(**params)



        self.sess = tf.Session()
        self.time_periods = self.params['dataset']["time_periods"]
        self.log_dir = self.dpath
        self.load_dataset()
        self.solver_signature = self.get_solver_signature()
        self.init_env()
        self.init()
        self.log['init_time'] = time.time() - t1

    def get_env_params(self):
        env_params = {
            "world": self.world,
            "orders": self.real_orders,
            "order_sampling_rate": 1./self.params['dataset']['days']*self.params['dataset']['order_sampling_multiplier'],
            "drivers_per_node": self.idle_driver_locations[0,:],
            "n_intervals": self.time_periods,
            "wc": self.params["wc"],
            "count_neighbors": self.params['count_neighbors'] == 1,
            "weight_poorest": self.params['weight_poorest'] == 1,
            "normalize_rewards": self.params['normalize_rewards'] == 1,
            "minimum_reward": self.params['minimum_reward'] == 1,
            "include_income_to_observation": self.params['include_income_to_observation'] == 1,
            "poorest_first": self.params.get("poorest_first", 0) == 1,
            "idle_reward": self.params.get("idle_reward", 0) == 1
        }
        return env_params

    def get_footprint_params(self):
        footprint_params = {
            "n_intervals": self.time_periods,
            "wc": self.params["wc"],
            "count_neighbors": self.params['count_neighbors'] == 1,
            "weight_poorest": self.params['weight_poorest'] == 1,
            "normalize_rewards": self.params['normalize_rewards'] == 1,
            "minimum_reward": self.params['minimum_reward'] == 1,
            "include_income_to_observation": self.params['include_income_to_observation'] == 1,
            "poorest_first": self.params.get("poorest_first", 0) == 1,
            "idle_reward": self.params.get("idle_reward", 0) == 1
        }
        return footprint_params

    def init_env(self):
        env_params = self.get_env_params()
        env_id = "TaxiEnvBatch{}-v01".format(str(uuid.uuid4()))
        gym.envs.register(
            id=env_id,
            entry_point='gym_taxi.envs:TaxiEnvBatch',
            kwargs=env_params
        )
        self.env = gym.make(env_id)

    def init(self):
        self.sess.close()

        self.sess = tf.Session()
        tf.set_random_seed(np.random.randint(1,10000))

        summary_dir = os.path.join(self.dpath, self.solver_signature)
        self.q_estimator = Estimator(self.sess, self.env.world, self.time_periods,
                                        scope=self.solver_signature, summary_dir=summary_dir, wc=self.params["wc"],
                                        include_income = self.params['include_income_to_observation'] == 1)
        self.stateprocessor = stateProcessor(self.q_estimator.action_dim, self.q_estimator.n_valid_grid, self.time_periods,
                                    self.params['include_income_to_observation'] == 1)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=5)
        if self.params.get('mode','Train') == "Test":
            iter = self.params["iterations"]-1 # load model of the last iteration
            self.saver.restore(self.sess, os.path.join(self.log_dir,"{}_model{}.ckpt".format(self.solver_signature, iter)))
        tf.reset_default_graph()

    def set_random_seed(self, seed):
        pass

    def train(self):
        t1 = time.time()
        time_periods = self.time_periods
        replay = ReplayMemory(memory_size=1e+6, batch_size=int(3e+3))
        policy_replay = policyReplayMemory(memory_size=1e+6, batch_size=int(3e+3))

        save_random_seed = []
        episode_dispatched_drivers = []

        all_rewards = []
        order_response_rate_episode = []
        value_table_sum = []
        episode_rewards = []
        episode_conflicts_drivers = []
        record_all_order_response_rate = []
        self.log['iterations_stats'] = {}

        global_step1 = 0
        global_step2 = 0
        RATIO = 1
        if self.verbose:
            pbar = tqdm(total=self.params["iterations"], desc="Training cA2C (iters)")
        for n_iter in np.arange(self.params["iterations"]):
            RANDOM_SEED = n_iter + 40 # don't know why 40
            self.env.seed(RANDOM_SEED)
            save_random_seed.append(RANDOM_SEED)
            batch_s, batch_a, batch_r = [], [], []
            batch_reward_gmv = []
            epsiode_reward = 0
            num_dispatched_drivers = 0

            # reset env
            observation = self.env.reset()
            init_new_info = self.env.get_reset_info()
            curr_state, info, income_mat = self.observation_to_old_fashioned_info(observation, init_new_info)
            context = self.stateprocessor.compute_context(info)
            curr_s = self.stateprocessor.utility_conver_states(curr_state) # [cars customers] flattened
            # removed normalization from here
            assert self.params['include_income_to_observation'] == 1 or income_mat == None

            s_grid = self.stateprocessor.to_grid_states(curr_s, self.env.time, income_mat)  # add one-hot encoding of time, grid_id and curr_s global

            # s_grid has income from by to_grid_states

            # record rewards to update the value table
            episodes_immediate_rewards = []
            num_conflicts_drivers = []
            curr_num_actions = []
            order_response_rates = []
            nodes_with_drivers_and_orders = []
            non_zero_periods = 0
            for ii in np.arange(time_periods):
                # record_curr_state.append(curr_state)
                # INPUT: state,  OUTPUT: action

                action_tuple, valid_action_prob_mat, policy_state, action_choosen_mat, \
                curr_state_value, curr_neighbor_mask, next_state_ids = self.q_estimator.action(s_grid, context, self.params["epsilon"])
                # context = merged driver locations + order locations

                new_action = self.action_from_valid_prob(valid_action_prob_mat)

                all_drivers = np.sum([n[1]['info'].get_driver_num() for n in self.env.world.nodes(data=True)])
                all_orders = np.sum([n[1]['info'].get_order_num() for n in self.env.world.nodes(data=True)])

                observation, new_reward, done, new_info = self.env.step(new_action)
                
                next_state, info, income_mat = self.observation_to_old_fashioned_info(observation, new_info)
                # immediate_reward = self.stateprocessor.reward_wrapper(info, curr_s) -- outdated, do not count neighbors, env provide averaging inside
                immediate_reward = new_info['reward_per_node']

                # Save transition to replay memory
                if ii != 0 and policy_state_prev.shape[0] > 0:
                    #not sure if it is valid to skip prev time intervals; zero is because there were no actions at prev step
                    # r1, c0
                    r_grid = self.stateprocessor.to_grid_rewards(immediate_reward)

                    # s0, a0, r1  for value newtwork
                    targets_batch = self.q_estimator.compute_targets(action_mat_prev, s_grid, r_grid, self.params["gamma"])


                    # advantage for policy network.
                    advantage = self.q_estimator.compute_advantage(curr_state_value_prev, next_state_ids_prev,
                                                              s_grid, r_grid, self.params["gamma"])

                    replay.add(state_mat_prev, action_mat_prev, targets_batch, s_grid)
                    policy_replay.add(policy_state_prev, action_choosen_mat_prev, advantage, curr_neighbor_mask_prev)
                    non_zero_periods += 1

                # for updating value network
                state_mat_prev = s_grid
                action_mat_prev = valid_action_prob_mat

                # for updating policy net
                action_choosen_mat_prev = action_choosen_mat
                curr_neighbor_mask_prev = curr_neighbor_mask
                policy_state_prev = policy_state
                # for computing advantage
                curr_state_value_prev = curr_state_value
                next_state_ids_prev = next_state_ids

                # s1
                curr_state = next_state
                curr_s = self.stateprocessor.utility_conver_states(next_state)
                normalized_curr_s = self.stateprocessor.utility_normalize_states(curr_s, len(self.world))
                s_grid = self.stateprocessor.to_grid_states(normalized_curr_s, self.env.time, income_mat)  # t0, s0

                # c1
                context = self.stateprocessor.compute_context(info)
                all_rewards.append(new_reward)
                batch_reward_gmv.append(new_reward)
                order_response_rates.append(new_info['served_orders']/(new_info['total_orders']+0.0001))
                curr_num_actions.append(new_info['served_orders'])
                num_conflicts_drivers.append(collision_action(action_tuple))
                nodes_with_drivers_and_orders.append((new_info['nodes_with_drivers'],new_info['nodes_with_orders'],len(self.world)))
            if non_zero_periods == 0:
                logging.error("No actions appeared on Train data, probably too busy drivers")

            episode_testing_stats = self.do_test_iteration()
            self.log['iterations_stats'][str(n_iter)] = episode_testing_stats
            episode_reward = np.sum(batch_reward_gmv[1:])
            episode_rewards.append(episode_reward)
            order_response_rate_episode.append(order_response_rates)
            record_all_order_response_rate.append(order_response_rates)
            episode_conflicts_drivers.append(int(np.sum(num_conflicts_drivers[:-1])))
            episode_dispatched_drivers.append(curr_num_actions)

            #(n_iter, episode_reward, n_iter_order_response_rate, episode_dispatched_drivers[-1], episode_conflicts_drivers[-1])

            # overwriting
            pkl.dump([episode_rewards, order_response_rate_episode, save_random_seed, episode_conflicts_drivers,
                         episode_dispatched_drivers, nodes_with_drivers_and_orders], open(os.path.join(self.log_dir, "{}_results.pkl".format(self.solver_signature)), "wb"))

            # update value network
            for _ in np.arange(self.params['batch_size']):
                batch_s, _, batch_r, _ = replay.sample()
                iloss = self.q_estimator.update_value(batch_s, batch_r, 1e-3, global_step1)
                global_step1 += 1

            # training method 2
            # update policy network
            for _ in np.arange(self.params['batch_size']):
                batch_s, batch_a, batch_r, batch_mask = policy_replay.sample()
                self.q_estimator.update_policy(batch_s, batch_r.reshape([-1, 1]), batch_a, batch_mask, self.params["learning_rate"],
                                          global_step2)
                global_step2 += 1

            self.saver.save(self.sess, os.path.join(self.log_dir,"{}_model{}.ckpt".format(self.solver_signature, n_iter)))
            if self.verbose:
                pbar.update()

        if self.verbose:
            pbar.close()

        self.log['train_time'] = time.time() - t1
        self.log["episode_rewards"] = episode_rewards
        self.log["order_response_rate_episode"] = order_response_rate_episode
        self.log["record_all_order_response_rate"] = record_all_order_response_rate
        self.log["episode_conflicts_drivers"] = episode_conflicts_drivers
        self.log["episode_dispatched_drivers"] = episode_dispatched_drivers


    def observation_to_old_fashioned_info(self, observation, new_info):
        '''
        :param observation: a vector returned from taxy_gym environment

        :return: current_state and info variables returned by the old env.reset_clean() and env.step_pre_order_assigin()

                - current_state is an array of shape [2, world-size] with idle_driver_num and order_num in two columns
                - info is is an array of shape [2, world-size] with remain drivers and remain orders per node after matching

        Note that observation is normalized, while current state shouldn't be!
        '''
        current_state = observation[:2*len(self.world)].reshape(2, len(self.world))
        current_state[0,:] *= new_info["driver normalization constant"]
        current_state[1,:] *= new_info["order normalization constant"]
        current_state_int = np.array(current_state, dtype=int)
        assert np.sum(current_state_int) - np.sum(current_state) < 0.001
        current_state = current_state_int
        info = self.env.compute_remaining_drivers_and_orders(current_state)
        if self.params['include_income_to_observation']:
            assert observation[:-3*len(self.world)].shape[0] == 2*len(self.world) + self.env.n_intervals, "Observation is missing income"
            income_mat = observation[-3*len(self.world):].reshape(len(self.world), 3)
        else:
            income_mat = None
        return current_state, info, income_mat

    def action_from_valid_prob(self, valid_prob):
        '''
        :param valid_prob: a list of probability distributions of actions for each node, returned by Estimator.action()

        :return: an action vector for taxi_gym_batch environment, which is a concatenation of actions per cell
        '''
        one_action_shape = self.env.get_action_space_shape()[0]
        action = np.zeros(one_action_shape*len(self.world))
        for n in self.world.nodes():
            action[n*one_action_shape:(n+1)*one_action_shape] = valid_prob[n]
        return action

    def get_dispatch_action(self, env, state, context):
        curr_s = self.stateprocessor.utility_conver_states(state)
        normalized_curr_s = self.stateprocessor.utility_normalize_states(curr_s, len(self.world))
        s_grid = self.stateprocessor.to_grid_states(normalized_curr_s, self.env.city_time)  # t0, s0

        context22 = self.stateprocessor.compute_context(context)
        aa = self.q_estimator.action(s_grid, context22, self.params["epsilon"])
        dispatch_action = aa[0]
        return dispatch_action

    def predict(self, state, info):
        '''
        Return action for state. State space and action space should be the same for testing and training env.
        '''
        curr_state, oldstyle_info, income_mat = self.observation_to_old_fashioned_info(state, info)
        context = self.stateprocessor.compute_context(oldstyle_info)
        curr_s = self.stateprocessor.utility_conver_states(curr_state) # [cars customers] flattened
        s_grid = self.stateprocessor.to_grid_states(curr_s, self.env.time, income_mat)

        action_tuple, valid_action_prob_mat, policy_state, action_choosen_mat, \
            curr_state_value, curr_neighbor_mask, next_state_ids = self.q_estimator.action(s_grid, context, self.params["epsilon"])
        new_action = self.action_from_valid_prob(valid_action_prob_mat)
        return new_action

    def load(self):
        pass # refactor eventually

    def save(self):
        pass # refactor eventually
