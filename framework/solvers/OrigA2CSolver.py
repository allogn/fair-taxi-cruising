import os, sys
import pickle as pkl
import logging
from framework.solvers.OrigSolver import OrigSolver
from framework.solvers.cA2C.ocA2C import *

class OrigA2CSolver(OrigSolver):
    def __init__(self, **params):
        super().__init__(**params)
        self.dpath = self.params['dataset']["dataset_path"]

    def test(self):
        pass

    @staticmethod
    def compute_context(target_grids, info):

        context = info.flatten()
        context = [context[idx] for idx in target_grids]
        return context

    def train(self):
        temp = np.array(self.env.target_grids) + self.env.M * self.env.N
        target_id_states = self.env.target_grids + temp.tolist()

        curr_s = np.array(self.env.reset_clean()).flatten()  # [0] driver dist; [1] order dist
        curr_s = utility_conver_states(curr_s, target_id_states)

        MAX_ITER = 50
        is_plot_figure = False
        city_time_start = 0
        EP_LEN = 144

        city_time_end = city_time_start + EP_LEN
        epsilon = 0.5
        gamma = 0.9
        learning_rate = 1e-3

        prev_epsiode_reward = 0

        all_rewards = []
        order_response_rate_episode = []
        value_table_sum = []
        episode_rewards = []
        episode_conflicts_drivers = []
        record_all_order_response_rate = []

        T = 144
        action_dim = 7
        state_dim = self.env.n_valid_grids * 3 + T


        # tf.reset_default_graph()
        sess = tf.Session()
        tf.set_random_seed(1)
        q_estimator = Estimator(sess, action_dim,
                                state_dim, self.env, scope="q_estimator", summaries_dir=self.dpath)


        sess.run(tf.global_variables_initializer())

        replay = ReplayMemory(memory_size=1e+6, batch_size=int(3e+3))
        policy_replay = policyReplayMemory(memory_size=1e+6, batch_size=int(3e+3))
        stateprocessor = stateProcessor(target_id_states, self.env.target_grids, self.env.n_valid_grids)
        self.stateprocessor = stateprocessor

        restore = True
        saver = tf.train.Saver()


        # record_curr_state = []
        # record_actions = []
        save_random_seed = []
        episode_dispatched_drivers = []
        global_step1 = 0
        global_step2 = 0
        RATIO = 1
        for n_iter in np.arange(self.params["iterations"]):
            RANDOM_SEED = n_iter + MAX_ITER - 10
            self.env.reset_randomseed(RANDOM_SEED)
            save_random_seed.append(RANDOM_SEED)
            batch_s, batch_a, batch_r = [], [], []
            batch_reward_gmv = []
            epsiode_reward = 0
            num_dispatched_drivers = 0

            # reset self.env
            is_regenerate_order = 0
            curr_state = self.env.reset_clean(generate_order=is_regenerate_order, ratio=RATIO, city_time=city_time_start)
            info = self.env.step_pre_order_assigin(curr_state)
            context = stateprocessor.compute_context(info)
            curr_s = stateprocessor.utility_conver_states(curr_state)
            normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
            s_grid = stateprocessor.to_grid_states(normalized_curr_s, self.env.city_time)  # t0, s0

            # record rewards to update the value table
            episodes_immediate_rewards = []
            num_conflicts_drivers = []
            curr_num_actions = []
            order_response_rates = []
            for ii in np.arange(EP_LEN):
                # record_curr_state.append(curr_state)
                # INPUT: state,  OUTPUT: action
                action_tuple, valid_action_prob_mat, policy_state, action_choosen_mat, \
                curr_state_value, curr_neighbor_mask, next_state_ids = q_estimator.action(s_grid, context, epsilon)
                # a0

                # ONE STEP: r0
                next_state, r, info = self.env.step(action_tuple, 2)

                # r0
                immediate_reward = stateprocessor.reward_wrapper(info, curr_s)

                # Save transition to replay memory
                if ii != 0:
                    # r1, c0
                    r_grid = stateprocessor.to_grid_rewards(immediate_reward)
                    # s0, a0, r1  for value newtwork
                    targets_batch = q_estimator.compute_targets(action_mat_prev, s_grid, r_grid, gamma)

                    # advantage for policy network.
                    advantage = q_estimator.compute_advantage(curr_state_value_prev, next_state_ids_prev,
                                                              s_grid, r_grid, gamma)

                    replay.add(state_mat_prev, action_mat_prev, targets_batch, s_grid)
                    policy_replay.add(policy_state_prev, action_choosen_mat_prev, advantage, curr_neighbor_mask_prev)

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
                curr_s = stateprocessor.utility_conver_states(next_state)
                normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
                s_grid = stateprocessor.to_grid_states(normalized_curr_s, self.env.city_time)  # t0, s0

                # c1
                context = stateprocessor.compute_context(info[1])

                # training method 1.
                # #    # Sample a minibatch from the replay memory and update q network
                # if replay.curr_lens != 0:
                #     # update policy network
                #     for _ in np.arange(30):
                #         batch_s, batch_a, batch_r, batch_mask = policy_replay.sample()
                #         q_estimator.update_policy(batch_s, batch_r.reshape([-1, 1]), batch_a, batch_mask, learning_rate,
                #                                   global_step2)
                #         global_step2 += 1

                # Perform gradient descent update
                # book keeping
                all_rewards.append(r)
                batch_reward_gmv.append(r)
                order_response_rates.append(self.env.order_response_rate)
                curr_num_action = np.sum([aa[2] for aa in action_tuple]) if len(action_tuple) != 0 else 0
                curr_num_actions.append(curr_num_action)
                num_conflicts_drivers.append(collision_action(action_tuple))

            episode_reward = np.sum(batch_reward_gmv[1:])
            episode_rewards.append(episode_reward)
            n_iter_order_response_rate = np.mean(order_response_rates[1:])
            order_response_rate_episode.append(n_iter_order_response_rate)
            record_all_order_response_rate.append(order_response_rates)
            episode_conflicts_drivers.append(np.sum(num_conflicts_drivers[:-1]))
            episode_dispatched_drivers.append(np.sum(curr_num_actions[:-1]))

            logging.info("******** iteration {} ********* reward {}, order_response_rate {} number drivers {}, conflicts {}".format(n_iter, episode_reward,
                                                                                                               n_iter_order_response_rate,
                                                                                                             episode_dispatched_drivers[-1],
                                                                                                            episode_conflicts_drivers[-1]))

            with open(os.path.join(self.dpath, "results.pkl"), "wb") as f:
                pkl.dump([episode_rewards, order_response_rate_episode, save_random_seed, episode_conflicts_drivers,
                                     episode_dispatched_drivers], f)
            if n_iter == 24:
                break

            # update value network
            for _ in np.arange(4000):
                batch_s, _, batch_r, _ = replay.sample()
                iloss = q_estimator.update_value(batch_s, batch_r, 1e-3, global_step1)
                global_step1 += 1

            # training method 2
            # update policy network
            for _ in np.arange(4000):
                batch_s, batch_a, batch_r, batch_mask = policy_replay.sample()
                q_estimator.update_policy(batch_s, batch_r.reshape([-1, 1]), batch_a, batch_mask, learning_rate,
                                          global_step2)
                global_step2 += 1



            saver.save(sess, os.path.join(self.dpath,"model.ckpt"))
            if RANDOM_SEED == 54:
                saver.save(sess, os.path.join(self.dpath,"model_before_testing.ckpt"))

    def load(self):
        pass # refactor eventually

    def save(self):
        pass # refactor eventually
