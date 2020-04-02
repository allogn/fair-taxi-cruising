import pickle as pkl
import numpy as np
import os, sys
import logging
from framework.solvers.OrigSolver import OrigSolver

class OrigNoSolver(OrigSolver):
    def __init__(self, **params):
        super().__init__(**params)

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

        MAX_ITER = 50  # 10 iteration the Q-learning loss will converge.
        is_plot_figure = False
        city_time_start = 0
        EP_LEN = 144
        assert self.params['dataset']['time_periods'] == 144, "Only 144 time periods supported originally"
        global_step = 0
        city_time_end = city_time_start + EP_LEN
        epsilon = 0.5
        gamma = 0.9
        learning_rate = 1e-3

        prev_epsiode_reward = 0
        curr_num_actions = []
        all_rewards = []
        order_response_rate_episode = []
        value_table_sum = []
        episode_rewards = []
        num_conflicts_drivers = []
        driver_numbers_episode = []
        order_numbers_episode = []

        T = 144
        action_dim = 7
        state_dim = self.env.n_valid_grids * 3 + T

        record_all_order_response_rate = []

        RATIO = 1 # ratio of how many drivers there should be in respect to loaded data

        save_random_seed = []
        episode_avaliables_vehicles = []
        for n_iter in np.arange(10):
            RANDOM_SEED = n_iter + MAX_ITER + 5
            self.env.reset_randomseed(RANDOM_SEED)
            save_random_seed.append(RANDOM_SEED)
            batch_s, batch_a, batch_r = [], [], []
            batch_reward_gmv = []
            epsiode_reward = 0
            num_dispatched_drivers = 0

            driver_numbers = []
            order_numbers = []
            curr_state = self.env.reset_clean(generate_order=0, ratio=RATIO, city_time=city_time_start) # do not generate orders, load them
            driver_numbers.append(np.sum(curr_state[0]))
            order_numbers.append(np.sum(curr_state[1]))
            info = self.env.step_pre_order_assigin(curr_state)
            context = self.compute_context(self.env.target_grids, np.array(info))

            # record rewards to update the value table
            episodes_immediate_rewards = []
            order_response_rates = []
            available_drivers = []
            for ii in np.arange(EP_LEN + 1):
                available_drivers.append(np.sum(context))
                # ONE STEP: r0
                next_state, r, info = self.env.step([], 2)
                driver_numbers.append(np.sum(next_state[0]))
                order_numbers.append(np.sum(next_state[1]))

                context = self.compute_context(self.env.target_grids, np.array(info[1]))
                # Perform gradient descent update
                # book keeping
                global_step += 1
                all_rewards.append(r)
                batch_reward_gmv.append(r)
                order_response_rates.append(self.env.order_response_rate)

            episode_reward = np.sum(batch_reward_gmv[1:])
            episode_rewards.append(episode_reward)
            driver_numbers_episode.append(np.sum(driver_numbers[:-1]))
            order_numbers_episode.append(np.sum(order_numbers[:-1]))
            episode_avaliables_vehicles.append(np.sum(available_drivers[:-1]))
            n_iter_order_response_rate = np.mean(order_response_rates[1:])
            order_response_rate_episode.append(n_iter_order_response_rate)
            record_all_order_response_rate.append(order_response_rates)

            logging.info("******** iteration {} ********* reward {}, order response rate {} available vehicle {}".format(n_iter,
                                                                                                                  episode_reward,
                                                                                                n_iter_order_response_rate,
                                                                                                episode_avaliables_vehicles[-1]))

            with open(os.path.join(self.dpath, "results.pkl"), "wb") as f:
                pkl.dump([episode_rewards, order_response_rate_episode, save_random_seed,
                            driver_numbers_episode, order_numbers_episode, episode_avaliables_vehicles], f)


        logging.info("averaged available vehicles per time step: {}".format(np.mean(episode_avaliables_vehicles)/144.0))

    def load(self):
        pass # refactor eventually

    def save(self):
        pass # refactor eventually
