import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from TestingSolver import *

from copy import deepcopy
from random import randint
import numpy as np
from collections import namedtuple
from tqdm import tqdm
import time

class ValIterSolver(TestingSolver):
    def __init__(self, **params):
        super().__init__(**params)
        self.batch_update = False

    def get_dispatch_action(self, env, state, context):
        dpath = self.params['dataset']["dataset_path"]
        time_periods = self.params['dataset']["time_periods"]

        action_tuple = []
        curr_value = self.dyn_value_table[env.city_time % env.n_intervals]
        for start_node_ii, grid_context in enumerate(context[0]): # number of drivers per cell
            if grid_context <= 0:
                continue
            neighbor_iis = list(env.world.neighbors(start_node_ii)) + [start_node_ii]

            temp_values = deepcopy(curr_value[neighbor_iis])
            # temp_values[temp_values - temp_values[-1] < 0] = 0 # better stay than go in a worst place
            p = np.exp(temp_values)/sum(np.exp(temp_values))

            # p[p - p[-1] < 0] = 0
            # p = p/np.sum(p)

            # t = np.min(temp_values)
            # if t < 0:
            #     temp_values += -t

            # if np.sum(temp_values) == 0:
            #     p = np.ones(temp_values.shape) / temp_values.shape[0]
            # else:
            #     p = temp_values/np.sum(temp_values)
            actions_idx = np.random.multinomial(int(grid_context), p)
            for jdx, action_num in enumerate(actions_idx):
                end_node_id = neighbor_iis[jdx]
                assert action_num >= 0
                if action_num == 0 or start_node_ii == end_node_id:
                    continue
                action_tuple.append((start_node_ii, end_node_id, action_num))

        return action_tuple


    def value_iterate_updates(self, world, nodes_reward, time_period, time_periods):
        curr_time = time_period
        next_time = (curr_time + 1) % time_periods
        curr_value = self.dyn_value_table[curr_time]
        next_value = self.dyn_value_table[next_time]

        new_table_value = np.zeros(len(nodes_reward))
        # print("target values per cell")
        for idx, node_reward in enumerate(nodes_reward):
            neighbor_iis = list(world.neighbors(idx)) + [idx]
            temp_values = deepcopy(curr_value[neighbor_iis])
            temp_values[temp_values - temp_values[-1] < 0] = 0

            probability_values = np.exp(temp_values)/sum(np.exp(temp_values))

            probability_values[probability_values - probability_values[-1] < 0] = 0
            probability_values = probability_values/np.sum(probability_values)

            # t = np.min(temp_values)
            # if t < 0:
            #     temp_values += -t
            #
            # if np.sum(temp_values) == 0:
            #     probability_values = np.ones(temp_values.shape) / temp_values.shape[0]
            # else:
            #     probability_values = temp_values/float(np.sum(temp_values))
            # print("prob vals",probability_values)
            # print(idx,probability_values * (nodes_reward[neighbor_iis] - self.params['wc']
            #                                                     + self.params['gamma'] * next_value[neighbor_iis]))
            new_table_value[idx] = np.sum(probability_values * (nodes_reward[neighbor_iis] - self.params['wc']
                                                                + self.params['gamma'] * next_value[neighbor_iis]))
            # print("neigh rewards", nodes_reward[neighbor_iis])
            # print("next val", next_value[neighbor_iis])

        self.dyn_value_table[curr_time] = new_table_value
        # print("value table", new_table_value)

    def get_node_value(self, time, node):
        return self.dyn_value_table[time][node]

    def train(self, maxincome = None):
        t1 = time.time()

        dpath = self.params['dataset']["dataset_path"]
        time_periods = self.params['dataset']["time_periods"]

        world = nx.read_gpickle(os.path.join(dpath, "world.pkl"))
        real_orders, idle_driver_locations, onoff_driver_locations = self.get_train_data()

        env = CityReal(world, idle_driver_locations, real_orders=real_orders,
                        onoff_driver_locations=onoff_driver_locations, n_intervals=time_periods, wc=self.params["wc"], maxincome=maxincome)

        self.dyn_value_table = np.zeros((self.time_periods, len(world)))

        total_train_days = self.first_test_day
        if self.verbose:
            pbar = tqdm(total=total_train_days, desc="Training ValIter Solver Iterations")

        env.reset_randomseed(self.random_seed)
        total_rewards = []
        for day in np.arange(total_train_days):
            state = env.reset_clean(city_time=0*day*self.time_periods)
            context = env.step_pre_order_assigin(state)

            order_response_rates = []
            total_reward = 0
            immidiate_rewards = []
            for T in range(time_periods):
                dispatch_action = self.get_dispatch_action(env, state, context)
                state, reward, info = env.step(dispatch_action)
                total_reward += reward
                context = info[1]
                nodes_reward = info[0][0]
                immidiate_rewards.append(nodes_reward)
                if not self.batch_update:
                    self.value_iterate_updates(env.world, nodes_reward, T, time_periods)
            if self.batch_update:
                for t in range(len(immidiate_rewards))[::-1]:
                    nodes_reward = immidiate_rewards[t]
                    self.value_iterate_updates(env.world, nodes_reward, t, time_periods)

            total_rewards.append(total_reward)
            # print(self.dyn_value_table[-1])
            if self.verbose:
                pbar.update()

        if self.verbose:
            pbar.close()


        self.log['train_time'] = time.time() - t1
        self.log['train_rewards'] = total_rewards

    def load(self):
        with open(os.path.join(self.dpath, "trained_ValIter.pkl"), "rb") as f:
            self.dyn_value_table = pkl.load(f)

    def save(self):
        with open(os.path.join(self.dpath, "trained_ValIter.pkl"), "wb") as f:
            pkl.dump(self.dyn_value_table, f)
