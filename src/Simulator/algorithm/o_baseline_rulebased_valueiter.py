# -*- coding: utf-8 -*-
from alg_utility import *
from collections import namedtuple
import scipy.signal as sci
from random import randint
import numpy as np
from copy import deepcopy

class ValueIter:

    def __init__(self, value_table, env, alpha, gamma):

        self.value_table = deepcopy(value_table)  # 144 x [n_grid]
        self.M = env.M
        self.N = env.N
        self.n_grid = env.n_valid_grids
        self.env = env
        self.alpha = alpha
        self.gamma = gamma

        # compute the indices of neighbors of each grid in curr_state,
        self.neighbors_list = []
        for idx, node_id in enumerate(env.target_grids):
            neighbor_indices = env.nodes[node_id].layers_neighbors_id[0]  # index in env.nodes
            neighbor_ids = [env.target_grids.index(env.nodes[item].get_node_index()) for item in neighbor_indices]
            neighbor_ids.append(idx)
            # index in env.target_grids == index in state
            self.neighbors_list.append(neighbor_ids)

    def action(self, curr_state, city_time):
        """ Multinomial sample the action according to the value in its neighbor and
        itself in the value table
        :param curr_state: [driver_dist, order_dist] one sample
        :param city_time: current city time
        :return:
        """

        # context = curr_state[:self.n_grid] - curr_state[self.n_grid:]  # idle driver - order number
        context = curr_state

        action_tuple = []
        curr_value = self.value_table[(city_time + 1) % 144]

        for idx, grid_context in enumerate(context):
            if grid_context <= 0:
                continue
            start_node_ii = idx
            start_node_id = self.env.target_grids[start_node_ii]

            neighbor_iis = self.neighbors_list[idx]

            temp_values = deepcopy(curr_value[neighbor_iis])
            temp_values[temp_values - temp_values[-1] < 0] = 0
            if np.sum(temp_values) == 0:
                continue
            actions_idx = np.random.multinomial(int(grid_context), temp_values/float(np.sum(temp_values)))

            for jdx, action_num in enumerate(actions_idx):  
                end_node_id = self.env.target_grids[neighbor_iis[jdx]]
                assert action_num >= 0
                if action_num == 0 or end_node_id == start_node_id:
                    continue
                action_tuple.append((start_node_id, end_node_id, action_num))

        return action_tuple

    def value_iterate_updates(self, nodes_reward, city_time):
        """ update value table via city_time + 1 reward

        :param nodes_reward: n_valid_grid x 1 list []
        :param city_time: current city time
        :return:
        """
        curr_time = city_time % 144
        next_time = (city_time + 1) % 144
        curr_value = self.value_table[curr_time]
        next_value = self.value_table[next_time]

        new_table_value = np.zeros(self.n_grid)
        for idx, node_reward in enumerate(nodes_reward):
            neighbor_iis = self.neighbors_list[idx]
            temp_values = deepcopy(curr_value[neighbor_iis])
            temp_values[temp_values - temp_values[-1] < 0] = 0
            if np.sum(temp_values) == 0:
                continue
            probability_values = temp_values/float(np.sum(temp_values))
            new_table_value[idx] = np.sum(probability_values * (nodes_reward[neighbor_iis]
                                                                + self.gamma * next_value[neighbor_iis]))

        self.value_table[curr_time] = new_table_value

    # prev_value[idx] += self.alpha * (node_reward + self.gamma * curr_value[idx] - prev_value[idx])