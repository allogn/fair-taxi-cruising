# -*- coding: utf-8 -*-
from alg_utility import *
from collections import namedtuple
import scipy.signal as sci
from random import randint
import numpy as np
from copy import deepcopy

class expectedSARSA:

    def __init__(self, qvalue_table, env, alpha, gamma, epsilon):

        self.qtable = qvalue_table  # 144 x [n_grid]
        self.M = env.M
        self.N = env.N
        self.n_grid = env.n_valid_grids
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # compute the indices of neighbors of each grid in curr_state
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

        action_valid_grid_tuple = []

        for idx, grid_context in enumerate(context):
            if grid_context <= 0:
                continue
            start_node_ii = idx
            start_node_id = self.env.target_grids[start_node_ii]

            neighbor_iis = self.neighbors_list[idx]

            temp_values = deepcopy(self.qtable[city_time][idx])
            temp_values[temp_values - temp_values[-1] < 0] = 0
            if np.sum(temp_values) == 0:
                continue

            actions_idx = np.random.multinomial(int(grid_context), temp_values/float(np.sum(temp_values)))

            # # epsilon greedy
            # act_dim = len(temp_values[temp_values > 0])   # valid action dimension
            # max_index = np.argmax(temp_values)
            # temp_values[temp_values > 0] = self.epsilon/act_dim
            # temp_values[max_index] += 1 - self.epsilon
            # actions_idx = np.random.multinomial(int(grid_context), temp_values)

            for jdx, action_num in enumerate(actions_idx):
                end_node_ii = neighbor_iis[jdx]
                end_node_id = self.env.target_grids[end_node_ii]
                assert action_num >= 0
                if action_num == 0:
                    continue
                action_valid_grid_tuple.append((start_node_ii, end_node_ii, action_num))
                if end_node_id == start_node_id:
                    continue
                action_tuple.append((start_node_id, end_node_id, action_num))

        return action_tuple, action_valid_grid_tuple

    def value_iterate_updates(self, nodes_reward, action_valid_grid_tuple, city_time):
        """  update value table via city_time + 1 reward

        :param nodes_reward: n_valid_grid x 1 list []
        :param city_time: current city time
        :return:
        """
        curr_time = city_time % 144
        next_time = (city_time + 1) % 144
        # curr_value = self.qtable[curr_time]
        # next_value = self.qtable[next_time]
        # new_table_value = np.zeros(self.n_grid)

        stored_updated_qtable = []
        for idx, one_action in enumerate(action_valid_grid_tuple):
            start_valid_grid_id = one_action[0]
            end_valid_grid_id = one_action[1]
            action_num = one_action[2]

            # temp_alpha = (1 - self.alpha) ** action_num
            neighbor_iis = self.neighbors_list[start_valid_grid_id]
            neighbor_id_in_currnode = neighbor_iis.index(end_valid_grid_id)

            temp_values = deepcopy(self.qtable[next_time][end_valid_grid_id])
            temp_values[temp_values - temp_values[-1] < 0] = 0
            if np.sum(temp_values) == 0:
                continue
            probability_values = temp_values / float(np.sum(temp_values))

            temp_q_sa = self.qtable[curr_time][start_valid_grid_id][neighbor_id_in_currnode]
            temp_q_sa = (1 - self.alpha) * temp_q_sa \
                        + self.alpha * (nodes_reward[end_valid_grid_id]
                                             + self.gamma * np.sum(probability_values * self.qtable[next_time][end_valid_grid_id]))

            stored_updated_qtable.append([start_valid_grid_id, neighbor_id_in_currnode, temp_q_sa])

        for item in stored_updated_qtable:
            xx, yy, new_qvalue = item
            self.qtable[curr_time][xx][yy] = new_qvalue


