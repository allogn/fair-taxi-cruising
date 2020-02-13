import os, sys, random, time
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import logging
import networkx as nx
from collections import defaultdict

from objects import *
from utilities import *


class CityReal:

    def __init__(self,
                    world_graph,
                    idle_driver_locations,
                    n_intervals=96,
                    wc=0,
                    maxincome = None,
                    order_num_dist = None,
                    idle_driver_dist_time = None,
                    order_time_dist = None,
                    order_price_dist = None,
                    l_max = None,
                    probability=1.0/28,
                    real_orders=None,
                    onoff_driver_locations=None,
                    global_flag="global",
                    time_interval=15
                    ):
        '''
        Now we will in each detail find the difference between our simulator and original simulator.

        1. This simulator never generates orders. Synthetic orders are produced by Generator class
        2. This simulator works with graphs instead of a grid with disabled cells


        Unknown parameters:
            - order_num_dist : 144 [{node_id1: [mu, std]}, {node_id2: [mu, std]}, ..., {node_idn: [mu, std]}]
                distribution of orders per time
            - idle_driver_dist_time [[mu1, std1], [mu2, std2], ..., [mu144, std144]] mean and variance of idle drivers in
            the city at each time
            - l_max: The max-duration of an order


        @input:
         - idle_driver_locations (previously idle_driver_location_mat): np array of shape < time , cells >, where time within a day
         - onoff_driver_locations: array of < time , cells, 2 >, where last dimention is [mu, sigma] of distribution, and time within a day
         - real_orders: list of < origin grid, destination grid, start time, duration, price >
         - world_graph: undirected nx graph

         Notes:
         - Those cars who stay also pay wc cost
        '''

        self.log = {}
        self.world = world_graph
        self.drivers = {}  # driver[driver_id] = driver_instance  , driver_id start from 0
        self.n_drivers = 0  # total idle number of drivers. online and not on service.
        self.n_offline_drivers = 0  # total number of offline drivers.
        self.city_time = 0
        self.n_intervals = n_intervals
        self.order_response_rate = 0
        self.wc = wc
        self.total_reward = 0
        if maxincome is not None:
            self.maxincome = maxincome
        else:
            self.maxincome = 10**9

        self.RANDOM_SEED = 0

        self.idle_driver_locations = idle_driver_locations # number of drivers to exist at each moment of time
        assert idle_driver_locations.shape == (n_intervals, len(world_graph)), "{}, {}, {}".format(idle_driver_locations.shape, n_intervals, len(world_graph))

        self.onoff_driver_locations = onoff_driver_locations # number of drivers to appear at each moment of time
        assert onoff_driver_locations.shape == (n_intervals, len(world_graph), 2), "{}, {}, {}".format(onoff_driver_locations.shape, n_intervals, len(world_graph))

        orders_by_day = defaultdict(lambda: [])
        for o in real_orders:
            orders_by_day[o[2]].append(o)
        self.real_orders_by_day = orders_by_day
        # this removed, because we dont consider orders that start outside world
        # self.out_grid_in_orders = np.zeros((self.n_intervals, len(self.world)))

        self.weights_layers_neighbors = [1.0] # else are zero

        for n in self.world.nodes():
            self.world.nodes[n]['o'] = Node(n)
            # self.world[n]['o'].get_layers_neighbors(self.l_max, self.M, self.N, self) -- do not consider so far

    def get_observation(self):
        next_state = np.zeros((2, len(self.world)))
        for node_id, node_data in self.world.nodes(data=True):
            next_state[0, node_id] = node_data['o'].idle_driver_num
            next_state[1, node_id] = node_data['o'].order_num
        return next_state

    def get_num_idle_drivers(self):
        return sum([node['o'].idle_driver_num for _node, node in self.world.nodes(data=True)])


    def reset_randomseed(self, random_seed):
        self.RANDOM_SEED = random_seed

    def reset_clean(self, generate_order=1, ratio=1, city_time=0):
        """ 1. bootstrap oneday's order data.
            2. clean current drivers and orders, regenerate new orders and drivers.
            can reset anytime
        :return:
        """
        self.city_time = city_time

        # clean orders and drivers
        self.drivers = {}  # driver[driver_id] = driver_instance  , driver_id start from 0
        self.n_drivers = 0  # total idle number of drivers. online and not on service.
        self.n_offline_drivers = 0  # total number of offline drivers.
        for nid, node in self.world.nodes(data=True):
            node['o'].clean_node()

        # Init orders of current time step
        self.step_bootstrap_order_real(self.real_orders_by_day[self.city_time])

        # Init current driver distribution
        num_idle_driver = self.utility_get_n_idle_drivers_nodewise()
        self.step_driver_online_offline_control_new(num_idle_driver)
        state = self.get_observation()

        assert np.sum(state[1,:]) == np.sum([len(self.world.nodes[n]['o'].orders) for n in self.world.nodes()])

        return state

    def utility_collect_offline_drivers_id(self):
        """count how many drivers are offline
        :return: offline_drivers: a list of offline driver id
        """
        count = 0 # offline driver num
        offline_drivers = []   # record offline driver id
        for key, _driver in self.drivers.items():
            if _driver.online is False:
                count += 1
                offline_drivers.append(_driver.get_driver_id())
        return offline_drivers

    def utility_get_n_idle_drivers_nodewise(self):
        """ compute idle drivers.
        :return:
        """
        time = self.city_time % self.n_intervals
        idle_driver_num = np.sum(self.idle_driver_locations[time])
        return int(idle_driver_num)

    def step_driver_online_offline_control_new(self, n_idle_drivers):
        """ control the online offline status of drivers

        :param n_idle_drivers: the number of idle drivers expected at current moment
        :return:
        """

        offline_drivers = self.utility_collect_offline_drivers_id()
        self.n_offline_drivers = len(offline_drivers)

        if n_idle_drivers > self.n_drivers:
            self.utility_add_driver_real_new_offlinefirst(n_idle_drivers - self.n_drivers)

        elif n_idle_drivers < self.n_drivers:
            self.utility_set_drivers_offline_real_new(self.n_drivers - n_idle_drivers)
        else:
            pass

    def utility_add_driver_real_new_offlinefirst(self, num_added_driver):
        '''
        old version: add proportionally to idle_driver_locations
        new version: add exactly that amount
        '''

        for n in self.world.nodes():
            node = self.world.nodes[n]['o']
            for i in range(int(self.idle_driver_locations[self.city_time % self.n_intervals, n])):
                n_total_drivers = len(self.drivers.keys())
                added_driver_id = n_total_drivers
                self.drivers[added_driver_id] = Driver(added_driver_id)
                self.drivers[added_driver_id].set_position(node)
                node.add_driver(added_driver_id, self.drivers[added_driver_id])
                self.n_drivers += 1
        return

        curr_idle_driver_distribution = self.get_observation()[0]
        idle_driver_distribution = self.idle_driver_locations[self.city_time % self.n_intervals, :]
        idle_diff = idle_driver_distribution.astype(int) - curr_idle_driver_distribution
        idle_diff[np.where(idle_diff <= 0)] = 0

        if float(np.sum(idle_diff)) == 0:
            return
        np.random.seed(self.RANDOM_SEED)
        node_ids = np.random.choice(len(self.world), size=[num_added_driver], p=idle_diff/float(np.sum(idle_diff)))

        for ii, node_id in enumerate(node_ids):
            node = self.world.nodes[node_id]['o']
            if node.offline_driver_num > 0:
                node.set_offline_driver_online()
                self.n_drivers += 1
                self.n_offline_drivers -= 1
            else:
                n_total_drivers = len(self.drivers.keys())
                added_driver_id = n_total_drivers
                self.drivers[added_driver_id] = Driver(added_driver_id)
                self.drivers[added_driver_id].set_position(node)
                node.add_driver(added_driver_id, self.drivers[added_driver_id])
                self.n_drivers += 1

    def utility_set_drivers_offline_real_new(self, n_drivers_to_off):


        curr_idle_driver_distribution = self.get_observation()[0]
        curr_idle_driver_distribution_resort = np.array([int(curr_idle_driver_distribution.flatten()[index])
                                                         for index in self.target_node_ids])

        # historical idle driver distribution
        idle_driver_distribution = self.idle_driver_locations[self.city_time % self.n_intervals, :]

        # diff of curr idle driver distribution and history
        idle_diff = curr_idle_driver_distribution_resort - idle_driver_distribution.astype(int)
        idle_diff[np.where(idle_diff <= 0)] = 0

        n_drivers_can_be_off = int(np.sum(curr_idle_driver_distribution_resort[np.where(idle_diff >= 0)]))
        if n_drivers_to_off > n_drivers_can_be_off:
            n_drivers_to_off = n_drivers_can_be_off

        sum_idle_diff = np.sum(idle_diff)
        if sum_idle_diff == 0:

            return
        np.random.seed(self.RANDOM_SEED)
        node_ids = np.random.choice(self.target_node_ids, size=[n_drivers_to_off],
                                    p=idle_diff / float(sum_idle_diff))

        for ii, node_id in enumerate(node_ids):
            if self.nodes[node_id].idle_driver_num > 0:
                self.nodes[node_id].set_idle_driver_offline_random()
                self.n_drivers -= 1
                self.n_offline_drivers += 1
                n_drivers_to_off -= 1

    def step_bootstrap_order_real(self, day_orders_t):
        for iorder in day_orders_t:
            start_node_id = iorder[0]
            end_node_id = iorder[1]
            start_node = self.world.nodes[start_node_id]['o']

            if end_node_id in self.world:
                end_node = self.world.nodes[end_node_id]['o']
            else:
                end_node = None
            start_node.add_order_real(self.city_time, end_node, iorder[3], iorder[4])

    def step_assign_order_broadcast_neighbor_reward_update(self):
        """ Consider the orders whose destination or origin is not in the target region
        :param num_layers:
        :param weights_layers_neighbors: [1, 0.5, 0.25, 0.125]
        :return:
        """

        node_reward = np.zeros((len(self.world)))
        # neighbor_reward = np.zeros((len(self.world)))
        # First round broadcast
        reward = 0  # R_{t+1}
        all_order_num = 0
        finished_order_num = 0
        # print("distributing revenue")
        for node_id, node in self.world.nodes(data=True):
            reward_node, all_order_num_node, finished_order_num_node = node['o'].simple_order_assign_real(self.city_time, self, self.wc, self.maxincome)
            reward += reward_node
            all_order_num += all_order_num_node
            finished_order_num += finished_order_num_node
            node_reward[node_id] += reward_node

        # Second round broadcast
        # for node in self.nodes:
        #     if node is not None:
        #         if node.order_num != 0:
        #             reward_node_broadcast, finished_order_num_node_broadcast \
        #                 = node.simple_order_assign_broadcast_update(self, neighbor_reward)
        #             reward += reward_node_broadcast
        #             finished_order_num += finished_order_num_node_broadcast

        node_reward = node_reward #+ neighbor_reward
        if all_order_num != 0:
            self.order_response_rate = finished_order_num/float(all_order_num)
        else:
            self.order_response_rate = -1
        # print(reward)
        return reward, [node_reward, None] #neighbor_reward]

    def step_remove_unfinished_orders(self):
        for node_id, node in self.world.nodes(data=True):
            node['o'].remove_unfinished_order(self.city_time)

    def step_pre_order_assigin(self, next_state):

        remain_drivers = next_state[0] - next_state[1]
        remain_drivers[remain_drivers < 0] = 0

        remain_orders = next_state[1] - next_state[0]
        remain_orders[remain_orders < 0] = 0

        if np.sum(remain_orders) == 0 or np.sum(remain_drivers) == 0:
            context = np.array([remain_drivers, remain_orders])
            return context

        remain_orders_1d = remain_orders.flatten()
        remain_drivers_1d = remain_drivers.flatten()

        ###  do not estimate neighbors so far
        # for node in self.nodes:
        #     if node is not None:
        #         curr_node_id = node.get_node_index()
        #         if remain_orders_1d[curr_node_id] != 0:
        #             for neighbor_node in node.neighbors:
        #                 if neighbor_node is not None:
        #                     neighbor_id = neighbor_node.get_node_index()
        #                     a = remain_orders_1d[curr_node_id]
        #                     b = remain_drivers_1d[neighbor_id]
        #                     remain_orders_1d[curr_node_id] = max(a-b, 0)
        #                     remain_drivers_1d[neighbor_id] = max(b-a, 0)
        #                 if remain_orders_1d[curr_node_id] == 0:
        #                     break

        context = np.array([remain_drivers_1d, remain_orders_1d])
        return context

    def step_dispatch_invalid(self, dispatch_actions):
        '''
        actions is a list of (from, to, number). tuples are unique by (from, to)
	cars always arrive to the existing grid. moving out of grid is not supported anymore.
        '''
        save_remove_id = []
        considered_actions = set()
        wc_costs = {}
        init_drivers = {} # number of drivers per node before dispatching
        for c in self.world.nodes():
            wc_costs[c] = 0
            init_drivers[c] = len(self.world.nodes[c]['o'].drivers)

        for action in dispatch_actions:

            start_node_id, end_node_id, num_of_drivers = action
            assert (start_node_id, end_node_id) not in considered_actions, "Actions contain duplicates"
            considered_actions.add((start_node_id, end_node_id))
            assert(num_of_drivers > 0)
            assert self.world.has_edge(start_node_id, end_node_id), "Actions are invalid: {}->{}".format(start_node_id, end_node_id)

            start_node = self.world.nodes[start_node_id]['o']
            if start_node.get_driver_numbers() < num_of_drivers:
                raise Exception("Can not dispatch more drivers ({}) than a node has ({}).".format(num_of_drivers, start_node.get_driver_numbers())) # although in original code such thing is possible?

            for _ in np.arange(num_of_drivers):
                remove_driver_id = start_node.remove_idle_driver_random()
                driver_inc_before = min([self.maxincome, self.drivers[remove_driver_id].reward])
                self.drivers[remove_driver_id].reward -= self.wc
                driver_inc_after = min([self.maxincome, self.drivers[remove_driver_id].reward])
                wc_costs[start_node_id] += -(driver_inc_after - driver_inc_before)

                save_remove_id.append((end_node_id, remove_driver_id))
                self.drivers[remove_driver_id].set_position(None)
                self.drivers[remove_driver_id].set_offline_for_start_dispatch()
                self.n_drivers -= 1

        # substract wc cost from non-moving cars
        for n in self.world.nodes():
            node = self.world.nodes[n]['o']
            number_of_customers = len(node.orders)
            number_of_cars = len(node.drivers)
            if number_of_customers < number_of_cars:
                wc_costs[n] += (number_of_cars - number_of_customers) * self.wc

            # we shouldn't move more drivers than necessary by excess
            assert (number_of_cars >= len(node.orders)) or (init_drivers[n] == number_of_cars), "Node {}: cars after {} and before {} dispatching, orders {}".format(n, number_of_cars, init_drivers[n], len(node.orders))

        return save_remove_id, wc_costs

    def step_add_dispatched_drivers(self, save_remove_id):
        # drivers dispatched at t, arrived at t + 1
        for destination_node_id, arrive_driver_id in save_remove_id:
            self.drivers[arrive_driver_id].set_position(self.world.nodes[destination_node_id]['o'])
            self.drivers[arrive_driver_id].set_online_for_finish_dispatch()
            self.world.nodes[destination_node_id]['o'].add_driver(arrive_driver_id, self.drivers[arrive_driver_id])
            self.n_drivers += 1

    def step_increase_city_time(self):
        self.city_time += 1
        # set city time of drivers
        for driver_id, driver in self.drivers.items():
            driver.set_city_time(self.city_time)


    def step_driver_status_control(self):
        # Deal with orders finished at time T=1, check driver status. finish order, set back to off service
        for key, _driver in self.drivers.items():
            _driver.status_control_eachtime(self)


    def step_driver_online_offline_nodewise(self):
        """ node wise control driver online offline
        :return:
        """
        moment = self.city_time % self.n_intervals
        curr_onoff_distribution = self.onoff_driver_locations[moment]

        self.all_grids_on_number = 0
        self.all_grids_off_number = 0
        for idx, target_node_id in enumerate(self.world.nodes()):
            curr_mu    = curr_onoff_distribution[idx, 0]
            curr_sigma = curr_onoff_distribution[idx, 1]
            on_off_number = np.round(np.random.normal(curr_mu, curr_sigma, 1)[0]).astype(int)

            if on_off_number > 0:
                self.utility_add_driver_real_nodewise(target_node_id, on_off_number)
                self.all_grids_on_number += on_off_number
            elif on_off_number < 0:
                self.utility_set_drivers_offline_real_nodewise(target_node_id, abs(on_off_number))
            else:
                pass

    def utility_add_driver_real_nodewise(self, node_id, num_added_driver):
        node = self.world.nodes[node_id]['o']
        while num_added_driver > 0:
            if node.offline_driver_num > 0:
                node.set_offline_driver_online()
                self.n_drivers += 1
                self.n_offline_drivers -= 1
            else:
                # print("Adding new driver")
                n_total_drivers = len(self.drivers.keys())
                added_driver_id = n_total_drivers
                self.drivers[added_driver_id] = Driver(added_driver_id)
                self.drivers[added_driver_id].set_position(node)
                node.add_driver(added_driver_id, self.drivers[added_driver_id])
                self.n_drivers += 1
            num_added_driver -= 1

    def utility_set_drivers_offline_real_nodewise(self, node_id, n_drivers_to_off):
        node = self.world.nodes[node_id]['o']
        while n_drivers_to_off > 0:
            if node.idle_driver_num > 0:
                node.set_idle_driver_offline_random()
                self.n_drivers -= 1
                self.n_offline_drivers += 1
                n_drivers_to_off -= 1
                self.all_grids_off_number += 1
            else:
                break

    def step(self, dispatch_actions, fair=False):
        '''
        Dispatch actions may not include all drivers, as some drivers may stay at the same cell.
        Walking cost should include only drivers that are relocated to other cells.

        @input:
            - dispatch_actions is a list of < start_node_id, end_node_id, num_of_drivers >

        @output:
            - next_state is an array of < distribution of cars, distribution of customers >
            - reward is a single number indicating reward per step (contribution to objective)
            - info is list of two elements,
                - first is reward node: an array of < array of rewards per node, None (deprecated) >
                - second is context : an array of difference between customers and drivers
        '''


        info = []

        # Loop over all dispatch action, change the driver distribution
        save_remove_id, wc_costs = self.step_dispatch_invalid(dispatch_actions)

        reward, reward_node = self.step_assign_order_broadcast_neighbor_reward_update()
        reward -= np.sum([wc_costs[k] for k in wc_costs])
        for start_node_id, end_node_id, num_of_drivers in dispatch_actions:
            reward_node[0][start_node_id] -= wc_costs[start_node_id]
        self.total_reward += reward

        # increase city time t + 1
        self.step_increase_city_time()
        self.step_driver_status_control()  # drivers finish order become available again.

        # drivers dispatched at t, arrived at t + 1, become available at t+1
        self.step_add_dispatched_drivers(save_remove_id)

        # generate orders for the next time interval. Here, we never generate order, only load real data or synthetic data.
        self.step_bootstrap_order_real(self.real_orders_by_day[self.city_time])

        # offline online control;
        self.step_driver_online_offline_nodewise()

        self.step_remove_unfinished_orders()
        # get states S_{t+1}  [driver_dist, order_dist]
        next_state = self.get_observation()

        context = self.step_pre_order_assigin(next_state)
        info = [reward_node, context]
        return next_state, reward, info


    def get_min_revenue(self):
        return float(np.min([self.drivers[d].get_income() for d in self.drivers]))

    def get_total_revenue(self):
        return self.total_reward
