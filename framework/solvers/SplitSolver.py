import os, sys
import networkx as nx
from collections import Counter
import math
from hilbertcurve.hilbertcurve import HilbertCurve

from framework.solvers.TestingSolver import *
from framework.solvers.cA2CSolver import *
from framework.solvers.DiffSolver import *

class SplitSolver(TestingSolver):
    def __init__(self, **params):
        super().__init__(**params)
        for n in self.world.nodes(data=True):
            assert 'coords' in n[1], "Graph must contain coords"
        superworld_size = (1. - self.params['shrinking_fraction']) * len(self.world)
        assert superworld_size > 0
        self.superworld = self.build_superworld(self.world, superworld_size)
        
    @staticmethod # bug: not necesarilly connected graph
    def build_superworld(world, new_size):
        p = int(math.log2(len(world))) + 1 # 2^p is number of points along each dimention of the space
        hc = HilbertCurve(p, 2)
        node_seq_coords = [(hc.distance_from_coordinates(n[1]['coords']), n[0]) for n in world.nodes(data=True)]
        node_seq_coords.sort()
        chunks = np.array_split(node_seq_coords, new_size)
        center_nodes = [chunk[len(chunk)//2][1] for chunk in chunks]
        node_mapping = nx.voronoi_cells(world, center_nodes)

        new_nodes = []
        i = 0
        for center_node in center_nodes:
            new_nodes.append(
                (i, {"nodes": node_mapping[center_node]})
            )
            for n in node_mapping[center_node]:
                world.nodes[n]['supernode'] = i
            i += 1
        superworld = nx.Graph() # we always use graph to simplify training, for both world and superworld
        superworld.add_nodes_from(new_nodes)
        # we assume all areas are approximately the same size, so we add edges in superworld of the same length
        edges = set()
        for e in world.edges():
            n1 = world.nodes[e[0]]['supernode'] 
            n2 = world.nodes[e[1]]['supernode']
            if n1 != n2 and (n1, n2) not in edges:
                edges.add((n1, n2))
        superworld.add_edges_from(list(edges))

        return superworld

    @staticmethod
    def get_flow_per_edge(idle_drivers, arriving_drivers, expected_orders, superworld):
        '''
        :param idle_drivers: a vector of idle drivers per supernode in the current time step
        :param arriving_drivers: a vector of drivers that arrive due to customers in the next time step
        :param expected_orders: a vector of number of orders expected in the next time step
        :param superworld: a superworld
        :returns: a dict keyed by [node1][node2] with number of cars to be travelled
        '''
        directed_superworld = nx.DiGraph(superworld)
        idle_drivers = np.array(idle_drivers)
        arriving_drivers = np.array(arriving_drivers)
        expected_orders = np.array(expected_orders)
        assert np.sum(idle_drivers) > 0
        if np.sum(expected_orders) == 0:
            expected_orders = np.ones(expected_orders.shape) # aim for a uniform driver distribution
        expected_orders = expected_orders / np.sum(expected_orders)
        total_drivers_with_arriving = np.sum(idle_drivers + arriving_drivers)

        # distribute drivers according to customers
        required_drivers = np.array([int(n*total_drivers_with_arriving) for n in expected_orders]) # rounds down
        required_drivers -= arriving_drivers # substract those we can not control
        # we might end up with negative values, because too many are arriving at the same place, for example
        # or with positive, becauce of rounding up/down
        # randomly assign or substract excess drivers
        required_drivers[required_drivers < 0] = 0
        total_idle_drivers = np.sum(idle_drivers)
        s = np.sum(required_drivers)
        while total_idle_drivers - s != 0:
            ind = np.nonzero(required_drivers)
            a = np.random.choice(ind[0])
            required_drivers[a] += (total_idle_drivers - s)/np.abs(total_idle_drivers - s)
            s = np.sum(required_drivers)

        # set demand property on superworld graph and solve mincostflow
        nx.set_edge_attributes(directed_superworld, 1, name='weight') # assume approximately equal distances
        demand = required_drivers - idle_drivers # positive demand = inflow
        assert np.sum(demand) == 0
        nx.set_node_attributes(directed_superworld, { i : demand[i] for i in range(len(demand)) }, "demand")
        flow_per_edge_dict = nx.min_cost_flow(directed_superworld, demand='demand', weight='weight')

        return flow_per_edge_dict

    def get_expected_drivers_per_supernode(self):
        '''
        Returns a vector of idle drivers in the next time step. Includes those who will arrive
        due to customers + idle driver distribution in the current step
        '''
        ...

    def get_expected_orders_per_supernode(self):
        '''
        @return 
        '''
        ...

    def train(self, db_save_callback = None):
        self.init_subsolvers()

        for i in range(self.params['iterations']):
            for n in self.superworld.nodes(data=True):
                s = n[1]['solver']
                s.load()
                # train each subsolver using an estimate of average drivers, and adding drivers in random
                self.generate_random_drivers()
                s.train()
                s.save()
            self.test()

        # make sure the model is saved
        self.save()

    def generate_random_drivers(self):
        ...

    def test(self):
        ...
        # at each step first apply targeted dispatching, then dispatching per solver
        # idle_drivers = self.get_idle_drivers_per_supernode()
        # arriving_drivers = self.get_arriving_drivers_per_supernode()
        # expected_orders = self.get_expected_orders_per_supernode()
        # cars_per_edge = self.get_flow_per_edge()
        # dispatch_action_list = self.get_dispatch_action_list_from_flow(cars_per_edge)
        # # input: dispatch_action_list: a list of <destination, number_of_drivers, reward, time_length>
        # # output: <destination, number_of_drivers, reward, time_length, a list of driver ids assigned>
        # assigned_drivers = self.env.dispatch_drivers(dispatch_action_list)

        # # predict per each subsolver, apply to subworlds, sync with superworld
        # for n in self.superworld.nodes(data=True):
        #     env = n[1]['env']
        #     env.sync(self.env)
        #     action = n[1]['solver'].predict(env.get_observation())
        #     env.step(action)
        #     self.env.sync(env)
        # self.env.apply() # apply all changes after each subworld processes the situation

    def get_dispatch_action_list_from_flow(self, flow_per_edge):
        ...

    def init_subsolvers(self):
        ...
